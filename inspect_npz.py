# import numpy as np

# f = np.load("/data/p_03183/data/pav_algermissen/derived/py_imported/vs_tc_200hz/sub-001_block1.npz")

# print(f.files)
# print(f['eeg'])
# print(f['eeg'].shape)

"""
Converts your matched/preprocessed .npz (eeg, fmri, events, roi_labels,
eeg_channel_names, event_channel_names) into the pickle format that
NeuroBOLT's dataset_maker/get_datasets.py -> prepare_full_dataloader()
expects via --prepro_datapath.

>>> ADJUST THE MARKED SECTIONS BELOW to match your actual data <<<
"""

import numpy as np
import torch
import pickle
import glob
import os

EEG_SFREQ = 200
WINDOW_SEC = 16
WINDOW_SAMPLES = int(WINDOW_SEC * EEG_SFREQ)  # 3200

# =========================================================================
# 1) POINT THIS AT YOUR FILES
# =========================================================================
NPZ_FILES = sorted(glob.glob("/data/p_03183/data/pav_algermissen/derived/py_imported/vs_tc_200hz/sub-001_block1.npz"))   # <-- EDIT: one .npz per subject/scan
OUTPUT_PKL = "/data/p_03183/personal_workspaces/sheker/NeuroBOLT/my_data_seq2one.pkl"       # <-- EDIT: output location

# =========================================================================
# 2) TIME-CHANNEL EXCLUSION
#    You confirmed eeg.shape = (64, 128240) — 64 rows but only 63 real
#    EEG channels once the timestamp row is removed. Set the index below
#    once you see eeg_channel_names (e.g. it might be named 'STI', 'TRIG',
#    'Status', 'Time', or similar rather than a real 10-20 electrode name).
# =========================================================================
TIME_CHANNEL_NAME = None  # <-- EDIT: e.g. "Status" — set to None to skip removal

def strip_time_channel(eeg, ch_names):
    if TIME_CHANNEL_NAME is None:
        return eeg, ch_names
    ch_names = list(ch_names)
    if TIME_CHANNEL_NAME not in ch_names:
        print(f"WARNING: '{TIME_CHANNEL_NAME}' not found in channel names, skipping removal")
        return eeg, ch_names
    idx = ch_names.index(TIME_CHANNEL_NAME)
    eeg = np.delete(eeg, idx, axis=0)
    ch_names = ch_names[:idx] + ch_names[idx+1:]
    return eeg, ch_names

# =========================================================================
# 3) EVENT -> TR ONSET INTERPRETATION
#    EDIT this function once you see the real `events` array. Common cases:
#      a) events is 1D array of integer sample indices marking each TR onset
#      b) events is (n, 3) MNE-style [sample, duration, event_id] triplets,
#         where only rows matching a specific event_id are TR onsets
#      c) events is 1D array of onset TIMES in seconds (needs *EEG_SFREQ)
# =========================================================================
def get_tr_onset_samples(events, event_channel_names):
    events = np.asarray(events)
    if events.ndim == 1:
        # Case (a) or (c) -- check dtype/magnitude to decide
        if np.issubdtype(events.dtype, np.floating):
            return np.round(events * EEG_SFREQ).astype(int)  # case (c)
        return events.astype(int)  # case (a)
    elif events.ndim == 2 and events.shape[1] >= 1:
        # Case (b) -- adjust column index / event_id filter as needed
        return events[:, 0].astype(int)
    else:
        raise ValueError(f"Unrecognized events shape: {events.shape}")

# =========================================================================
# MAIN CONVERSION
# =========================================================================
def process_one_file(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    eeg = d["eeg"]                       # (channels, time)
    fmri = d["fmri"]                     # expect (n_roi, n_TRs)
    events = d["events"]
    event_channel_names = d["event_channel_names"]
    eeg_channel_names = d["eeg_channel_names"]

    eeg, eeg_channel_names = strip_time_channel(eeg, eeg_channel_names)

    tr_onset_samples = get_tr_onset_samples(events, event_channel_names)

    eeg_epochs, fmri_epochs = [], []
    n_trs = fmri.shape[1]
    n_events = len(tr_onset_samples)

    if n_events != n_trs:
        print(f"NOTE: {npz_path} -> {n_events} events vs {n_trs} fMRI TRs "
              f"(mismatch is expected if some events aren't TR triggers, "
              f"or if some TRs are dropped for magnet stabilization)")

    n_pairs = min(n_events, n_trs)
    for i in range(n_pairs):
        end_sample = tr_onset_samples[i]
        start_sample = end_sample - WINDOW_SAMPLES
        if start_sample < 0:
            continue  # not enough preceding EEG for this TR yet
        eeg_epochs.append(eeg[:, start_sample:end_sample])
        fmri_epochs.append(fmri[:, i])

    return np.stack(eeg_epochs), np.stack(fmri_epochs), eeg_channel_names


def main():
    all_eeg, all_fmri = [], []
    ch_names_ref = None

    for f in NPZ_FILES:
        print(f"Processing {f} ...")
        eeg_epochs, fmri_epochs, ch_names = process_one_file(f)
        print(f"  -> {eeg_epochs.shape[0]} epochs, eeg epoch shape {eeg_epochs.shape[1:]}, "
              f"fmri epoch shape {fmri_epochs.shape[1:]}")
        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif list(ch_names) != list(ch_names_ref):
            raise ValueError(f"Channel name mismatch in {f} vs previous files")
        all_eeg.append(eeg_epochs)
        all_fmri.append(fmri_epochs)

    eeg_arr = np.concatenate(all_eeg, axis=0)
    fmri_arr = np.concatenate(all_fmri, axis=0)
    print(f"\nTotal epochs across all files: {eeg_arr.shape[0]}")
    print(f"Final EEG channel list ({len(ch_names_ref)}): {ch_names_ref}")

    # Shuffle-free chronological 80/10/10 split (matches repo's own convention)
    n = len(eeg_arr)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    train_data = (to_tensor(eeg_arr[:train_end]), to_tensor(fmri_arr[:train_end]))
    val_data = (to_tensor(eeg_arr[train_end:val_end]), to_tensor(fmri_arr[train_end:val_end]))
    test_data = (to_tensor(eeg_arr[val_end:]), to_tensor(fmri_arr[val_end:]))

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump([train_data, val_data, test_data, None], f)

    print(f"\nSaved: {OUTPUT_PKL}")
    print(f"Train/Val/Test sizes: {train_end}, {val_end - train_end}, {n - val_end}")
    print(f"\n>>> Save this channel list, you'll need it for main.py's ch_names branch: <<<")
    print(list(ch_names_ref))


if __name__ == "__main__":
    main()

   

