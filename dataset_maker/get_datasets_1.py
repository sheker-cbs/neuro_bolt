import mne
import os
import json

import numpy as np
import pandas as pd
import re

from dataset_maker import preproc
from scipy.signal import butter, filtfilt
import torch
import math
import pickle


def convert_to_tensor(data):
    """Convert numpy array to torch tensor, handling object arrays."""
    # First convert object array to contiguous float32 array
    if data.dtype == np.dtype('O'):
        data = np.stack([np.asarray(x, dtype=np.float32) for x in data])
    
    # Then convert to tensor
    return torch.tensor(data, dtype=torch.float32)


def load_npz_dataset(path_to_dataset, ch_names):
    """Load EEG and fMRI data from .npz file."""
    data = np.load(path_to_dataset, allow_pickle=True)
    
    # Print data structure for debugging
    print("Available arrays in .npz file:", list(data.files))
    
    eeg_data = data['eeg']  
    eeg_data = eeg_data[..., :3200]  # Crop EEG data to 3200 samples
    
    #eeg_channel_names = data['eeg_channels']  # List of all EEG channel names
    #channel_inds = [eeg_channel_names.tolist().index(ch) for ch in ch_names if ch in eeg_channel_names] # Find indices of the channels in eeg_channel_names that match config_channels
    #eeg_data = eeg_data[channel_inds, :]         # Filter the EEG data to retain only the specified channels
    #eeg_data = eeg_data.T
    fmri_data = data['fmri']
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"fMRI data shape: {fmri_data.shape}")
    
    return eeg_data, fmri_data

def _infer_hz(time_row):
    """Infer sampling rate (Hz) from the per-sample time axis stored in the npz."""
    dt = np.median(np.diff(np.asarray(time_row, dtype=np.float64)))
    return int(round(1.0 / dt))


def _load_algermissen_blocks(dataset_root, subject, n_runs):
    """Load the per-block Algermissen npz files for one subject.

    Each block file (produced by import_and_preproc_algermissen_vstc.py) holds
    continuous data:
        eeg  : [n_channels + 1, n_samples]  (one row named 'time')
        fmri : [2, n_samples]               (row 0 = time, row 1 = VS)

    Returns a list of (eeg[C, T], time[T], vs[T]) tuples (one per available
    block) plus the ordered EEG channel names (the 'time' row removed).
    """
    blocks   = []
    ch_names = None
    for b in range(1, n_runs + 1):
        path = os.path.join(dataset_root, f"{subject}_block{b}.npz")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping block {b}")
            continue
        data = np.load(path, allow_pickle=True)
        chs  = list(data["eeg_channel_names"])
        eeg_idx  = [i for i, c in enumerate(chs) if c != "time"]
        time_idx = chs.index("time")
        block_ch = [str(chs[i]) for i in eeg_idx]
        if ch_names is None:
            ch_names = block_ch
        elif block_ch != ch_names:
            raise RuntimeError(f"Channel order mismatch in {path}")
        eeg  = np.asarray(data["eeg"][eeg_idx, :],  dtype=np.float64)
        tvec = np.asarray(data["eeg"][time_idx, :], dtype=np.float64)
        vs   = np.asarray(data["fmri"][1, :],       dtype=np.float64)  # row 1 = VS
        blocks.append((eeg, tvec, vs))

    if not blocks:
        raise RuntimeError(f"No block files found for {subject} in {dataset_root}")
    return blocks, ch_names


def prepare_algermissen_onesub_dataloader(dataset_root, subject, model_hz=200,
                                          window_sec=16, tr=1.4, n_runs=6,
                                          val_frac=0.1, test_frac=0.1,
                                          eeg_to_uv=True):
    """Build NeuroBOLT seq2one datasets from continuous Algermissen recordings.

    NeuroBOLT expects one (EEG epoch -> single fMRI value) pair per sample, with
    the EEG epoch reshaped downstream into 1 s / `model_hz`-sample patches
    (LaBraM style). This adapter epochs the continuous per-block data onto the
    fMRI TR grid: for each fMRI volume time `t` it takes the preceding
    `window_sec` seconds of EEG as the input and the VS value at `t` as target.

    LaBraM/NeuroBOLT is pretrained on 200 Hz EEG. The Algermissen files are
    100 Hz, so by default EEG is resampled to `model_hz`. NOTE: upsampling
    100 -> 200 Hz is a stop-gap; for best transfer regenerate the npz files at
    200 Hz by setting TARGET_SAMPLERATE = 200 in
    import_and_preproc_algermissen_vstc.py (it decimates from native 1000 Hz).

    Splitting follows NeuroBOLT's time-based scheme: epochs are kept in temporal
    order (block 1 .. block n) and cut 80/10/10, with a gap of one window's
    worth of epochs between splits so overlapping EEG windows never leak across
    the boundary. fMRI targets are z-scored using train statistics; EEG is left
    in microvolts (the engine divides by 100, matching LaBraM's input scale).

    Returns: dataset_train, dataset_test, dataset_val, ch_names
    """
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    blocks, ch_names = _load_algermissen_blocks(dataset_root, subject, n_runs)

    win = int(round(window_sec * model_hz))
    if win % 200 != 0:
        raise ValueError(f"EEG window ({win} samples) must be divisible by the "
                         f"200-sample patch size; adjust window_sec/model_hz.")

    epochs_eeg, epochs_vs = [], []
    for eeg, tvec, vs in blocks:
        src_hz = _infer_hz(tvec)
        if src_hz != model_hz:
            from mne.filter import resample
            if model_hz > src_hz:
                print(f"  NOTE: upsampling EEG {src_hz} -> {model_hz} Hz "
                      f"(consider regenerating the npz at {model_hz} Hz)")
            eeg   = resample(eeg, up=model_hz, down=src_hz, axis=1)
            n_new = eeg.shape[1]
            # keep VS on the same (resampled) time grid via linear interpolation
            vs    = np.interp(np.linspace(0, len(vs) - 1, n_new),
                              np.arange(len(vs)), vs)

        if eeg_to_uv:
            eeg = eeg * 1e6  # Volts -> microvolts

        n_samp = eeg.shape[1]
        # target fMRI volume times: on the TR grid, once a full EEG window fits
        k0 = int(np.ceil(window_sec / tr))
        k  = k0
        while True:
            end = int(round(k * tr * model_hz))
            if end > n_samp:
                break
            epochs_eeg.append(eeg[:, end - win:end].astype(np.float32))
            epochs_vs.append(np.float32(vs[min(end, n_samp - 1)]))
            k += 1

    eeg_arr = np.stack(epochs_eeg)            # [M, C, win]
    vs_arr  = np.asarray(epochs_vs, np.float32)  # [M]
    M       = eeg_arr.shape[0]
    print(f"  {subject}: built {M} EEG->VS epochs "
          f"(EEG {eeg_arr.shape[1]}ch x {eeg_arr.shape[2]} samp @ {model_hz}Hz)")

    # ── time-ordered 80/10/10 split with anti-leakage gaps ────────────────────
    gap        = int(np.ceil(window_sec / tr))  # epochs whose windows may overlap
    train_end  = int((1.0 - val_frac - test_frac) * M)
    val_start  = train_end + gap
    val_end    = int((1.0 - test_frac) * M)
    test_start = val_end + gap

    idx = dict(
        train=slice(0, train_end),
        val=slice(val_start, val_end),
        test=slice(test_start, M),
    )

    # z-score fMRI targets with train statistics only
    vs_mean = vs_arr[idx["train"]].mean()
    vs_std  = vs_arr[idx["train"]].std() + 1e-8
    vs_norm = (vs_arr - vs_mean) / vs_std

    def _ds(sl):
        eeg_t = torch.tensor(eeg_arr[sl], dtype=torch.float32)
        vs_t  = torch.tensor(vs_norm[sl], dtype=torch.float32)  # 1-D [n]
        return torch.utils.data.TensorDataset(eeg_t, vs_t)

    dataset_train = _ds(idx["train"])
    dataset_val   = _ds(idx["val"])
    dataset_test  = _ds(idx["test"])
    print(f"  split sizes — train {len(dataset_train)}, "
          f"val {len(dataset_val)}, test {len(dataset_test)}")

    return dataset_train, dataset_test, dataset_val, ch_names


def prepare_onesub_dataloader(dataset_root, dataname, ch_names):
    # Set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load data from .npz file using args parameters
    eeg_data, fmri_data = load_npz_dataset(os.path.join(dataset_root, dataname), ch_names)
    #breakpoint()
    
    # Split data into train, val, test
    total_samples = len(eeg_data)
    traincrop = int(0.8 * total_samples)
    valcrop = int(0.1 * total_samples) + traincrop
    
    eeg_train = eeg_data[:traincrop]
    fmri_train = fmri_data[:traincrop]

    # consider the auto-correlaiton of fMRI, preventing data leakage
    tmin = -16
    tr = 2
    t_overlap = 20 if abs(tmin) <= 20 else abs(tmin)  # Length of HRF consideration
    N_overlap = math.ceil(t_overlap / tr)

    traincrop += N_overlap
    eeg_val = eeg_data[traincrop:valcrop]
    fmri_val = fmri_data[traincrop:valcrop]

    print(f"Validation split shapes:")
    print(f"eeg_val shape: {eeg_val.shape}")
    print(f"fmri_val shape: {fmri_val.shape}")
        
    valcrop += N_overlap
    eeg_test = eeg_data[valcrop:]
    fmri_test = fmri_data[valcrop:]

    eeg_train_tensor = convert_to_tensor(eeg_train)
    eeg_val_tensor = convert_to_tensor(eeg_val)
    eeg_test_tensor = convert_to_tensor(eeg_test)

    fmri_train_tensor = convert_to_tensor(fmri_train)
    fmri_val_tensor = convert_to_tensor(fmri_val)
    fmri_test_tensor = convert_to_tensor(fmri_test)

    dataset_train = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
    dataset_val = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
    dataset_test = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)
    
    return dataset_train, dataset_test, dataset_val

