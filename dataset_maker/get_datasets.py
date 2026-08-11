"""
dataset_maker.get_datasets

Build Algermissen continuous per-block npz recordings into
`torch.utils.data.TensorDataset` train/val/test splits for NeuroBOLT
(seq2one: EEG window -> single fMRI ROI value).
"""

import os

import numpy as np
import torch


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
    blocks = []
    ch_names = None
    for b in range(1, n_runs + 1):
        path = os.path.join(dataset_root, f"{subject}_block{b}.npz")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping block {b}")
            continue
        data = np.load(path, allow_pickle=True)
        chs = list(data["eeg_channel_names"])
        eeg_idx = [i for i, c in enumerate(chs) if c != "time"]
        time_idx = chs.index("time")
        block_ch = [str(chs[i]) for i in eeg_idx]
        if ch_names is None:
            ch_names = block_ch
        elif block_ch != ch_names:
            raise RuntimeError(f"Channel order mismatch in {path}")
        eeg = np.asarray(data["eeg"][eeg_idx, :], dtype=np.float64)
        tvec = np.asarray(data["eeg"][time_idx, :], dtype=np.float64)
        vs = np.asarray(data["fmri"][1, :], dtype=np.float64)  # row 1 = VS
        blocks.append((eeg, tvec, vs))

    if not blocks:
        raise RuntimeError(f"No block files found for {subject} in {dataset_root}")
    return blocks, ch_names


def prepare_algermissen_onesub_dataloader(dataset_root, subject, model_hz=200,
                                          window_sec=16, tr=1.4, n_runs=6,
                                          val_frac=0.1, test_frac=0.1,
                                          eeg_to_uv=True):
    """Build NeuroBOLT seq2one datasets from continuous Algermissen recordings.

    For each fMRI volume time `t`, takes the preceding `window_sec` seconds of
    EEG as input and the VS value at `t` as target. Time-ordered 80/10/10 split
    with anti-leakage gaps; VS z-scored on train stats.

    Returns:
        dataset_train, dataset_test, dataset_val, ch_names
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
            eeg = resample(eeg, up=model_hz, down=src_hz, axis=1)
            n_new = eeg.shape[1]
            vs = np.interp(np.linspace(0, len(vs) - 1, n_new),
                           np.arange(len(vs)), vs)

        if eeg_to_uv:
            eeg = eeg * 1e6  # Volts -> microvolts

        n_samp = eeg.shape[1]
        k0 = int(np.ceil(window_sec / tr))
        k = k0
        while True:
            end = int(round(k * tr * model_hz))
            if end > n_samp:
                break
            epochs_eeg.append(eeg[:, end - win:end].astype(np.float32))
            epochs_vs.append(np.float32(vs[min(end, n_samp - 1)]))
            k += 1

    eeg_arr = np.stack(epochs_eeg)  # [M, C, win]
    vs_arr = np.asarray(epochs_vs, np.float32)  # [M]
    M = eeg_arr.shape[0]
    print(f"  {subject}: built {M} EEG->VS epochs "
          f"(EEG {eeg_arr.shape[1]}ch x {eeg_arr.shape[2]} samp @ {model_hz}Hz)")

    gap = int(np.ceil(window_sec / tr))
    train_end = int((1.0 - val_frac - test_frac) * M)
    val_start = train_end + gap
    val_end = int((1.0 - test_frac) * M)
    test_start = val_end + gap

    idx = dict(
        train=slice(0, train_end),
        val=slice(val_start, val_end),
        test=slice(test_start, M),
    )

    vs_mean = vs_arr[idx["train"]].mean()
    vs_std = vs_arr[idx["train"]].std() + 1e-8
    vs_norm = (vs_arr - vs_mean) / vs_std

    def _ds(sl):
        eeg_t = torch.tensor(eeg_arr[sl], dtype=torch.float32)
        vs_t = torch.tensor(vs_norm[sl], dtype=torch.float32)
        return torch.utils.data.TensorDataset(eeg_t, vs_t)

    dataset_train = _ds(idx["train"])
    dataset_val = _ds(idx["val"])
    dataset_test = _ds(idx["test"])
    print(f"  split sizes — train {len(dataset_train)}, "
          f"val {len(dataset_val)}, test {len(dataset_test)}")

    return dataset_train, dataset_test, dataset_val, ch_names
