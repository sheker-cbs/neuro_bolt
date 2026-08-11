"""
NeuroBOLT: Resting-state EEG-to-fMRI Synthesis with Multi-dimensional Feature Mapping
By Yamin Li

NeuroBOLT is built upon the LaBraM, BIOT, BEiT-v2, timm, DeiT, and DINO codebases,
we extend our gratitude to the authors for their contributions:

- https://github.com/935963004/LaBraM
- https://github.com/ycq091044/BIOT
- https://github.com/microsoft/unilm/tree/master/beitv2
- https://github.com/rwightman/pytorch-image-models/tree/master/timm
- https://github.com/facebookresearch/deit/
- https://github.com/facebookresearch/dino

Script purpose
--------------
This script trains one NeuroBOLT
model per subject on the "Algermissen" EEG/fMRI dataset, using intra-subject
("intrascan") training. Configuration is done by directly setting attributes
on an `argparse.Namespace` object (rather than parsing CLI flags), so the
script is meant to be run as-is or edited in place for a given experiment.
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
import platform
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
import matplotlib.pyplot as plt
# import wandb


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# Older Windows/Colab path hints, kept for reference but disabled.
#if platform.system() == 'Windows':
#    code_path = 'c:/Users/herzo/Documents/work/DLEEGfMRI/NeuroBOLT/code'
#else:
    # Assuming the code is in /content/NeuroBOLT/code in Colab
#    code_path = '/content/NeuroBOLT/code'

# engine.py / runtime.py / optim_factory.py / arch/ / dataset_maker/ live one
# level up, at models/neuroBOLT/ -- this file is in experiments/ (ADR-0004).
# Resolve that directory relative to this file and add it to sys.path so the
# sibling modules imported below can be found regardless of the working
# directory the script is launched from.
code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_path)

from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from engine import train_one_epoch, evaluate
from runtime import NativeScalerWithGradNormCount as NativeScaler
import runtime as utils
from dataset_maker import get_datasets
from scipy import interpolate
import arch.model

# ---------------------------------------------------------------------------
# Filesystem locations
# ---------------------------------------------------------------------------
# LaBraM pretrained weights to finetune from (read-only backup location)
labram_ckpt_path = "/data/p_03183/personal_workspaces/sheker/NeuroBOLT/checkpoints/labram-base.pth"
# Writable location for NeuroBOLT outputs (checkpoints, logs, plots) — kept
# OUT of the code tree so training artefacts don't pollute source control.
checkpoint_path = "/data/p_03183/personal_workspaces/sheker/NeuroBOLT/checkpoints"

# ---- Algermissen (VS timecourse) dataset ----
# Per-block npz files from import_and_preproc_algermissen_vstc.py
ALGERMISSEN_DATA_ROOT = "/data/p_03183/data/pav_algermissen/derived/py_imported/per_block/for_NeuroBOLT/"
# Subjects to exclude from the training loop (e.g. failed preprocessing / QC).
ALGERMISSEN_SKIP      = {"sub-004", "sub-015", "sub-025"}

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
# `args` is configured here as a plain Namespace (instead of via
# argparse.parse_args()) so every hyperparameter for this experiment is
# visible and editable in one place.
args = argparse.Namespace()

# Update args with platform-aware paths
args.finetune = labram_ckpt_path  # Finetune from LaBraM checkpoint
args.dataset_root = ALGERMISSEN_DATA_ROOT
args.output_dir = str(Path(checkpoint_path))
args.log_dir = str(Path(checkpoint_path) / 'log/neurobolt_algermissen')
args.dataname = 'sub-001'        # Subject id (intra-subject training on Algermissen)

# ---- Weights & Biases logging ----
# One run per subject. The wandb/ folder is written under the data tree
# (never the code tree). Set use_wandb = False to disable.
args.use_wandb     = False
args.wandb_project = "neurobolt_algermissen"
args.wandb_dir     = str(Path(checkpoint_path))
# Same key file every other script in this repo reads (see run_neurobolt_sample_size_sweep.sh,
# models/beira/slurm/*.sh) -- not hardcoded in source.
_wandb_keyfile = "/data/p_03183/.wandb_key"
args.wandb_key = (open(_wandb_keyfile).read().strip()
                  if os.path.exists(_wandb_keyfile) else None)


# ---- Core optimization hyperparameters ----
args.lr = 1e-4                        # Learning rate
args.batch_size = 8                 # Training batch size
args.epochs = 20                      # Number of training epochs
args.drop = 0.3                       # Dropout rate
args.weight_decay = 0.01              # Weight decay


# Set device based on actual CUDA availability
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"                       using device: {args.device}")


# ---- Model / architecture configuration ----
args.model = 'neurobolt_default'      # Model architecture to use
args.nb_eegchan = 63                  # Number of EEG channels
args.train_test_mode = 'intrascan'    # Training mode
args.update_freq = 2                  # Update frequency (gradient accumulation steps)
args.save_ckpt_freq = 5               # Save checkpoint frequency
args.robust_test = None               # Robust evaluation dataset
args.qkv_bias = False                 # Use qkv bias
args.rel_pos_bias = False             # Use relative position bias
args.disable_eval_during_finetuning = False
args.abs_pos_emb = True               # Use absolute position embedding
args.layer_scale_init_value = 0.1     # Layer scale initialization value
args.input_size = 200                 # EEG input size
args.attn_drop_rate = 0.0             # Attention dropout rate
args.drop_path = 0.1                  # Drop path rate
args.model_ema = False                # Use model EMA
args.model_ema_decay = 0.9999         # Model EMA decay
args.model_ema_force_cpu = False      # Model EMA force CPU
args.opt = 'adamw'                    # Optimizer
args.opt_eps = 1e-8                   # Optimizer epsilon
args.opt_betas = None                 # Optimizer betas
args.clip_grad = None                 # Clip gradient norm
args.momentum = 0.9                   # SGD momentum
args.weight_decay_end = None          # Final value of the weight decay
args.layer_decay = 0.65               # Layer decay
args.warmup_lr = 1e-6                 # Warmup learning rate
args.min_lr = 1e-6                    # Lower LR bound
args.warmup_epochs = 5                # Warmup epochs
args.warmup_steps = -1                # Warmup steps
args.smoothing = 0.1                  # Label smoothing
args.reprob = 0.25                    # Random erase prob
args.remode = 'pixel'                 # Random erase mode
args.recount = 1                      # Random erase count
args.resplit = False                  # Random erase resplit
args.model_key = 'model|module'       # Model key
args.model_prefix = ''                # Model prefix
args.model_filter_name = 'gzp'        # Model filter name
args.init_scale = 0.001               # Initialization scale
args.use_mean_pooling = True          # Use mean pooling
args.use_cls = False                  # Use classification
args.disable_weight_decay_on_rel_pos_bias = False
args.seed = 12345                      # Seed
args.resume = ''                      # Resume
args.auto_resume = False              # Auto resume
args.no_auto_resume = False           # No auto resume
args.save_ckpt = True                 # Save checkpoint
args.no_save_ckpt = False             # No save checkpoint
args.start_epoch = 0
args.eval = False                      # Perform evaluation only
args.dist_eval = True                  # Enabling distributed evaluation
args.num_workers = 1                   # Number of workers
args.pin_mem = True                    # Pin memory
args.no_pin_mem = False                # No pin memory
args.world_size = 1                    # Number of distributed processes
args.local_rank = -1                   # Local rank
args.dist_on_itp = False               # Distributed on ITP
args.dist_url = 'env://'               # Distributed URL
args.enable_deepspeed = False          # Enable DeepSpeed

# ---- Dataset / task configuration ----
args.atlas = 'Difumo'                  # Atlas used for extracting ROI
args.nb_roi = 1                        # Number of the output ROI
args.labels_roi = 'VS'                 # fMRI ROI name in str
args.dataset = 'algermissen'           # Dataset (algermissen / VU)
args.prepro_datapath = None            # Path to the preprocessed epoch datasets
args.save_input_tensor = False         # Flag to save the input tensor when do inter-subject training
args.train_test_mode = 'intrascan'     # Intrascan/full_test/full_retainvu/  (overrides the earlier duplicate default)
args.split_index_sheet = './scan_split.xlsx'  # Path to the Excel sheet specifying train-test split indices when do cross-subject training
args.TR = 1.4                          # TR of the fMRI data (Algermissen = 1.4 s)
args.window_sec = 16                   # EEG window length per epoch (s); 16 s x 200 Hz = 3200
args.model_hz = 200                    # Target EEG sampling rate fed to the model (LaBraM = 200 Hz)

# Default 10-20 montage channel names, used as a fallback for datasets other
# than 'algermissen' (Algermissen's own channel names are read from its npz
# files and override this list later in get_dataset()).
args.ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
       'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4',
       'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'FT9',
       'FT10', 'PO9', 'PO10', 'CPz', 'POz']

def get_models(args):
    """Instantiate the model architecture named by `args.model`.

    Currently only 'neurobolt_default' is supported: it builds a
    NeuroBOLTransformer via timm's `create_model` registry using
    hyperparameters pulled off `args`. If `args.EEG_length` is set, it is
    passed through explicitly so the model's internal MSSEncoder shape can
    never silently drift out of sync with `window_sec` / `model_hz`.

    Args:
        args: argparse.Namespace with model hyperparameters.

    Returns:
        The instantiated model (nn.Module), or None if `args.model` doesn't
        match a known branch (falls through silently otherwise).
    """
    if args.model == "neurobolt_default":
        model_kwargs = dict(
            EEG_channel=len(args.ch_names),
            num_roi=args.nb_roi,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
        )
        # EEG_length defaults to 3200 (=16s x 200Hz) inside NeuroBOLTransformer
        # if not passed -- harmless for existing callers (window_sec/model_hz
        # already match that default), but pass it explicitly whenever a
        # caller sets it so window_sec/model_hz changes can never silently
        # mismatch the model's internal MSSEncoder shape.
        eeg_length = getattr(args, "EEG_length", None)
        if eeg_length is not None:
            model_kwargs["EEG_length"] = eeg_length
        model = create_model(args.model, **model_kwargs)
    return model


def get_dataset(args):
    """Load and split the train/test/val datasets for `args.dataset`.

    Two dataset families are supported:
      - 'algermissen': continuous per-block npz recordings converted into
        NeuroBOLT seq2one epochs via `prepare_algermissen_onesub_dataloader`.
        Channel names are read from the npz files themselves.
      - anything else (e.g. 'VU'): loaded via the generic
        `prepare_onesub_dataloader`, using `args.ch_names` as the channel list.

    Args:
        args: argparse.Namespace with dataset configuration
            (dataset_root, dataname, model_hz, window_sec, TR, ch_names, ...).

    Returns:
        Tuple of (train_dataset, test_dataset, val_dataset, ch_names,
        metrics, spec_chan_ind), where `metrics` is the fixed list
        ["mse", "corr"] and `spec_chan_ind` is currently always None.
    """
    spec_chan_ind = None
    metrics = ["mse", "corr"]

    if args.dataset == 'algermissen':
        # Continuous per-block Algermissen data -> NeuroBOLT seq2one epochs.
        # ch_names are read from the npz (63 channels, all in the 10-20 montage).
        train_dataset, test_dataset, val_dataset, ch_names = \
            get_datasets.prepare_algermissen_onesub_dataloader(
                args.dataset_root,
                args.dataname,
                model_hz=args.model_hz,
                window_sec=args.window_sec,
                tr=args.TR,
            )
    else:
        ch_names = args.ch_names
        # Load and split data
        train_dataset, test_dataset, val_dataset = get_datasets.prepare_onesub_dataloader(
            args.dataset_root,
            args.dataname,
            ch_names)

    return train_dataset, test_dataset, val_dataset, ch_names, metrics, spec_chan_ind

def main(args, ds_init):
    """Run one full training job (all epochs) for a single subject.

    This sets up distributed/GPU context, builds the datasets, dataloaders,
    model, optimizer, and LR/WD schedules, optionally resumes from a
    checkpoint, then either runs evaluation only (`args.eval`) or trains for
    `args.epochs` epochs, logging metrics (stdout, TensorBoard, W&B, and a
    JSON-lines log file) and periodically saving prediction plots and
    checkpoints whenever validation MSE improves. Intended to be called once
    per subject from the `__main__` block below, with `args.dataname` set to
    the current subject id before each call.

    Args:
        args: argparse.Namespace with the full experiment configuration.
        ds_init: DeepSpeed initializer function, or None if DeepSpeed is
            disabled (`args.enable_deepspeed == False`).
    """
    if torch.cuda.device_count() > 1:
        # Set distributed training parameters for multi-GPU
        args.distributed = True
        args.world_size = torch.cuda.device_count()  # Automatically detect number of GPUs
        args.dist_url = 'env://'
        args.dist_backend = 'nccl'
        utils.init_distributed_mode(args)
    else:
        args.distributed = False
        args.gpu = 0
        args.rank = 0
        args.world_size = 1
        print('Not using distributed mode')

    if ds_init is not None:
        utils.create_ds_config(args)

    #print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    dataset_train, dataset_test, dataset_val, ch_names, metrics, spec_chan_ind = get_dataset(args)
    ch_names = [c.upper() for c in ch_names] #added to make channelnames uppercase
    # Keep args.ch_names in sync so the model is built with the dataset's channels
    args.ch_names = ch_names

    print("\nDataset shapes:")
    print(f"Train dataset - EEG: {dataset_train.tensors[0].shape}, fMRI: {dataset_train.tensors[1].shape}")
    print(f"Test dataset  - EEG: {dataset_test.tensors[0].shape}, fMRI: {dataset_test.tensors[1].shape}")
    print(f"Val dataset   - EEG: {dataset_val.tensors[0].shape}, fMRI: {dataset_val.tensors[1].shape}")

    if True:  # args.distributed:
        # NOTE: this branch always runs regardless of args.distributed (the
        # condition is hardcoded to True); the DistributedSampler classes
        # used below work fine in single-process mode too (world_size=1).
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Weights & Biases: one run per subject (main() is called once per subject).
    use_wandb = getattr(args, "use_wandb", False) and global_rank == 0
    if use_wandb:
        if getattr(args, "wandb_key", None):
            wandb.login(key=args.wandb_key)
        os.makedirs(args.wandb_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.dataname,
            group=args.dataset,
            dir=args.wandb_dir,
            reinit=True,
            config={k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))},
        )
        wandb.define_metric("val/corr", summary="max")
        wandb.define_metric("test/corr", summary="max")
        wandb.define_metric("val/mse", summary="min")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        if args.dataset == "VU" and args.train_test_mode == 'full_test':
            bs_val = int(len(dataset_val) / 5)
            bs_test = int(len(dataset_test) / 6)
            # modify to adapt other dataset with different number of scans in test/val for printing metrics
        else:
            bs_val = len(dataset_val)
            bs_test = len(dataset_test)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=bs_val,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                # batch_size=int(1.5 * args.batch_size),
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=bs_test,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    if args.model == "neurobolt_default":
        patch_size = model.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size

    if args.finetune:
        # Load a pretrained (LaBraM) checkpoint to finetune from, either
        # from a URL or a local path.
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        # Try each candidate key (e.g. 'model', 'module') to find the actual
        # state_dict inside the checkpoint file.
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            # Strip a 'student.' prefix (as used by DINO/BEiT-style teacher-
            # student checkpoints) from matching keys; keys without that
            # prefix are dropped entirely.
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        # Drop the classification head weights if their shape doesn't match
        # the current model (e.g. different num_roi / output dimension).
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        # model.load_state_dict(checkpoint['model'])

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 and args.model == "neurobolt_default":
        # Layer-wise LR decay: earlier transformer layers get progressively
        # smaller learning rates than later ones.
        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.model == "neurobolt_default":
        skip_weight_decay_list = model.no_weight_decay()
    else:
        skip_weight_decay_list = None

    if args.disable_weight_decay_on_rel_pos_bias and args.model == "neurobolt_default":
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    # Cosine LR schedule with warmup, stepped once per training iteration
    # (not once per epoch) for finer-grained control.
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    criterion = torch.nn.MSELoss()
    print("criterion = %s" % str(criterion))

    # Resume from a checkpoint if one is configured / auto-detected.
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        # Evaluation-only mode: run over each test dataloader, report
        # aggregate MSE/correlation stats, and exit without training.
        test_mse = []
        test_corr = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=(args.nb_roi == 1))
            test_mse.append(test_stats['mse'])
            test_corr.append(test_stats['corr'])
        print(f"======MSE: {np.mean(test_mse)} {np.std(test_mse)}, corr: {np.mean(test_corr)} {np.std(test_corr)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_corr = 0.0
    max_corr_test = 0.0
    min_mse = 10
    min_mse_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        if args.model != "neurobolt_default":
            ch_names = None
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_roi == 1, spec_chan=spec_chan_ind
        )

        # todo: the code below saves the model regularly. uncomment if you want to save regularly
        # if args.output_dir and args.save_ckpt:
        #     utils.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                 is_binary=args.nb_roi == 1)
            print(f"Correlation of the network on the {len(dataset_val)} val EEG: {val_stats['corr']:.2f}")
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=args.nb_roi == 1)
            print(f"Correlation of the network on the {len(dataset_test)} test EEG: {test_stats['corr']:.2f}")
            train_corr = train_stats["corr"]
            val_mse = val_stats["mse"]
            val_corr = val_stats["corr"]
            test_mse = test_stats["mse"]
            test_corr = test_stats["corr"]
            epochname = f"{epoch}" + f"-trcorr{train_corr:.2f}-valmse{val_mse:.4f}-valcorr{val_corr:.4f}-testmse{test_mse:.4f}-testcorr{test_corr:.4f}"
            if min_mse > val_stats["mse"]:
                # New best validation MSE: track the corresponding test
                # correlation, save a prediction-vs-truth plot, and
                # checkpoint the model (this is a "best model so far" save,
                # distinct from the periodic save_ckpt_freq saving above,
                # which is currently commented out).
                min_mse = val_stats["mse"]
                max_corr_test = test_stats["corr"]
                max_test_corr_glb = test_stats["corr"]

                # Only create plots on rank 0
                if utils.get_rank() == 0:
                    #PLOT
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                    # Convert list of tensors to numpy arrays
                    if isinstance(val_stats['true'], list):
                        val_true = torch.cat(val_stats['true'], dim=0).cpu().numpy()
                        val_pred = torch.cat(val_stats['predictions'], dim=0).cpu().numpy()
                        test_true = torch.cat(test_stats['true'], dim=0).cpu().numpy()
                        test_pred = torch.cat(test_stats['predictions'], dim=0).cpu().numpy()
                    else:
                        val_true = val_stats['true']
                        val_pred = val_stats['predictions']
                        test_true = test_stats['true']
                        test_pred = test_stats['predictions']

                    # Ensure arrays are 1D
                    val_true = np.squeeze(val_true)
                    val_pred = np.squeeze(val_pred)
                    test_true = np.squeeze(test_true)
                    test_pred = np.squeeze(test_pred)

                    # Validation plot
                    ax1.plot(val_true, 'b-', label='True', alpha=0.7)
                    ax1.plot(val_pred, 'r-', label='Predicted', alpha=0.7)
                    ax1.set_title('Final Validation Set Predictions')
                    ax1.set_xlabel('Time Points')
                    ax1.set_ylabel('Value')
                    ax1.legend()
                    val_corr = val_stats['corr']
                    ax1.text(0.02, 0.98, f'Correlation: {val_corr:.3f}',
                            transform=ax1.transAxes, verticalalignment='top')

                    # Test plot
                    ax2.plot(test_true, 'b-', label='True', alpha=0.7)
                    ax2.plot(test_pred, 'r-', label='Predicted', alpha=0.7)
                    ax2.set_title('Final Test Set Predictions')
                    ax2.set_xlabel('Time Points')
                    ax2.set_ylabel('Value')
                    ax2.legend()
                    test_corr = test_stats['corr']
                    ax2.text(0.02, 0.98, f'Correlation: {test_corr:.3f}',
                            transform=ax2.transAxes, verticalalignment='top')

                    plt.tight_layout()
                    save_path = os.path.join(args.output_dir, f'predictions_vs_true_epoch{epoch}.png')
                    plt.savefig(save_path)
                    if use_wandb:
                        wandb.log({"predictions/vs_true": wandb.Image(fig)}, step=epoch)
                    plt.close()

                if args.output_dir and args.save_ckpt and test_corr > 0:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epochname, model_ema=model_ema)

            print(f'Min mse val: {min_mse:.2f}, max corr test: {max_corr_test:.2f}')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'corr':
                        log_writer.update(correlation=float(value), head="val", step=epoch)
                    elif key == 'mse':
                        log_writer.update(MSE=float(value), head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=float(value), head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'corr':
                        log_writer.update(correlation=float(value), head="test", step=epoch)
                    elif key == 'mse':
                        log_writer.update(MSE=float(value), head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=float(value), head="test", step=epoch)

            # Define which keys are scalar values (not arrays)
            scalar_keys = ['mse', 'corr', 'loss']

            log_stats = {
                **{f'train_{k}': float(v) for k, v in train_stats.items()},
                **{f'val_{k}': float(v) for k, v in val_stats.items() if k in scalar_keys},
                **{f'test_{k}': float(v) for k, v in test_stats.items() if k in scalar_keys},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        else:
            log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if use_wandb:
            # Reshape train_/val_/test_ prefixes into wandb "train/…" groups.
            wandb_log = {}
            for k, v in log_stats.items():
                if k in ("epoch", "n_parameters"):
                    continue
                for p in ("train_", "val_", "test_"):
                    if k.startswith(p):
                        k = p[:-1] + "/" + k[len(p):]
                        break
                wandb_log[k] = v
            wandb.log(wandb_log, step=epoch)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            # Append this epoch's stats as one JSON line for easy later parsing.
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # After training is complete, do final evaluation and plotting
    print("Final evaluation and plotting...")
    val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names,
                        metrics=metrics, is_binary=args.nb_roi == 1)
    test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names,
                         metrics=metrics, is_binary=args.nb_roi == 1)

    if use_wandb:
        wandb.log({
            "final/val_corr": float(val_stats["corr"]),
            "final/val_mse":  float(val_stats["mse"]),
            "final/test_corr": float(test_stats["corr"]),
            "final/test_mse":  float(test_stats["mse"]),
        })
        wandb.finish()



if __name__ == '__main__':
    # Remove get_args() call since we defined args directly above
    ds_init = None  # Since we're not using DeepSpeed

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == 'algermissen':
        # Intra-subject training: fit one NeuroBOLT model per subject.
        # Discover subject ids from the per-block npz filenames
        # (e.g. "sub-001_block1.npz" -> "sub-001"), then drop any subjects
        # listed in ALGERMISSEN_SKIP.
        all_files = os.listdir(args.dataset_root)
        subjects = sorted({
            f.split('_block')[0] for f in all_files
            if f.endswith('.npz') and '_block' in f
        })
        subjects = [s for s in subjects if s not in ALGERMISSEN_SKIP]
        print(f"Algermissen subjects to train: {subjects}")

        # Train one full model per subject, reusing the same `args` object
        # (only `args.dataname` changes between iterations).
        for subject in subjects:
            print(f"\n{'='*60}\nStarting NeuroBOLT training for {subject}\n{'='*60}\n")
            args.dataname = subject
            main(args, ds_init)
            print(f"Completed training for {subject}")
        print("\nAll subjects completed!")
    else:
        main(args, ds_init)