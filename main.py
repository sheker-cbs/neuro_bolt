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
This script trains one NeuroBOLT model per subject on the Algermissen EEG/fMRI
dataset (intra-subject / "intrascan"). Configuration is set on an
`argparse.Namespace` in-place (no CLI parsing).
"""

import argparse
import datetime
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

# Resolve sibling modules (engine, runtime, arch, …) regardless of cwd.
code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_path)

from optim_factory import create_optimizer, LayerDecayValueAssigner
from engine import train_one_epoch, evaluate
from runtime import NativeScalerWithGradNormCount as NativeScaler
import runtime as utils
from dataset_maker import get_datasets
# import arch.model  
import models.model

# ---------------------------------------------------------------------------
# Filesystem locations
# ---------------------------------------------------------------------------
labram_ckpt_path = "/data/p_03183/personal_workspaces/sheker/NeuroBOLT/checkpoints/labram-base.pth"
checkpoint_path = "/data/p_03183/personal_workspaces/sheker/NeuroBOLT/checkpoints"

ALGERMISSEN_DATA_ROOT = "/data/p_03183/data/pav_algermissen/derived/py_imported/per_block/for_NeuroBOLT/"
ALGERMISSEN_SKIP = {"sub-004", "sub-015", "sub-025"}

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
args = argparse.Namespace()

args.finetune = labram_ckpt_path
args.dataset_root = ALGERMISSEN_DATA_ROOT
# args.output_dir = str(Path(checkpoint_path))
# args.log_dir = str(Path(checkpoint_path) / 'log/neurobolt_algermissen')
args.output_dir = str(Path(checkpoint_path) / 'attn')
args.log_dir = str(Path(checkpoint_path) / 'log/neurobolt_algermissen_attn')
args.dataname = 'sub-001'

args.lr = 1e-4
args.batch_size = 2
args.epochs = 20
args.drop = 0.3
args.weight_decay = 0.01

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"                       using device: {args.device}")

args.model = 'neurobolt_default'
args.update_freq = 4
args.qkv_bias = False
args.rel_pos_bias = False
args.abs_pos_emb = True
args.layer_scale_init_value = 0.1
args.input_size = 200
args.attn_drop_rate = 0.0
args.drop_path = 0.1
args.model_ema = False  # kept so runtime.auto_load_model / save_model see the flag
args.opt = 'adamw'
args.opt_eps = 1e-8
args.opt_betas = None
args.clip_grad = None
args.weight_decay_end = None
args.layer_decay = 0.65
args.warmup_lr = 1e-6
args.min_lr = 1e-6
args.warmup_epochs = 5
args.warmup_steps = -1
args.model_key = 'model|module'
args.model_prefix = ''
args.model_filter_name = 'gzp'
args.init_scale = 0.001
args.use_mean_pooling = True
args.disable_weight_decay_on_rel_pos_bias = False
args.seed = 12345
args.resume = ''
args.auto_resume = False
args.save_ckpt = True
args.start_epoch = 0
args.dist_eval = True
args.num_workers = 1
args.pin_mem = True
args.world_size = 1
args.local_rank = -1
args.dist_on_itp = False
args.dist_url = 'env://'

args.nb_roi = 1
args.labels_roi = 'VS'
args.dataset = 'algermissen'
args.train_test_mode = 'intrascan'
args.TR = 1.4
args.window_sec = 16
args.model_hz = 200
args.ch_names = []  # filled from Algermissen npz in get_dataset / main


def get_models(args):
    """Instantiate neurobolt_default via timm's create_model registry."""
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
    eeg_length = getattr(args, "EEG_length", None)
    if eeg_length is not None:
        model_kwargs["EEG_length"] = eeg_length
    return create_model(args.model, **model_kwargs)


def get_dataset(args):
    """Load Algermissen train/test/val TensorDatasets for one subject."""
    metrics = ["mse", "corr"]
    train_dataset, test_dataset, val_dataset, ch_names = \
        get_datasets.prepare_algermissen_onesub_dataloader(
            args.dataset_root,
            args.dataname,
            model_hz=args.model_hz,
            window_sec=args.window_sec,
            tr=args.TR,
        )
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args):
    """Run one full training job (all epochs) for a single subject."""
    if torch.cuda.device_count() > 1:
        args.distributed = True
        args.world_size = torch.cuda.device_count()
        args.dist_url = 'env://'
        args.dist_backend = 'nccl'
        utils.init_distributed_mode(args)
    else:
        args.distributed = False
        args.gpu = 0
        args.rank = 0
        args.world_size = 1
        print('Not using distributed mode')

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    ch_names = [c.upper() for c in ch_names]
    args.ch_names = ch_names

    print("\nDataset shapes:")
    print(f"Train dataset - EEG: {dataset_train.tensors[0].shape}, fMRI: {dataset_train.tensors[1].shape}")
    print(f"Test dataset  - EEG: {dataset_test.tensors[0].shape}, fMRI: {dataset_test.tensors[1].shape}")
    print(f"Val dataset   - EEG: {dataset_val.tensors[0].shape}, fMRI: {dataset_val.tensors[1].shape}")

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
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    bs_val = args.batch_size
    bs_test = args.batch_size
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=bs_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=bs_test,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
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

    if args.layer_decay < 1.0:
        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
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

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_corr_test = 0.0
    min_mse = 10
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_roi == 1, spec_chan=None
        )

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
        epochname = (
            f"{epoch}-trcorr{train_corr:.2f}-valmse{val_mse:.4f}"
            f"-valcorr{val_corr:.4f}-testmse{test_mse:.4f}-testcorr{test_corr:.4f}"
        )
        if min_mse > val_stats["mse"]:
            min_mse = val_stats["mse"]
            max_corr_test = test_stats["corr"]

            if utils.get_rank() == 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
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

                val_true = np.squeeze(val_true)
                val_pred = np.squeeze(val_pred)
                test_true = np.squeeze(test_true)
                test_pred = np.squeeze(test_pred)

                ax1.plot(val_true, 'b-', label='True', alpha=0.7)
                ax1.plot(val_pred, 'r-', label='Predicted', alpha=0.7)
                ax1.set_title('Final Validation Set Predictions')
                ax1.set_xlabel('Time Points')
                ax1.set_ylabel('Value')
                ax1.legend()
                val_corr = val_stats['corr']
                ax1.text(0.02, 0.98, f'Correlation: {val_corr:.3f}',
                         transform=ax1.transAxes, verticalalignment='top')

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

        scalar_keys = ['mse', 'corr', 'loss']
        log_stats = {
            **{f'train_{k}': float(v) for k, v in train_stats.items()},
            **{f'val_{k}': float(v) for k, v in val_stats.items() if k in scalar_keys},
            **{f'test_{k}': float(v) for k, v in test_stats.items() if k in scalar_keys},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print("Final evaluation and plotting...")
    evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names,
             metrics=metrics, is_binary=args.nb_roi == 1)
    evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names,
             metrics=metrics, is_binary=args.nb_roi == 1)


if __name__ == '__main__':
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # all_files = os.listdir(args.dataset_root)
    # subjects = sorted({
    #     f.split('_block')[0] for f in all_files
    #     if f.endswith('.npz') and '_block' in f
    # })
    # subjects = [s for s in subjects if s not in ALGERMISSEN_SKIP]
    # print(f"Algermissen subjects to train: {subjects}")


    checkpoint_root = checkpoint_path
    Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
    # ----- LIVE DEFAULT: Algermissen per-subject intrascan -----
    # SLURM array / NEUROBOLT_SUBJECT → one subject; otherwise loop all.
    if args.dataset == 'algermissen' and args.train_test_mode == 'intrascan':
        all_subjects = list_algermissen_subjects(args.dataset_root)
        subjects = select_subjects(all_subjects)
        print(f"Algermissen subject list ({len(all_subjects)}): {all_subjects}")
        print(f"This job will train: {subjects}")
        for i, s in enumerate(all_subjects):
            print(f"  array task {i} -> {s}")

    # for subject in subjects:
    #     print(f"\n{'='*60}\nStarting NeuroBOLT training for {subject}\n{'='*60}\n")
    #     args.dataname = subject
    #     main(args)
    #     print(f"Completed training for {subject}")
    # print("\nAll subjects completed!")

    for subject in subjects:
            # Skip-completed (not live). Copy into this loop if re-running a
            # full all-subjects job on `short` after a timeout:
            # hits = list(Path(checkpoint_root).glob(
            #     f"runs/*_{subject}/{subject}-VS_smooth_16s-*/checkpoint-epoch*.pth"))
            # hits += list(Path(checkpoint_root).glob(
            #     f"{subject}-VS_smooth_16s-*/checkpoint-epoch*.pth"))  # pre-runs/ layout
            # if hits:
            #     print(f"Skipping {subject}: found {hits[0]}")
            #     continue
        print(f"\n{'='*60}\nStarting NeuroBOLT training for {subject}\n{'='*60}\n")
        args.dataname = subject
        prepare_run_dirs(args, subject, checkpoint_root)
        main(args)
        print(f"Completed training for {subject}")
    print("\nAll subjects completed!")
