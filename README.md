<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>





<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">NeuroBOLT fork guide</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Project files layout

```text
neuro_bolt-main/
├── main.py                 # Namespace config + subject select + run folders
├── run_neuro.sh            # SLURM script
├── engine.py               # train / evaluate steps
<!-- ├── runtime.py              # distributed helpers, checkpoint save/load, TB logger glue -->
├── utils.py                # shared utilities (includes alternate save helpers)
├── optim_factory.py
├── requirements.txt        #modules to import
├── scan_split_example.xlsx # VU / cross-subject splits (optional path)
├── arch/                   # baseline model + MSS (sum/mean)
│   ├── model.py
│   └── model_multiscale.py
<!-- ├── models/                  -->
│   ├── model.py  
│   ├── model_multiscale.py
│   ├── new_layers.py       # BandAttention, ChannelAttention attention pooling 
│   └── check.py
├── dataset_maker/
│   ├── get_datasets.py     # custom dataset loaders (Algermissen for this project)
│   └── preproc.py          # VU path helper
├── checkpoints/            # place labram-base.pth (or point finetune elsewhere)
├── logs/                   # SLURM logs with .out / .err
├── README.md               # this file (fork guide)
└── README_neurobolt.md     # original NeuroBOLT docs
```
---
## About this project

This project contains implementation of NeuroBOLT pipeline adapted to Algermissen dataset. NeuroBOLT is a 


<p align="right">(<a href="#readme-top">back to top</a>)</p>







<!-- GETTING STARTED -->
## Installation
To run this fork or the official NeuroBOLT pipeline, run the following commands: 
```sh

conda create -n neurobolt python=3.9
conda activate neurobolt
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tensorboardX
pip install -r requirements.txt

```

### Code setup

1. To use the pipeline, you will need to submit a SLURM job by running `run_neuro.sh`.

2.  `main.py` is the main entry point, responsible for 'orchestrating' the pipeline. When invoked by `run_neuro.sh`, the script loads the data, builds the model,finetunes with LaBraM hyperparameters, trains epochs, evaluates and saves logs/checkpoints/plots. It doesn't contain any training logic by itself. Contains configuration object `args` with hardcoded parameters that can be customized. Note: configuration is edited in this file and not in the command line. 


| Parameter | Role |
|-----------|------|
| `labram_ckpt_path` / `args.finetune` | LaBraM pretrained weights path |
| `checkpoint_path` | Base for checkpoints / run folders |
| `ALGERMISSEN_DATA_ROOT` / `args.dataset_root` | Per-block Algermissen `.npz` directory |
| `ALGERMISSEN_SKIP` | Subjects excluded from the intrascan loop (default: `sub-004`, `sub-015`, `sub-025`) |
| `args.output_dir` | Run artifacts root (overwritten by `prepare_run_dirs` to `checkpoints/runs/...`) |
| `args.log_dir` | TensorBoard dir (overwritten to `{run_dir}/tb/`) |
| `args.dataname` | Current subject id (set per loop iteration) |
| `args.labels_roi` | Default `'VS'` (checkpoint naming) |
| `args.dataset` | Default `'algermissen'` |
| `args.train_test_mode` | Default `'intrascan'` |
| `args.TR`, `args.window_sec`, `args.model_hz` | Timing / windowing (1.4 s, 16 s, 200 Hz) |
| `args.lr`, `args.batch_size`, `args.epochs`, `args.drop`, `args.weight_decay` | Core training hypers |
| `args.update_freq` | Gradient accumulation steps |
| `args.layer_decay`, `args.warmup_*`, `args.min_lr` | LR schedule / layer-wise decay |
| `args.model`, `args.nb_roi`, drop/attn/pos flags | Architecture knobs passed into `create_model` |
| `args.resume`, `args.auto_resume`, `args.save_ckpt` | Checkpoint resume / save |
| `NEUROBOLT_SUBJECT` | Env: force one subject id |
| `SLURM_ARRAY_TASK_ID` | Env: 0-based index into skip-filtered subject list |
| `list_algermissen_subjects` | Discover subjects from `*_block*.npz` |
| `select_subjects` | Apply env / array selection |
| `prepare_run_dirs` | Create `runs/{timestamp}_j{JOB}[_a{ARRAY}]_{subject}/` + `run_meta.json` |
| `get_models` / `get_dataset` / `main` | Build model, load data, run one subject job |
| VU-only knobs | `split_index_sheet`, `mri_sync_event`, `prepro_datapath`, `VU_CH_NAMES` (idle on Algermissen) |
---


### Training mechanics

1. `engine.py` turns an EEG batch into a loss. 

| Parameter | Role |
|-----------|------|
| `model`, `criterion` | Network + MSE loss |
| `data_loader` | Yields `(EEG, target)` batches |
| `optimizer`, `loss_scaler` | Step weights; AMP GradScaler wrapper from `runtime` |
| `ch_names` → `input_chans` | Via `utils.get_input_chans` before forward |
| `spec_chan` | Optional spectral channel subset (`None` on live Algermissen) |
| `update_freq` | Accumulate this many micro-batches before optimizer step |
| `lr_schedule_values`, `wd_schedule_values` | Per-iteration LR / weight-decay |
| `max_norm` / `args.clip_grad` | Optional grad clip inside scaler |
| `is_binary` | If true, unsqueeze target to `[B,1]` (single ROI) |
| `/100`, `T=200` | EEG scale and patch length (hard-coded in loop) |

---

2. `optim.py` constructs an optimizer and optional ViT layer-decay parameter groups for fine-tuning.

| Parameter | Role |
|-----------|------|
| `create_optimizer(args, model, …)` | Construct AdamW (default) / other opts from `args.opt` |
| `args.lr`, `args.weight_decay`, `args.opt_eps`, `args.opt_betas` | Optimizer hyperparameters |
| `args.opt` | Optimizer name (`'adamw'`) |
| `LayerDecayValueAssigner` | Maps layer id → LR scale from `args.layer_decay` |
| `get_parameter_groups` | Split decay / no_decay (+ per-layer LR scale) |
| `get_num_layer_for_vit` | Assign transformer block depth for decay |
| `skip_list` | Params without weight decay (e.g. bias, `pos_embed`) |

---

3. `utils.py` and `runtime.py` share the same code infrastructure. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>








<!-- LICENSE
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



