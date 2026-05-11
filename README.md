# HyGEMamba

HyGEMamba is a genotype-environment deep learning model for hybrid maize trait
prediction. It combines dual-parent genotype sequences with weather time series
and supports scalar trait prediction, growth-curve prediction, and joint
multi-target training.

The main training entry point is [`train.py`](train.py).

## What This Repository Contains

- `src/models/hyge_mamba.py`: the `HyGEMamba` model.
- `train.py`: the primary training script used for experiments.
- `src/modules/`: genome, weather, fusion, MoE, and prediction-head modules.
- `src/data/`: reusable dataset, dataloader, and synthetic data utilities.
- `src/losses/`: multitask, curve, and MoE auxiliary losses.
- `tests/`: unit tests for model, data, and loss components.

Generated outputs, checkpoints, logs, and private datasets are excluded by
`.gitignore` and should not be committed to a public repository.

## Model Module Names

The main `HyGEMamba` model registers its core biological modules with these
public names:

| Module | Meaning |
| --- | --- |
| `HGIE` | Hybrid Genomic Information Encoder |
| `HEE` | Hierarchical Environment Encoder |
| `PPH` | Phenotype Prior Head |

## Environment

The project is developed and verified in WSL with Conda:

```bash
cd HyGEMamba
conda create -n HyGEMamba python=3.11 -y
conda activate HyGEMamba
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

If you already have a `HyGEMamba` environment, activate it directly and run the
`pip install` commands.

For CUDA-accelerated Mamba kernels:

```bash
python -m pip install -e ".[mamba]"
```

If `mamba_ssm` is unavailable, or if CPU tensors are passed to the model,
HyGEMamba uses a pure PyTorch fallback for the Mamba block. This keeps tests and
CPU development runnable. For real GPU training, keep the model and tensors on
CUDA so the official `mamba_ssm` kernels are used.

PyTorch wheels must match the local NVIDIA driver and CUDA runtime. If the
default PyPI wheel reports that the driver is too old, either install a CPU-only
PyTorch build for code verification or install a CUDA build supported by your
driver before installing this project.

## Verify Installation

```bash
conda activate HyGEMamba
python scripts/verify_installation.py
pytest -q
python train.py --help
```

Expected status in the verified WSL environment:

- `scripts/verify_installation.py`: passes all checks.
- `pytest -q`: 59 tests pass.
- `train.py --help`: shows the train.py command-line interface.

## Reproduce a Minimal Run Without Private Data

The real experiments require an HDF5 dataset that follows the schema below.
Private datasets, logs, and checkpoints are intentionally not included in this
repository. To verify that the installed code can run end-to-end, create a tiny
synthetic HDF5 file and run one CPU epoch:

```bash
python scripts/create_smoke_dataset.py --output /tmp/hygemamba_smoke.h5
python train.py \
  --h5 /tmp/hygemamba_smoke.h5 \
  --split M1 \
  --target_mode scalar \
  --epochs 1 \
  --batch_size 2 \
  --device cpu \
  --num_workers 0 \
  --output_dir /tmp/hygemamba_smoke_out
```

This smoke test is for reproducibility of the software pipeline only. Reported
research metrics should be reproduced with the real HDF5 dataset.

## Training

`train.py` supports three target modes:

| Mode | Targets |
| --- | --- |
| `scalar` | `yield`, `plant_height_cm`, `ear_height_cm` |
| `curve` | `Maize.AboveGround.Wt`, `LAI`, `LeafWt`, `StemWt` |
| `all` | all scalar and curve targets |

Train scalar targets:

```bash
python train.py \
  --h5 data/hyge_mamba_dataset.h5 \
  --split M1 \
  --target_mode scalar \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

Train curve targets:

```bash
python train.py \
  --h5 data/hyge_mamba_dataset.h5 \
  --split M1 \
  --target_mode curve \
  --primary_task Maize.AboveGround.Wt \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

Train all targets:

```bash
python train.py \
  --h5 data/hyge_mamba_dataset.h5 \
  --split M1 \
  --target_mode all \
  --primary_task yield \
  --epochs 100 \
  --batch_size 32 \
  --device cuda
```

After installation, the console entry points are also available:

```bash
hyge-train --h5 data/hyge_mamba_dataset.h5 --split M1 --target_mode all
```

## Important Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--h5` | required | HDF5 dataset path |
| `--split` | `M1` | Experiment split: `M1`, `M2`, `M3`, `M4`, or `M5` |
| `--target_mode` | `all` | `scalar`, `curve`, or `all` |
| `--primary_task` | mode-dependent | Early-stopping target |
| `--loss_strategy` | `famo` | `famo`, `uncertainty`, or `dwa` |
| `--device` | auto | `cuda` or `cpu` |
| `--lazy_load` | off | Load HDF5 samples lazily instead of preloading arrays |
| `--output_dir` | auto | Output directory for checkpoints and metrics |
| `--resume` | none | Resume from a checkpoint |

## HDF5 Dataset Schema

`train.py` expects an HDF5 file with these fields and attributes.

Required root attributes:

- `num_samples`
- `num_snps`
- `num_days`
- `num_weather_features`
- optional `is_shared_genomic_coords`

Required arrays:

- `genotypes_a`: shape `(N, num_snps)`
- `genotypes_b`: shape `(N, num_snps)`
- `chromosomes`: shape `(num_snps,)` or `(N, num_snps)`
- `positions`: shape `(num_snps,)` or `(N, num_snps)`
- `weather_data`: shape `(N, num_days, num_weather_features)`

Targets:

- Curve targets: `targets`, shape `(N, 4, T)` or `(N, T, 4)`.
- Scalar targets: either `scalar_targets`, shape `(N, 3)`, or separate arrays
  named `yield`, `plant_height_cm`, and `ear_height_cm`.
- Optional APSIM targets: `apsim_params`, shape `(N, 6)`.

Splits:

```text
splits/
  M1/
    train_indices
    val_indices
    test_indices
  M2/
  M3/
  M4/
  M5/
```

Each split group may include:

- `description`
- `use_genotype`
- `use_environment`

`M4` is intended for environment-only ablations and `M5` for genotype-only
ablations when the dataset split attributes are set accordingly.

## Outputs

By default, training writes to:

```text
outputs/{split}_small_{target_mode}_{timestamp}/
```

Typical files:

- `config.json`
- `latest.pt`
- `best_overall.pt`
- `best_<task>.pt`
- `training_history.json`
- `test_results.json`

These files are generated artifacts and are ignored by `.gitignore`.

## Development Checks

Run these before publishing changes:

```bash
python scripts/verify_installation.py
pytest -q
python -m compileall src scripts configs tests train.py
python train.py --help
```

## Citation

If you use this code in academic work, cite the repository and the associated
paper or preprint when available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
