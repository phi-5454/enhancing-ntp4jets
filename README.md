# Enhancing NTP for ORBIT and L1T parquet data

This repository combines three related codebases:

- the Hydra and Lightning training pipeline from `enhancing-ntp4jets`;
- the event-level jet tokenization use case from the L1T tokenizer repo;
- the ORBIT parquet loader, split-quantizer architecture, and selected plotting
  conventions.

The current focus is VQ-VAE tokenization of absolute-coordinate particle or jet
sequences stored in parquet files. The original JetClass and downstream
pre-training code remains available, but it is not the primary workflow described
below.

## Quick Start

### Install with UV

Install [UV](https://docs.astral.sh/uv/) and create the local environment:

```bash
uv sync --locked
```

Set a log directory before launching Hydra jobs:

```bash
export LOG_DIR="$PWD/outputs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/enhancing-mplconfig}"
```

Alternatively, copy `.env.example` to `.env` and set the values there. The
training entrypoint loads the repository-root `.env` automatically through
`pyrootutils`. The file is ignored by Git and is the appropriate place for
`WANDB_API_KEY`, `WANDB_ENTITY`, and `COMET_API_TOKEN`.

`MPLCONFIGDIR` is useful on systems where the home directory is read-only. For
online W&B logging, authenticate once with `uv run wandb login` or provide the
usual W&B environment variables. The W&B project can be selected with
`logger.wandb.project=...`; explicit Hydra logger settings take precedence over
environment defaults.

### Arrange parquet files

The ORBIT/L1T loader accepts one or more parquet files or directories. Use a
separate directory for the test set:

```text
/path/to/dataset/
├── train_val/
│   ├── sample_000.parquet
│   ├── sample_001.parquet
│   └── ...
└── test/
    ├── sample_100.parquet
    └── ...
```

The loader deterministically shuffles and splits the files under `train_val/`
using `data.split_seed` and `data.train_fraction`. The training dataset then
reshuffles parquet row groups between iterations. Validation and test row groups
are not shuffled. Automatic train/validation splitting requires at least two
parquet files.

Explicit lists remain available for fixed splits and smoke tests:

```bash
data.parquet_files_train='[/path/train.parquet]' \
data.parquet_files_val='[/path/val.parquet]' \
data.parquet_files_test='[/path/test.parquet]'
```

### Run a smoke test

The checked-in smoke script runs both particle and PUPPI AK8 jet modes against a
single parquet fixture:

```bash
PARQUET_FILE=/path/to/example.parquet \
OUTPUT_DIR=/tmp/enhancing-smoke \
./scripts/smoke_test_orbit_parquet.sh
```

### Train with the single VQ quantizer

The following is a runnable starting point for particle tokenization:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer \
  logger=wandb.yaml \
  task_name=orbit_particle_vq \
  logger.wandb.name=orbit_particle_vq \
  logger.wandb.project=orbit-tokenizer \
  'data.parquet_files_train_val=[/path/to/dataset/train_val]' \
  'data.parquet_files_test=[/path/to/dataset/test]' \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null \
  trainer.enable_checkpointing=true
```

### Train with the split quantizer

The split model applies:

```text
encoder latent -> Phi -> branch quantizers -> Psi -> decoder latent
```

`Phi` and `Psi` use NormFormer stacks. Each branch may use FSQ or VQ. The
checked-in split config uses FSQ for both `mu` and `alpha`:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer_split \
  logger=wandb.yaml \
  task_name=orbit_particle_split_fsq \
  logger.wandb.name=orbit_particle_split_fsq \
  logger.wandb.project=orbit-tokenizer \
  'data.parquet_files_train_val=[/path/to/dataset/train_val]' \
  'data.parquet_files_test=[/path/to/dataset/test]' \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null \
  trainer.enable_checkpointing=true
```

To use the L1T-style event representation, replace
`experiment=orbit_parquet_smoke` with `experiment=orbit_jet_parquet_smoke`.
Each sequence element is then one absolute-coordinate PUPPI AK8 jet.

The current ORBIT experiment configs are smoke-ready integration starting
points. Before a long production run, review model size, sequence length, batch
size, checkpoint policy, and train/validation limits.

## Data Contract

`gabbro.data.orbit_parquet.OrbitParquetDataModule` supports:

| `sequence_type` | Input prefix | Default length |
| --- | --- | ---: |
| `particle` | `L1T_PUPPIPart` | 128 |
| `jet_ak4` | `L1T_JetAK4` | 16 |
| `jet_ak8` | `L1T_JetAK8` | 8 |
| `jet_puppi_ak4` | `L1T_JetPuppiAK4` | 16 |
| `jet_puppi_ak8` | `L1T_JetPuppiAK8` | 8 |

All modes emit the same model input contract:

```text
part_features:   [batch, sequence, 4]
part_mask:       [batch, sequence]
jet_type_labels: [batch]
```

The four features are scaled `eta`, `cos(phi)`, `sin(phi)`, and transformed
`pT`. Coordinates remain absolute: `eta` and `phi` are not made relative to a
jet axis.

## Repository Structure

The main training entrypoint is `gabbro/train.py`. It composes Hydra configs,
instantiates the datamodule, model, callbacks, loggers, and Lightning trainer,
then runs train/validation/test according to the config.

Pipeline responsibilities are split as follows:

| Pipeline task | Main files |
| --- | --- |
| Hydra experiment composition | `configs/train.yaml`, `configs/experiment/*.yaml` |
| Data source and batch schema | `configs/data/*.yaml`, `gabbro/data/orbit_parquet.py`, `gabbro/data/iterable_dataset_jetclass.py` |
| Feature preprocessing definitions | `configs/feature_dict/*.yaml`, `gabbro/utils/arrays.py` |
| Model architecture | `configs/model/*.yaml`, `gabbro/models/vqvae.py`, `gabbro/models/transformer.py` |
| Single and split quantizers | `gabbro/models/quantizers.py`, `configs/model/model_vqvae_transformer*.yaml` |
| Lightning training/evaluation logic | `gabbro/models/vqvae.py`, `gabbro/models/lightning_models.py`, `gabbro/models/backbone_multihead.py` |
| Trainer, accelerator, and run paths | `configs/trainer/*.yaml`, `configs/hydra/default.yaml`, `configs/paths/default.yaml` |
| Logging backends | `configs/logger/*.yaml`, `gabbro/utils/utils.py` |
| Per-run plotting callbacks | `configs/callbacks/*.yaml`, `gabbro/callbacks/orbit_plotting_callback.py`, `gabbro/callbacks/tokenization_callback.py` |
| Plot rendering utilities | `gabbro/plotting/orbit.py`, `gabbro/plotting/feature_plotting.py`, `gabbro/plotting/utils.py` |
| Token export/reconstruction helpers | `gabbro/models/vqvae.py`, `gabbro/data/data_tokenization.py`, `scripts/create_tokenized_jetclass_files.py` |
| Multirun post-processing | `scripts/collect_orbit_multirun.py`, `gabbro/plotting/orbit.py` |
| Standalone checks and utilities | `scripts/smoke_test_orbit_parquet.sh`, `scripts/evaluate_token_quality.py`, `scripts/filter_sparse_orbit_parquet.py` |
| Environment and container setup | `pyproject.toml`, `uv.lock`, `.env.example`, `docker/` |

The ORBIT/L1T parquet path is configured through `orbit_parquet_smoke` or
`orbit_jet_parquet_smoke`. The original enhancing JetClass path is configured
through `example_experiment_tokenization_transformer` and
`example_experiment_backbone_and_head`.

## Tracking Runs

### Output directories

Hydra writes each normal run under:

```text
${LOG_DIR}/${project_name}/runs/<timestamp>_<generated-id>/
```

Relevant files include:

```text
config.yaml
config_resolved.yaml
checkpoints/
plots/
saved_histograms/
saved_metrics/
wandb/
csv/
```

The ORBIT plotting callback saves reconstruction histograms, residual plots, and
codebook-usage plots under `plots/`. It also saves compressed histogram arrays
under `saved_histograms/` and a compact metrics JSON under `saved_metrics/`. The
same plot images are logged to W&B or Comet when those loggers are active.

### Supported loggers

Select a logger with a Hydra override:

```bash
logger=wandb.yaml
logger=csv.yaml
logger=comet.yaml
logger=many_loggers.yaml
```

W&B run names default to `task_name` and can be overridden explicitly:

```bash
task_name=orbit_vq_v1 logger.wandb.name=orbit_vq_v1
```

For an offline W&B run:

```bash
logger=wandb.yaml logger.wandb.offline=true
```

Offline payloads land under the run directory in `wandb/offline-run-*` and can
later be uploaded with `uv run wandb sync <offline-run-directory>`.

Comet requires `COMET_API_TOKEN`. CSV logging requires no external service.
MLflow is not currently configured in this repository.

### Hydra multiruns

Basic parameter sweeps use Hydra's `-m` mode. Each job gets its own output
directory under `${LOG_DIR}/${project_name}/multiruns/`:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py -m \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer \
  logger=wandb.yaml \
  task_name=orbit_vq_scan \
  'logger.wandb.name=orbit_vq_scan_codes_${model.model_kwargs.vq_kwargs.num_codes}_seed_${seed}' \
  logger.wandb.group=orbit_vq_scan \
  'data.parquet_files_train_val=[/path/to/dataset/train_val]' \
  'data.parquet_files_test=[/path/to/dataset/test]' \
  model.model_kwargs.vq_kwargs.num_codes=256,512,1024 \
  seed=42,43 \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null
```

Each job writes its own plots, compressed histograms, and compact metrics JSON.
After the sweep finishes, aggregate the jobs with:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --multirun-dir "${LOG_DIR}/orbit-smoke/multiruns/<timestamp>"
```

Use `--stage test` to select test artifacts instead of validation artifacts. By
default, the script creates `comparisons/` inside the multirun directory. It
contains:

```text
manifest.json
summary.csv
combined_reconstruction_features.png
combined_reconstruction_residuals.png
codebook_size_vs_mse_total.png
codebook_size_vs_utilization_total.png
codebook_size_vs_val_loss.png
```

The collector is adapted from ORBIT's multirun aggregation logic, but it does
not launch training jobs itself. Hydra remains responsible for sweep expansion
and execution. The collector reads each job's Hydra config to infer single-VQ or
split-quantizer codebook metadata, then consumes the local
`saved_histograms/*.npz` and `saved_metrics/*.json` artifacts. It also falls back
to Lightning CSV metrics for older runs.

W&B sweeps and an Optuna sweeper are possible future additions, but there is no
checked-in integration for them yet.

## What Changed

### From enhancing-ntp4jets

The original Hydra/Lightning structure remains the base. Existing JetClass
loading, tokenization scripts, Comet/CSV logging, and downstream backbone
workflows remain in the tree.

Additions and adjustments:

- UV project metadata and lockfile for local development without Docker;
- W&B as an explicit logger option with predictable run names;
- the schema-neutral ORBIT plotting callback;
- the split quantizer abstraction and split-token reconstruction;
- the L1T-compatible Docker pin `tables==3.10.1`.

### From the L1T tokenizer repo

The event-level jet use case is preserved as a separate loader mode rather than
replacing particle loading:

- `jet_puppi_ak8` represents each event as a variable-length sequence of PUPPI
  AK8 jets;
- particle and jet modes emit the same batch keys and feature width;
- the main architectural difference seen by the model is sequence length;
- absolute `phi` is represented with `cos(phi)` and `sin(phi)`.

Sparse-event parquet filtering is intentionally left as an optional boxed-out
script until profiling shows that it is needed.

### From ORBIT

The ORBIT-oriented additions are:

- parquet streaming by row group;
- deterministic shuffled file splitting for train/validation and a separate
  test directory;
- row-group reshuffling for training to avoid long contiguous data periods;
- absolute-coordinate preprocessing;
- a `Phi -> branches -> Psi` split quantizer with FSQ/VQ branches;
- storage of combined `part_token_id` plus explicit `part_token_<branch>`
  fields;
- reconstruction from explicit branch tokens or packed combined IDs;
- selected reconstruction, residual, codebook-usage, and attention plotting
  helpers.

Attention plotting helpers are present, but they are not wired into the
callback until the VQ-VAE exposes attention tensors during validation and test.

## Rotation Trick Extension

The rotation-trick VQ gradient estimator from ORBIT has not been ported yet.
The natural implementation point is `gabbro/models/quantizers.py`:

1. Add a rotation-trick autograd function equivalent to ORBIT's
   `_RotationTrick`.
2. Extend `VQBranch` with a `gradient_estimator` option such as `"ste"` or
   `"rotation_trick"`.
3. Pass that option through `split_quantizer_cfg.vq_kwargs`.
4. Add the same option to the single VectorQuant path if rotation-trick support
   is also required without `Phi` and `Psi`.
5. Preserve the existing normalized output contract so token export,
   reconstruction, plotting, and `part_token_id` compatibility continue to
   work.

The ORBIT-style plotting helpers already reserve a `vq_rotation` family for
future comparison plots.

## Legacy Enhancing Workflows

The original JetClass tokenizer, token export, joint pre-training, and
classification paths remain available. Their starting points are:

```text
configs/experiment/example_experiment_tokenization_transformer.yaml
configs/experiment/example_experiment_backbone_and_head.yaml
scripts/create_tokenized_jetclass_files.py
```

These paths retain assumptions from the original enhancing repository and
should be reviewed separately before mixing them with ORBIT/L1T parquet runs.
