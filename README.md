# ORBIT Tokenizer

This repository combines three related codebases:

- the Hydra and Lightning training pipeline from [`enhancing-ntp4jets`](https://github.com/uhh-pd-ml/enhancing-ntp4jets/tree/main);
- the event-level jet tokenization use case from the [L1T tokenizer repo](https://github.com/philiw/vq-tokenizer-l1t);
- the [Phaedra prototype repo](https://github.com/phi-5454/ORBIT) parquet loader, split-quantizer architecture, and selected plotting
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

You can also point a run at an explicit env file, which is useful on batch
systems or when keeping W&B credentials outside the repo:

```bash
GABBRO_ENV_FILE=/path/to/wandb.env \
uv run --locked python gabbro/train.py experiment=orbit_parquet_smoke logger=wandb.yaml
```

`GABBRO_ENV_FILE` is loaded before Hydra composes the config, so it can provide
`LOG_DIR`, `MPLCONFIGDIR`, `WANDB_API_KEY`, `WANDB_ENTITY`, and
`COMET_API_TOKEN`. Values in the explicit env file override values already
loaded from the repository `.env`. Do not pass this as a Hydra override such as
`env_file=...`; that would be too late for `${oc.env:LOG_DIR}`.

`MPLCONFIGDIR` is useful on systems where the home directory is read-only. For
online W&B logging, authenticate once with `uv run wandb login` or provide the
usual W&B environment variables. The W&B project can be selected with
`logger.wandb.project=...`; explicit Hydra logger settings take precedence over
environment defaults.

### Arrange parquet files

The ORBIT parquet loader accepts parquet files, directories, or text manifests.
A manifest is a `.txt`, `.list`, or `.lst` file containing one
parquet path per line, with blank lines and `#` comments ignored:

```text
/path/to/sample_000.parquet
/path/to/sample_001.parquet
# /path/to/temporarily_disabled.parquet
```

Relative paths inside a manifest are resolved relative to the manifest file.
Direct directories remain supported. Use a separate manifest or directory for
the test set:

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

Manifest inputs are the preferred production shape:

```bash
'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
'data.parquet_files_test=[/path/to/pq_files_test.txt]'
```

Explicit parquet lists remain available for fixed splits and smoke tests:

```bash
data.parquet_files_train='[/path/train.parquet]' \
data.parquet_files_val='[/path/val.parquet]' \
data.parquet_files_test='[/path/test.parquet]'
```

### Multi-class mode

When data lives in separate per-class directories (e.g. on EOS), use
`parquet_files_train_val_per_class` instead of `parquet_files_train_val`. Pass
a mapping of class name to manifest (or directory). Class names are sorted
case-insensitively and assigned deterministic integer labels (0, 1, …). Each
class is split independently into train/val using the same `data.split_seed`
and `data.train_fraction`. The resulting `class_to_label` mapping is saved in
the run's `hparams` for reproducibility.

`parquet_files_train_val_per_class` is mutually exclusive with
`parquet_files_train_val` and explicit `parquet_files_train`/`parquet_files_val`.

**Step 1 — generate manifests.** The EOS dataset contains many process classes
as subdirectories. Run `make_eos_manifests.py` once to write one `.txt` manifest
per subdirectory:

```bash
python scripts/make_eos_manifests.py \
  --eos-root /eos/project/f/foundational-model-dataset/samples/production_final \
  --out-dir /path/to/manifests/
```

**Step 2 — select the classes you want.** The manifest directory will contain a
`.txt` file for every class in the EOS tree, but you only load the ones you
explicitly list in `parquet_files_train_val_per_class`. The rest are ignored.
The production experiment config `orbit_jet_puppi_ak8_production` already
selects four classes via `$ORBIT_MANIFEST_DIR`:

```yaml
# configs/experiment/orbit_jet_puppi_ak8_production.yaml (excerpt)
data:
  parquet_files_train_val_per_class:
    ggHbb:         ${oc.env:ORBIT_MANIFEST_DIR}/ggHbb.txt
    QCD_HT50toInf: ${oc.env:ORBIT_MANIFEST_DIR}/QCD_HT50toInf.txt
    VBFHbb:        ${oc.env:ORBIT_MANIFEST_DIR}/VBFHbb.txt
    minbias:       ${oc.env:ORBIT_MANIFEST_DIR}/minbias.txt
```

To use a different class selection, create a new experiment config with the
desired subset. Then set the manifest directory and launch:

```bash
export ORBIT_MANIFEST_DIR=/path/to/manifests/
uv run --locked python gabbro/train.py experiment=orbit_jet_puppi_ak8_production
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
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
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
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
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

### Progress bars and short runs

Lightning progress bars are disabled by default with
`trainer.enable_progress_bar=false` to keep batch-system logs compact. Enable the
default tqdm-style Lightning progress bar with:

```bash
trainer.enable_progress_bar=true
```

The usual Lightning batch limits are supported and are the preferred way to run
short tests:

```bash
trainer.max_epochs=1 \
trainer.limit_train_batches=10 \
trainer.limit_val_batches=2 \
trainer.limit_test_batches=2
```

Use integer limits for ORBIT parquet runs. The dataloader is iterable and
does not advertise a fixed epoch length, so fractional limits such as
`trainer.limit_train_batches=0.1` are not the robust choice.

Validation/test plotting has a separate retention limit. The model still
computes validation loss for the batches selected by `trainer.limit_val_batches`,
but by default it only keeps the first validation batch for reconstruction plots
and future FastJet-style physics diagnostics:

```bash
model.max_validation_plot_batches=1
model.max_test_plot_batches=null
```

Set `model.max_validation_plot_batches=10` if you want validation plots over ten
validation batches, or `model.max_test_plot_batches=20` to keep test plotting
bounded. `null` means keep all processed batches for that stage.

The ORBIT plotting callback skips Lightning's validation sanity check. The
sanity loop can still compute validation loss unless you disable it with
`trainer.num_sanity_val_steps=0`.

For an even smaller end-to-end check, use Lightning's fast dev mode:

```bash
trainer.fast_dev_run=true
```

If you want to skip the initial validation sanity loop during a quick run, add:

```bash
trainer.num_sanity_val_steps=0
```

## Data Contract

`gabbro.data.orbit_parquet.OrbitParquetDataModule` supports:

| `sequence_type` | Input prefix      | Default length |
| --------------- | ----------------- | -------------: |
| `particle`      | `L1T_PUPPIPart`   |            128 |
| `jet_ak4`       | `L1T_JetAK4`      |             14 |
| `jet_ak8`       | `L1T_JetAK8`      |              7 |
| `jet_puppi_ak4` | `L1T_JetPuppiAK4` |             14 |
| `jet_puppi_ak8` | `L1T_JetPuppiAK8` |              7 |

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

| Pipeline task                       | Main files                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Hydra experiment composition        | `configs/train.yaml`, `configs/experiment/*.yaml`                                                                      |
| Data source and batch schema        | `configs/data/*.yaml`, `gabbro/data/orbit_parquet.py`, `gabbro/data/iterable_dataset_jetclass.py`                      |
| Feature preprocessing definitions   | `configs/feature_dict/*.yaml`, `gabbro/utils/arrays.py`                                                                |
| Model architecture                  | `configs/model/*.yaml`, `gabbro/models/vqvae.py`, `gabbro/models/transformer.py`                                       |
| Single and split quantizers         | `gabbro/models/quantizers.py`, `configs/model/model_vqvae_transformer*.yaml`                                           |
| Lightning training/evaluation logic | `gabbro/models/vqvae.py`, `gabbro/models/lightning_models.py`, `gabbro/models/backbone_multihead.py`                   |
| Trainer, accelerator, and run paths | `configs/trainer/*.yaml`, `configs/hydra/default.yaml`, `configs/paths/default.yaml`                                   |
| Logging backends                    | `configs/logger/*.yaml`, `gabbro/utils/utils.py`                                                                       |
| Per-run plotting callbacks          | `configs/callbacks/*.yaml`, `gabbro/callbacks/orbit_plotting_callback.py`, `gabbro/callbacks/tokenization_callback.py` |
| Plot rendering utilities            | `gabbro/plotting/orbit.py`, `gabbro/plotting/feature_plotting.py`, `gabbro/plotting/utils.py`                          |
| Token export/reconstruction helpers | `gabbro/models/vqvae.py`, `gabbro/data/data_tokenization.py`, `scripts/create_tokenized_jetclass_files.py`             |
| Multirun post-processing            | `scripts/collect_orbit_multirun.py`, `gabbro/plotting/orbit.py`                                                        |
| Standalone checks and utilities     | `scripts/smoke_test_orbit_parquet.sh`, `scripts/evaluate_token_quality.py`, `scripts/filter_sparse_orbit_parquet.py`   |
| Environment and container setup     | `pyproject.toml`, `uv.lock`, `.env.example`, `docker/`                                                                 |

The ORBIT parquet path is configured through `orbit_parquet_smoke` or
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

The ORBIT plotting callback saves reconstruction histograms, residual plots,
codebook-usage plots, and the single-run physical-coordinate paper plots under
`plots/`. For particle runs, the paper plot set includes FastJet-backed jet
`pT` resolution, missing transverse energy, jet mass, and `tau32` residuals.
Attention-weight plots are intentionally not produced. Compressed histogram
arrays are saved under `saved_histograms/`, and compact metrics JSON files land
under `saved_metrics/`. The same plot images are logged to W&B or Comet when
those loggers are active.

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
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
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

The collector is adapted from the Phaedra prototype's multirun aggregation logic, but it does
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

### From the Phaedra Prototype

The imported parquet-tokenization additions are:

- parquet streaming by row group;
- text manifests containing parquet paths;
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

The rotation-trick VQ gradient estimator from the Phaedra prototype has not been ported yet.
The natural implementation point is `gabbro/models/quantizers.py`:

1. Add a rotation-trick autograd function equivalent to the prototype's
   `_RotationTrick`.
2. Extend `VQBranch` with a `gradient_estimator` option such as `"ste"` or
   `"rotation_trick"`.
3. Pass that option through `split_quantizer_cfg.vq_kwargs`.
4. Add the same option to the single VectorQuant path if rotation-trick support
   is also required without `Phi` and `Psi`.
5. Preserve the existing normalized output contract so token export,
   reconstruction, plotting, and `part_token_id` compatibility continue to
   work.

The plotting helpers already reserve a `vq_rotation` family for
future comparison plots.

## Legacy Workflows

The original JetClass tokenizer, token export, joint pre-training, and
classification paths remain available. Their starting points are:

```text
configs/experiment/example_experiment_tokenization_transformer.yaml
configs/experiment/example_experiment_backbone_and_head.yaml
scripts/create_tokenized_jetclass_files.py
```

These paths retain assumptions from the original enhancing repository and
should be reviewed separately before mixing them with the ORBIT parquet runs.
