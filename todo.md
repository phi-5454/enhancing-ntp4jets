# TODO

## Data Loading

- Decide the class-label convention for ORBIT/L1T parquet inputs.
  The current loader emits a configurable placeholder `jet_type_label`; token-quality
  evaluation and class-balanced splitting need real labels from either directory
  layout, filename patterns, metadata columns, or an external manifest.
- Add class-aware train/val splitting once the label convention is fixed.
  The current train/val split deterministically shuffles files globally, which is
  fine for many mixed files but does not guarantee balanced classes for small
  datasets.
- Add production configs for the intended train/val and test directory layout.
  Smoke configs intentionally reuse one sample file for train, val, and test.
- ORBIT-style text manifests are supported for all parquet path fields. Direct
  parquet files and directories remain supported.
- Decide whether jet modes should default to `L1T_Jet*` or `L1T_JetPuppi*`.
  The current smoke test uses `jet_puppi_ak8` because it matches the L1T-style
  event-tokenization path discussed so far.
- Verify sequence lengths against the final physics use case.
  Current defaults are 128 particles, 16 AK4 jets, and 8 AK8 jets.

## Model And Config Integration

- Add non-smoke Hydra experiment configs for particle and jet training.
  The current configs are enough for a UV smoke test but are not tuned training
  configs.
- Check whether model positional embeddings, masking, and plotting utilities need
  explicit assumptions about sequence length.
  Particle and jet loaders now expose the same batch schema but different lengths.
- Tune the split quantizer configs for the actual particle and jet runs.
  The model now supports plain VQ and Phi/Psi split quantization with FSQ/VQ
  branches, but the checked-in split config is still a smoke-ready starting point.
- Split quantizer tokenization now stores explicit branch token fields
  `part_token_<branch>` alongside the combined `part_token_id`.
- Split-token reconstruction is implemented for explicit branch tokens and for
  combined `part_token_id` fallback, including FSQ unpacking and branch VQ
  codebook lookup.

## Token Evaluation

- Connect token export from the train/eval pipeline to
  `scripts/evaluate_token_quality.py`.
  The evaluator currently consumes an `.npz` containing `token_ids`, `mask`, and
  `labels`; it does not yet generate that file from model outputs.
- Decide where token-quality evaluation should run.
  Options are a standalone script, a Hydra job, or a callback after validation/test.
- Add a real token-quality smoke test once labelled multi-class parquet inputs are
  available.

## Logging And Plotting

- W&B logging is available through `logger=wandb.yaml`, with predictable run names
  from `task_name` and support for explicit `logger.wandb.name=...` overrides.
  Keep the existing CSV and Comet logger paths usable.
- Hydra multiruns now resolve `paths.output_dir` through the active runtime
  directory, so each sweep job keeps its artifacts under `multiruns/<timestamp>/<job>`.
- The README documents UV setup, parquet layout, single-VQ and split-quantizer
  runs, Hydra multiruns, supported loggers, output locations, and source-repo
  integration notes.
- Root `.env` loading is available through `pyrootutils`; `.env.example`
  documents `LOG_DIR`, W&B credentials/defaults, and the optional Comet token.
- Explicit run env files are supported with `GABBRO_ENV_FILE=/path/to/file.env`.
  This is loaded before Hydra composes the config, so it can provide `LOG_DIR`,
  `MPLCONFIGDIR`, and logger credentials.
- ORBIT-style schema-neutral reconstruction and token-usage plots are ported via
  `gabbro.plotting.orbit` and `OrbitPlottingCallback`.
  The plotting helpers use enhancing's existing style and color palette.
- Single-run physical-coordinate paper plots are wired into validation/test:
  kinematic distributions and residuals, energy, MET, jet pT resolution, and
  particle-run FastJet jet mass/tau32 plots. Attention plots remain intentionally
  unwired.
- Validation plotting/evaluation stores only
  `model.max_validation_plot_batches` batches by default, currently one batch to
  match ORBIT's lightweight validation sample. Test plotting can retain all
  processed batches by default, or be bounded with `model.max_test_plot_batches`.
  The ORBIT plotting callback now skips Lightning sanity validation.
- `scripts/collect_orbit_multirun.py` post-processes Hydra multirun directories.
  It writes a manifest, summary CSV, reconstruction overlays, and codebook-size
  scans from per-run histogram and metric artifacts.
- Wire attention diagnostics into a callback once the model exposes attention
  tensors during validation/test.
  The `delta eta/phi` and attention-map figure helpers are available, but the
  current VQ-VAE validation outputs only contain features, masks, and code IDs.
- Check future plotting additions against the absolute-coordinate feature names:
  `Eta`, `Phi_cos`, `Phi_sin`, and `PT`.

## Sparse Parquet Filtering

- Keep `scripts/filter_sparse_orbit_parquet.py` boxed out until profiling shows it
  is useful.
- If it becomes useful, extend it to process directories and preserve relative
  paths instead of only filtering one source file at a time.

## Testing

- Add focused unit tests for:
  - deterministic file splitting;
  - row-group shuffling changing train order between iterations;
  - common batch schema across particle and jet loaders;
  - absolute `phi` preprocessing through `cos(phi)` and `sin(phi)`;
  - explicit rejection of unknown `sequence_type` values.
- Add an integration test with at least two train/val files and one separate test
  file once small fixtures are available in the repo.

## Repository Cleanup

- Decide whether the UV setup should become the primary local development path or
  remain an alternative to Docker.
- Docker requirements use the L1T-compatible `tables==3.10.1` pin because
  `tables==3.10.2` breaks the container dependency set.
- Review untracked files and existing modified files before committing.
- Remove or document any legacy JetClass-only assumptions that remain in configs,
  plotting, or evaluation scripts.
