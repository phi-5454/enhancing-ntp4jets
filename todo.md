# TODO

## Data Loading

- ~~Decide the class-label convention for ORBIT/L1T parquet inputs.~~
  EOS directory name is the label source. `OrbitParquetDataModule` now
  accepts `parquet_files_train_val_per_class: {class_name: paths}` and assigns
  integer labels from sorted class names. See `gabbro/data/orbit_parquet.py`.
- ~~Add class-aware train/val splitting once the label convention is fixed.~~
  `parquet_files_train_val_per_class` splits each class independently
  before chaining, so class balance is preserved in both train and val.
- ~~Add production configs for the intended train/val and test directory layout.~~
  `configs/experiment/orbit_jet_puppi_ak8_production.yaml` added.
  Reads manifests via `$ORBIT_MANIFEST_DIR`. Test set left empty until decided.
- ~~Decide whether jet modes should default to `L1T_Jet*` or `L1T_JetPuppi*`.~~
  `jet_puppi_ak8` will be tried for first production run.
- Verify sequence lengths against the final physics use case. 128 particles might still need checking, jets should be done

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

## Token Evaluation

- Decide where token-quality evaluation should run.
  Options are a standalone script, a Hydra job, or a callback after validation/test.
- Add a real token-quality smoke test once labelled multi-class parquet inputs are
  available.

## Logging And Plotting

- Add branch-aware codebook plots for split quantizers.
  The current combined-token plot is sparse-safe, but it does not yet show
  per-branch FSQ/VQ usage in the same detail as the split-token metadata.
- Decide whether physical-coordinate paper plots should be generated every
  validation epoch or only for explicit validation/test plotting runs.
  FastJet-backed particle plots are useful but materially slower than loss-only
  validation.
- Extend multirun post-processing to consume the new physical-coordinate paper
  histograms, including MET, jet pT resolution, jet mass, and tau32 overlays.

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

- Remove or document any legacy JetClass-only assumptions that remain in configs,
  plotting, or evaluation scripts.
