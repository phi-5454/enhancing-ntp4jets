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
- Decide whether jet modes should default to `L1T_Jet*` or `L1T_JetPuppi*`.
  The current smoke test uses `jet_puppi_ak8` because it matches the L1T-style
  event-tokenization path discussed so far.
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

- Connect token export from the train/eval pipeline to
  `scripts/evaluate_token_quality.py`.
  The evaluator currently consumes an `.npz` containing `token_ids`, `mask`, and
  `labels`; it does not yet generate that file from model outputs.
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

- Review untracked files and existing modified files before committing.
- Remove or document any legacy JetClass-only assumptions that remain in configs,
  plotting, or evaluation scripts.
