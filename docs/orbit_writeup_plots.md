# ORBIT writeup plots

`scripts/build_orbit_writeup_plots.py` is a cache-only renderer for the
LaTeX-ready plot collection. It does not use a plotting YAML and never launches
inference. Model evaluation and rendering remain separate so labels, bins,
layouts, and styles can be changed without repeating GPU work.

Every figure is written as PDF and 300-dpi PNG without a main title. The output
also contains CSV/JSON/LaTeX tables, numerical NPZ caches, a linked README, and
model provenance in Markdown, LaTeX, CSV, and JSON.

## Expensive caches

The exact SM downstream model must be the unweighted, full-500,
SM-mixture-trained VQ-STE checkpoint with 4096 codes (W&B training run
`kit9ge98`). Stage its `best.ckpt` and resolved config together in one run
directory before running the following commands.

The established Higgs and tau32 evaluators create two of the caches:

```bash
python scripts/evaluate_orbit_higgs_mass.py \
  --candidate-mode leading_pt \
  --run-dir "$SM_VQ4096_RUN" \
  --gghbb-test-manifest "$ORBIT_MANIFEST_DIR/ggHbb_test.txt" \
  --output-dir "$CACHE_DIR/higgs" \
  --events 10000 --device cuda

python scripts/evaluate_orbit_tau32.py \
  --run-dir "$SM_VQ4096_RUN" \
  --test-manifest "$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt" \
  --output-dir "$CACHE_DIR/tau32" \
  --events 10000 --device cuda
```

Jet and PID caches accept repeated top-level classes. Comma-separated manifests
inside each class are balanced within that class; the classes themselves are
balanced against one another.

```bash
SM_CLASSES=(
  --test-class "QCD=$ORBIT_MANIFEST_DIR/QCD_HT50tobb_test.txt,$ORBIT_MANIFEST_DIR/QCD_HT50toInf_test.txt"
  --test-class "tt=$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt,$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_leptonic_test.txt,$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt"
  --test-class "VJets=$ORBIT_MANIFEST_DIR/WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt"
  --test-class "VV=$ORBIT_MANIFEST_DIR/WW_hadronic_test.txt,$ORBIT_MANIFEST_DIR/WW_leptonic_test.txt,$ORBIT_MANIFEST_DIR/WW_semileptonic_test.txt,$ORBIT_MANIFEST_DIR/WZ_hadronic_test.txt,$ORBIT_MANIFEST_DIR/WZ_leptonic_test.txt,$ORBIT_MANIFEST_DIR/WZ_semileptonic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_hadronic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_leptonic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_semileptonic_test.txt"
  --test-class "DY=$ORBIT_MANIFEST_DIR/DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"
)

python scripts/evaluate_orbit_jet_residuals.py \
  --run-dir "$SM_VQ4096_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/jets" --events 10000 --device cuda

python scripts/evaluate_orbit_pid_pulls.py \
  --run-dir "$SM_VQ4096_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/sm_pid" --events 10000 --device cuda

# MET uses the dedicated 8192-code SM-mixture model. This paired cache retains
# decoded-minus-original residuals event by event.
python scripts/evaluate_orbit_met.py \
  --run-dir "$SM_VQ8192_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/met" --events 10000 --device cuda
```

For the canonical ttbar VQ-STE 4096 PID panels, run the PID evaluator on its
exact golden checkpoint and the established ttbar test manifest. The resulting
NPZ supplies both residuals and the confusion matrix.

## Rendering

```bash
python scripts/build_orbit_writeup_plots.py \
  --golden-manifest "$GOLDEN_MANIFEST" \
  --campaign-summary-root "$CAMPAIGN_WANDB_MIRROR" \
  --campaign-summary-root "$CAMPAIGN_7_8_RUN_ROOT" \
  --downstream-run-dir "$SM_VQ4096_RUN" \
  --golden-pid-confusion "$GOLDEN_PID_CACHE" \
  --golden-pid-residuals "$GOLDEN_PID_CACHE" \
  --higgs-cache "$CACHE_DIR/higgs/higgs_candidates.csv" \
  --tau32-cache "$CACHE_DIR/tau32/tt_hadronic_ak8_pt450_tau32_values.npz" \
  --met-cache "$CACHE_DIR/met/sm_mixture_met_histograms.npz" \
  --met-run-dir "$SM_VQ8192_RUN" \
  --jet-residual-cache "$CACHE_DIR/jets/sm_mixture_jet_residuals.npz" \
  --sm-pid-residuals "$CACHE_DIR/sm_pid/pid_conditional_pull_values.npz" \
  --output-dir "$WRITEUP_DIR"
```

By default, missing inputs make the renderer fail after it writes the status
README. During plot development, `--allow-incomplete` renders every available
product and records missing inputs under **Pending inputs**.
