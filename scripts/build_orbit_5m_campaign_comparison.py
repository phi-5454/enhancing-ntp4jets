#!/usr/bin/env python
"""Render the joint eight-model 5M campaign comparison from durable caches."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

from gabbro.plotting.orbit import ORIGINAL_COLOR, RUN_COLORS, SCATTER_MARKERS

SUITES = ("ttbar", "sm_mixture", "gghbb", "minbias")
SELECTIONS = ("full500", "puppi128")
PID_LABELS = (r"$h^0$", r"$\gamma$", r"$h^-$", r"$h^+$", r"$e^-$", r"$e^+$", r"$\mu^-$", r"$\mu^+$")


def load_registry(path: Path):
    rows = yaml.safe_load(path.read_text())["models"]
    if len(rows) != 8:
        raise ValueError("Expected eight campaign models")
    for index, row in enumerate(rows):
        row["color"] = RUN_COLORS[index]
        row["linestyle"] = ("-", "--", "-.", ":")[index % 4]
        row["marker"] = SCATTER_MARKERS[index]
    return rows


def save(figure, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(figure)


def density(counts, bins):
    counts = np.asarray(counts, dtype=float)
    return counts / max(np.sum(counts * np.diff(bins)), 1e-12)


def model_cache(cache_dir, suite, model):
    path = cache_dir / suite / f"{model['id']}.npz"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError(path)
    return np.load(path), json.loads(path.with_suffix(".json").read_text())


def external_legend(figure, axes, models):
    handles = [axes[0].plot([], [], color=m["color"], linestyle=m["linestyle"], linewidth=2.0, label=m["label"])[0] for m in models]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False, fontsize=11)


def plot_particle(cache_dir, suite, models):
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    specs = (("pT", r"$p_T^{dec.}-p_T^{orig.}$ [GeV]"), ("Eta", r"$\eta^{dec.}-\eta^{orig.}$"), ("Phi", r"wrapped $\phi^{dec.}-\phi^{orig.}$"))
    for model in models:
        cache, _ = model_cache(cache_dir, suite, model)
        for axis, (feature, xlabel) in zip(axes, specs, strict=True):
            bins = cache[f"particle_{feature}_diff_bins"]
            axis.stairs(cache[f"particle_{feature}_diff_counts"], bins, color=model["color"], linestyle=model["linestyle"], linewidth=1.8)
            axis.set(xlabel=xlabel, yscale="log")
            axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    external_legend(figure, axes, models)
    return figure


def plot_jets(cache_dir, suite, models):
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    specs = (("delta_pt", (-1, 1), r"$(p_T^{dec.}-p_T^{orig.})/p_T^{orig.}$"), ("delta_eta", (-1, 1), r"$\eta^{dec.}-\eta^{orig.}$"), ("delta_phi", (-1, 1), r"wrapped $\phi^{dec.}-\phi^{orig.}$"))
    for model in models:
        cache, _ = model_cache(cache_dir, suite, model)
        for axis, (key, limits, xlabel) in zip(axes, specs, strict=True):
            edges = np.linspace(*limits, 81)
            counts, _ = np.histogram(cache[f"jet_{key}"], bins=edges, density=True)
            axis.stairs(counts, edges, color=model["color"], linestyle=model["linestyle"], linewidth=1.8)
            axis.set(xlabel=xlabel, xlim=limits, yscale="log")
            axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    external_legend(figure, axes, models)
    return figure


def plot_pid(cache_dir, suite, model):
    cache, _ = model_cache(cache_dir, suite, model)
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.1), constrained_layout=True)
    specs = (("delta_eta", (-2, 2), r"$\eta^{dec.}-\eta^{orig.}$"), ("delta_phi", (-1, 1), r"wrapped $\phi^{dec.}-\phi^{orig.}$"), ("delta_log_pt", (-1.5, 1.5), r"$\log(p_T^{dec.}/p_T^{orig.})$"))
    pid = cache["pid_original"].astype(int)
    for axis, (key, limits, xlabel) in zip(axes, specs, strict=True):
        edges = np.linspace(*limits, 81)
        for value, label in enumerate(PID_LABELS):
            axis.hist(cache[f"pid_{key}"][pid == value], bins=edges, density=True, histtype="step", linewidth=1.5, color=RUN_COLORS[value], label=label)
        axis.set(xlabel=xlabel, xlim=limits, yscale="log")
        axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    axes[-1].legend(ncol=2, frameon=False, fontsize=12)
    return figure


def plot_downstream(cache_dir, selection, models, kind):
    suite = "gghbb" if kind == "higgs" else "tau32"
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), constrained_layout=True)
    first, _ = model_cache(cache_dir, suite, models[0])
    if kind == "higgs":
        panels = (("higgs_n_jets_original", "higgs_n_jets_decoded", np.arange(-0.5, 30.5, 1), "Number of AK4 jets"), ("higgs_mass_original", "higgs_mass_decoded", np.arange(40, 405, 5), "Leading-dijet mass [GeV]"))
    else:
        panels = (("original_count", "decoded_count", np.arange(-0.5, 10.5, 1), r"Number of AK8 jets ($p_T\geq450$ GeV)"), ("original_tau32", "decoded_tau32", np.linspace(0, 1.2, 51), r"AK8 jet $\tau_{32}$"))
    for axis, (orig_key, reco_key, bins, xlabel) in zip(axes, panels, strict=True):
        original = first[orig_key]
        axis.hist(original[np.isfinite(original)], bins=bins, density=True, histtype="stepfilled", alpha=0.32, color=ORIGINAL_COLOR, label="Original")
        for model in models:
            cache, _ = model_cache(cache_dir, suite, model)
            values = cache[reco_key]
            axis.hist(values[np.isfinite(values)], bins=bins, density=True, histtype="step", linewidth=1.8, color=model["color"], linestyle=model["linestyle"], label=model["label"])
        axis.set(xlabel=xlabel, ylabel="Normalized density")
    axes[0].set_yscale("log")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False, fontsize=11)
    return figure


def plot_rate_distortion(cache_dir, canonical_manifest, models):
    records = json.loads(canonical_manifest.read_text())
    canonical = sorted((float(row["metrics/rate/marginal_bits_per_input_particle"]) / 40.0, float(row["plot_metrics/reco_mse_total"])) for row in records if row.get("plot_family") == "VQ STE" and row.get("metrics/rate/marginal_bits_per_input_particle") is not None and row.get("plot_metrics/reco_mse_total") is not None)
    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    axis.plot([x for x, _ in canonical], [y for _, y in canonical], color="0.3", marker="s", markerfacecolor="white", linewidth=1.8, label="Canonical 200k full-500 tt VQ-STE")
    values = []
    for model in models:
        _, summary = model_cache(cache_dir, "ttbar", model)
        x = summary["metrics/rate/marginal_bits_per_input_particle"] / 40.0
        y = summary["metrics/mse_total"]
        axis.scatter(x, y, s=90, color=model["color"], marker=model["marker"], edgecolor="black", linewidth=0.6, label=model["gimmick"], zorder=3)
        values.append({"model": model["id"], "gimmick": model["gimmick"], "rate_over_40bit": x, "mse": y})
    axis.set(xlabel="Marginal compressed rate / 40-bit input candidate", ylabel="Reconstruction MSE", yscale="log")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=10, ncol=2)
    return figure, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--fixed-comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = load_registry(args.registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = []
    for suite in SUITES:
        save(plot_particle(args.cache_dir, suite, models), args.output_dir / "residuals" / "particles" / suite)
        save(plot_jets(args.cache_dir, suite, models), args.output_dir / "residuals" / "ak4_jets" / suite)
        products.extend((f"residuals/particles/{suite}", f"residuals/ak4_jets/{suite}"))
        for model in (row for row in models if row.get("pid_weighted")):
            save(plot_pid(args.cache_dir, suite, model), args.output_dir / "pid_residuals" / model["selection"] / suite)
            products.append(f"pid_residuals/{model['selection']}/{suite}")
    for selection in SELECTIONS:
        selected = [model for model in models if model["selection"] == selection]
        for kind in ("higgs", "tau32"):
            save(plot_downstream(args.cache_dir, selection, selected, kind), args.output_dir / "downstream" / kind / selection)
            products.append(f"downstream/{kind}/{selection}")
    fixed_target = args.output_dir / "training_scale" / "matched_tt"
    fixed_target.mkdir(parents=True, exist_ok=True)
    for path in args.fixed_comparison_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, fixed_target / path.name)
    figure, rate_values = plot_rate_distortion(args.cache_dir, args.canonical_manifest, models)
    save(figure, args.output_dir / "training_scale" / "all_eight_rate_distortion")
    (args.output_dir / "training_scale" / "all_eight_rate_distortion.json").write_text(json.dumps(rate_values, indent=2) + "\n")
    products.extend(("training_scale/matched_tt", "training_scale/all_eight_rate_distortion"))
    status = {"status": "complete", "models": [{k: v for k, v in row.items() if k not in {"color", "linestyle", "marker"}} for row in models], "products": products, "cache_dir": str(args.cache_dir), "output_dir": str(args.output_dir)}
    (args.output_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.output_dir / "README.md").write_text("# Joint 5M campaign comparison\n\nAll plots are title-free PDF and 300-dpi PNG products. See `STATUS.json` for the complete model and product inventory. AK4 residual jets use unconstrained minimum-total-$\\Delta R$ Hungarian assignment.\n")
    print(f"Rendered {len(products)} plot groups in {args.output_dir}")


if __name__ == "__main__":
    main()
