"""ORBIT-style plotting helpers for schema-neutral tokenization outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import gabbro.plotting.utils as plot_utils

ORIGINAL_COLOR = plot_utils.DEFAULT_COLORS[0]
RECONSTRUCTED_COLOR = plot_utils.DEFAULT_COLORS[1]
RESIDUAL_COLOR = plot_utils.DEFAULT_COLORS[1]
TRUTH_REFERENCE_COLOR = "grey"
RUN_COLORS = tuple(plot_utils.DEFAULT_COLORS)
CODEBOOK_FAMILY_COLORS = {
    "fsq": plot_utils.DEFAULT_COLORS[0],
    "vq_ste": plot_utils.DEFAULT_COLORS[1],
    "vq_rotation": plot_utils.DEFAULT_COLORS[2],
}
CODEBOOK_FAMILY_MARKERS = {
    "fsq": "o",
    "vq_ste": "s",
    "vq_rotation": "^",
}
CODEBOOK_FAMILY_LABELS = {
    "fsq": "FSQ",
    "vq_ste": "VQ STE",
    "vq_rotation": "VQ rotation",
}
SCATTER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8")
HISTOGRAM_LINEWIDTH = 2
HISTOGRAM_FILL_ALPHA = 0.35
TRUTH_REFERENCE_FILL_ALPHA = 0.35
REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_STYLE = "--"
REFERENCE_LINE_ALPHA = 0.5
RATIO_YLIM = (0.5, 1.5)
FEATURE_FIGSIZE = (21, 8)
ENERGY_FIGSIZE = (16, 5)
RESOLUTION_FIGSIZE = (8, 6)
SUBSTRUCTURE_SIMPLE_FIGSIZE = (18, 8)
MONEY_TRIPLET_FIGSIZE = (21, 6)
RATIO_HEIGHT_RATIOS = (3, 1)
RATIO_HSPACE = 0.08
RATIO_WSPACE = 0.25
ATTENTION_DELTA_FIGSIZE = (6, 5)
ATTENTION_MAP_FIGSIZE = (5, 4)
SCATTER_FIGSIZE = (8, 6)
KINEMATIC_LABELS = {
    "pT": r"$p_T$",
    "Eta": r"$\eta$",
    "Phi": r"$\phi$",
}
KINEMATIC_RESIDUAL_LABELS = {
    "pT": r"$p_T^\mathrm{reco} - p_T^\mathrm{orig}$",
    "Eta": r"$\eta^\mathrm{reco} - \eta^\mathrm{orig}$",
    "Phi": r"$\phi^\mathrm{reco} - \phi^\mathrm{orig}$",
}

PLOT_TITLES_ENABLED = True


def set_plot_titles_enabled(enabled: bool) -> None:
    global PLOT_TITLES_ENABLED
    PLOT_TITLES_ENABLED = enabled


def _set_title(ax, title, **kwargs) -> None:
    if PLOT_TITLES_ENABLED:
        ax.set_title(title, **kwargs)


def _set_suptitle(fig, title, **kwargs) -> None:
    if PLOT_TITLES_ENABLED:
        fig.suptitle(title, **kwargs)


def _feature_label(feature_name: str) -> str:
    return plot_utils.DEFAULT_LABELS.get(feature_name, feature_name)


def _clean_feature_name(feature_name: str) -> str:
    return feature_name.replace("/", "_").replace(" ", "_")


def _masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return array[mask.astype(bool)]


def _finite_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original) & np.isfinite(reconstructed)
    return original[finite], reconstructed[finite]


def _hist_bins(original: np.ndarray, reconstructed: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.concatenate([original, reconstructed])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value == max_value:
        width = max(abs(min_value) * 0.1, 1e-3)
        min_value -= width
        max_value += width
    return np.linspace(min_value, max_value, n_bins + 1)


def collect_reconstruction_histograms(
    feature_names: list[str],
    x_np: np.ndarray,
    x_hat_np: np.ndarray,
    mask_np: np.ndarray,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    """Collect per-feature reconstruction histograms from padded model arrays."""
    histograms = {}
    for i, feature_name in enumerate(feature_names):
        original, reconstructed = _finite_pair(
            _masked_values(x_np[..., i], mask_np),
            _masked_values(x_hat_np[..., i], mask_np),
        )
        bins = _hist_bins(original, reconstructed, n_bins)
        clean_name = _clean_feature_name(feature_name)
        histograms[f"{clean_name}_orig_counts"] = np.histogram(
            original,
            bins=bins,
            density=True,
        )[0]
        histograms[f"{clean_name}_reco_counts"] = np.histogram(
            reconstructed,
            bins=bins,
            density=True,
        )[0]
        histograms[f"{clean_name}_bins"] = bins
        diff_counts, diff_bins = np.histogram(
            reconstructed - original,
            bins=n_bins,
            density=True,
        )
        histograms[f"{clean_name}_diff_counts"] = diff_counts
        histograms[f"{clean_name}_diff_bins"] = diff_bins
    return histograms


def _hist_step(ax, bins, counts, label=None, color=RECONSTRUCTED_COLOR):
    ax.stairs(
        counts,
        bins,
        label=label,
        color=color,
        linewidth=HISTOGRAM_LINEWIDTH,
    )


def _hist_fill(ax, bins, counts, label=None, color=ORIGINAL_COLOR):
    ax.stairs(
        counts,
        bins,
        label=label,
        color=color,
        fill=True,
        alpha=HISTOGRAM_FILL_ALPHA,
        linewidth=0,
    )


def _plot_original_reconstructed_histograms(ax, bins, original, reconstructed):
    _hist_fill(ax, bins, original, label="Original", color=ORIGINAL_COLOR)
    _hist_step(
        ax,
        bins,
        reconstructed,
        label="Reconstructed",
        color=RECONSTRUCTED_COLOR,
    )


def _plot_ratio_histogram(
    ax,
    bins,
    numerator,
    denominator,
    label=None,
    color=RECONSTRUCTED_COLOR,
):
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    _hist_step(ax, bins, ratio, label=label, color=color)


def _single_run_residual_series(counts):
    return [(counts, None, RESIDUAL_COLOR)]


def _triplet_axes(fig, ratio_indices=(0,)):
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = []
    ratio_axes = {}
    for i in range(3):
        if i in ratio_indices:
            axis = fig.add_subplot(grid[0, i])
            ratio_axes[i] = fig.add_subplot(grid[1, i], sharex=axis)
        else:
            axis = fig.add_subplot(grid[:, i])
        axes.append(axis)
    return axes, ratio_axes


def _pair_axes(fig, with_first_ratio=False):
    if not with_first_ratio:
        return fig.subplots(1, 2), {}

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[:, 1]),
    ]
    return axes, {0: fig.add_subplot(grid[1, 0], sharex=axes[0])}


def _configure_ratio_axis(ax, xlabel):
    ax.axhline(
        1.0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.set_ylim(*RATIO_YLIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reco / orig")


def _adjust_ratio_layout(fig):
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.88)


def _grid(n_items: int, max_cols: int = 3, figsize_per_axis=(5.0, 3.6)):
    n_cols = min(max_cols, max(n_items, 1))
    n_rows = math.ceil(max(n_items, 1) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_axis[0] * n_cols, figsize_per_axis[1] * n_rows),
        squeeze=False,
    )
    return fig, axes.flatten()


def plot_feature_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    mse_per_feature: np.ndarray | None = None,
    title: str = "Original vs. reconstructed features",
):
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        bins = histograms[f"{clean_name}_bins"]
        _hist_fill(
            ax,
            bins,
            histograms[f"{clean_name}_orig_counts"],
            label="Original",
            color=ORIGINAL_COLOR,
        )
        _hist_step(
            ax,
            bins,
            histograms[f"{clean_name}_reco_counts"],
            label="Reconstructed",
            color=RECONSTRUCTED_COLOR,
        )
        metric = ""
        if mse_per_feature is not None:
            metric = f" (MSE: {mse_per_feature[i]:.4g})"
        _set_title(ax, f"{_feature_label(feature_name)}{metric}")
        ax.set_xlabel(_feature_label(feature_name))
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_residual_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    title: str = "Reconstructed minus original",
):
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        _hist_step(
            ax,
            histograms[f"{clean_name}_diff_bins"],
            histograms[f"{clean_name}_diff_counts"],
            color=RESIDUAL_COLOR,
        )
        ax.axvline(
            0.0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(f"Reco - original {_feature_label(feature_name)}")
        ax.set_ylabel("Density")
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _validate_data_level(data_level: str) -> None:
    if data_level not in ("particle", "jet"):
        raise ValueError(f"Unsupported data level: {data_level}")


def _safe_density_hist(values, bins):
    counts, _ = np.histogram(values, bins=bins, density=True)
    return np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def _physical_feature_bins(feature_name: str, original: np.ndarray, reconstructed: np.ndarray):
    values = np.concatenate([original, reconstructed])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, 50)
    if feature_name == "pT":
        min_val = max(float(values.min()), 1e-8)
        max_val = max(float(values.max()), min_val * 1.01)
        return np.logspace(np.log10(min_val), np.log10(max_val), 50)
    min_val = float(values.min())
    max_val = float(values.max())
    if min_val == max_val:
        width = max(abs(min_val) * 0.1, 1e-3)
        min_val -= width
        max_val += width
    return np.linspace(min_val, max_val, 50)


def collect_physical_reconstruction_histograms(
    feature_names: list[str],
    x_np: np.ndarray,
    x_hat_np: np.ndarray,
    true_jet_pts=(),
    reco_jet_pts=(),
    true_jet_masses=(),
    reco_jet_masses=(),
    true_tau32s=(),
    reco_tau32s=(),
    true_missing_ets=(),
    reco_missing_ets=(),
    data_level: str = "particle",
) -> dict[str, np.ndarray]:
    """Collect ORBIT paper-style histograms in physical coordinates.

    `x_np` and `x_hat_np` are flattened valid objects with columns
    `[Eta, Phi, pT]`.
    """
    _validate_data_level(data_level)
    histograms = {}

    if len(true_jet_pts) > 0:
        true_jet_pts = np.asarray(true_jet_pts)
        reco_jet_pts = np.asarray(reco_jet_pts)
        fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
        counts, bins = np.histogram(fractional_diff, bins=50, range=(-0.5, 0.5))
        histograms["jet_pt_resolution_counts"] = counts
        histograms["jet_pt_resolution_bins"] = bins

    for i, feature_name in enumerate(feature_names):
        original, reconstructed = _finite_pair(x_np[:, i], x_hat_np[:, i])
        bins = _physical_feature_bins(feature_name, original, reconstructed)
        clean_name = _clean_feature_name(feature_name)
        histograms[f"{clean_name}_orig_counts"] = _safe_density_hist(original, bins)
        histograms[f"{clean_name}_reco_counts"] = _safe_density_hist(reconstructed, bins)
        histograms[f"{clean_name}_bins"] = bins
        diff_counts, diff_bins = np.histogram(
            reconstructed - original,
            bins=50,
            density=True,
        )
        histograms[f"{clean_name}_diff_counts"] = np.nan_to_num(diff_counts)
        histograms[f"{clean_name}_diff_bins"] = diff_bins

    energy_orig = x_np[:, 2] * np.cosh(x_np[:, 0])
    energy_reco = x_hat_np[:, 2] * np.cosh(x_hat_np[:, 0])
    finite_energy = np.isfinite(energy_orig) & np.isfinite(energy_reco)
    energy_orig = energy_orig[finite_energy]
    energy_reco = energy_reco[finite_energy]
    if energy_orig.size > 0:
        min_val = max(float(min(energy_orig.min(), energy_reco.min())), 1e-8)
        max_val = max(float(max(energy_orig.max(), energy_reco.max())), min_val * 1.01)
        energy_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)
        histograms["energy_orig_counts"] = _safe_density_hist(energy_orig, energy_bins)
        histograms["energy_reco_counts"] = _safe_density_hist(energy_reco, energy_bins)
        histograms["energy_bins"] = energy_bins
        counts, bins = np.histogram(energy_reco - energy_orig, bins=50, density=True)
        histograms["energy_residuals_counts"] = np.nan_to_num(counts)
        histograms["energy_residuals_bins"] = bins

    if data_level == "particle" and len(true_missing_ets) > 0:
        true_missing_ets = np.asarray(true_missing_ets)
        reco_missing_ets = np.asarray(reco_missing_ets)
        max_missing_et = max(float(true_missing_ets.max()), float(reco_missing_ets.max()), 1e-8)
        missing_et_bins = np.linspace(0, max_missing_et, 50)
        histograms["missing_et_orig_counts"] = _safe_density_hist(
            true_missing_ets,
            missing_et_bins,
        )
        histograms["missing_et_reco_counts"] = _safe_density_hist(
            reco_missing_ets,
            missing_et_bins,
        )
        histograms["missing_et_bins"] = missing_et_bins

    if data_level == "particle" and len(true_jet_masses) > 0:
        true_jet_masses = np.asarray(true_jet_masses)
        reco_jet_masses = np.asarray(reco_jet_masses)
        true_tau32s = np.asarray(true_tau32s)
        reco_tau32s = np.asarray(reco_tau32s)

        mass_bins = np.linspace(0, 600, 50)
        histograms["jet_mass_orig_counts"] = _safe_density_hist(
            true_jet_masses,
            mass_bins,
        )
        histograms["jet_mass_reco_counts"] = _safe_density_hist(
            reco_jet_masses,
            mass_bins,
        )
        histograms["jet_mass_bins"] = mass_bins

        mass_diff_bins = np.linspace(-50, 50, 50)
        histograms["jet_mass_diff_counts"] = _safe_density_hist(
            reco_jet_masses - true_jet_masses,
            mass_diff_bins,
        )
        histograms["jet_mass_diff_bins"] = mass_diff_bins

        tau_diff_bins = np.linspace(-0.4, 0.4, 50)
        histograms["tau32_diff_counts"] = _safe_density_hist(
            reco_tau32s - true_tau32s,
            tau_diff_bins,
        )
        histograms["tau32_diff_bins"] = tau_diff_bins

    return histograms


def plot_physical_residual_histogram(
    ax,
    bins,
    series,
    xlabel,
    title=None,
    ylabel="Density",
):
    for counts, label, color in series:
        _hist_step(ax, bins, counts, label=label, color=color)
    ax.axvline(
        0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    if title:
        _set_title(ax, title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any(label for _, label, _ in series):
        ax.legend(prop={"size": 10})


def plot_physical_feature_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    reconstructed_series=None,
    mse_per_feature=None,
    include_all_ratios: bool = False,
    title: str = "Original vs. Reconstructed Features",
):
    plot_utils.set_mpl_style()
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=FEATURE_FIGSIZE)
    ratio_indices = (0, 1, 2) if include_all_ratios else (0,)
    axes, ratio_axes = _triplet_axes(fig, ratio_indices=ratio_indices)
    _set_suptitle(fig, title, fontsize=16)

    for axis_idx, feature_idx in enumerate((2, 0, 1)):
        axis = axes[axis_idx]
        feature_name = feature_names[feature_idx]
        clean_name = _clean_feature_name(feature_name)
        bins = histograms[f"{clean_name}_bins"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                histograms[f"{clean_name}_reco_counts"],
            )
        else:
            _hist_fill(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
            )
        for data, label, color in reconstructed_series:
            if not single_run:
                _hist_step(
                    axis,
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    label=label,
                    color=color,
                )
            if axis_idx in ratio_axes:
                _plot_ratio_histogram(
                    ratio_axes[axis_idx],
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    histograms[f"{clean_name}_orig_counts"],
                    label=label,
                    color=color,
                )

        metric = ""
        if mse_per_feature is not None:
            metric = f" (MSE: {mse_per_feature[feature_idx]:.4f})"
        _set_title(axis, f"{feature_name}{metric}")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Density")
        axis.legend(prop={"size": 10})
        if feature_name == "pT":
            axis.set_xscale("log")
        if axis_idx in ratio_axes:
            _configure_ratio_axis(ratio_axes[axis_idx], xlabel=feature_name)
            axis.tick_params(labelbottom=False)

    _adjust_ratio_layout(fig)
    return fig


def plot_energy_histograms(
    histograms: dict[str, np.ndarray],
    reconstructed_series=None,
    include_ratio: bool = False,
    title: str = "Original vs. Reconstructed Energy (m=0)",
):
    plot_utils.set_mpl_style()
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=ENERGY_FIGSIZE)
    axes, ratio_axes = _pair_axes(fig, with_first_ratio=include_ratio)
    bins = histograms["energy_bins"]
    if single_run:
        _plot_original_reconstructed_histograms(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            histograms["energy_reco_counts"],
        )
    else:
        _hist_fill(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            label="Original (Truth)",
            color=TRUTH_REFERENCE_COLOR,
        )
    for data, label, color in reconstructed_series:
        if not single_run:
            _hist_step(axes[0], bins, data["energy_reco_counts"], label=label, color=color)
        if include_ratio:
            _plot_ratio_histogram(
                ratio_axes[0],
                bins,
                data["energy_reco_counts"],
                histograms["energy_orig_counts"],
                label=label,
                color=color,
            )
    axes[0].set_xscale("log")
    _set_title(axes[0], title)
    axes[0].set_xlabel("Energy [GeV]")
    axes[0].set_ylabel("Density")
    axes[0].legend(prop={"size": 10})
    if include_ratio:
        _configure_ratio_axis(ratio_axes[0], xlabel="Energy [GeV]")
        axes[0].tick_params(labelbottom=False)

    plot_physical_residual_histogram(
        axes[1],
        histograms["energy_residuals_bins"],
        _single_run_residual_series(histograms["energy_residuals_counts"]),
        xlabel=r"$E^\mathrm{reco} - E^\mathrm{orig}$ [GeV]",
        title=r"Energy Residuals: $E^\mathrm{reco} - E^\mathrm{orig}$",
    )
    if include_ratio:
        _adjust_ratio_layout(fig)
    else:
        plt.tight_layout()
    return fig


def plot_missing_transverse_energy(histograms: dict[str, np.ndarray]):
    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
    bins = histograms["missing_et_bins"]
    _plot_original_reconstructed_histograms(
        ax,
        bins,
        histograms["missing_et_orig_counts"],
        histograms["missing_et_reco_counts"],
    )
    ax.set_xlabel(r"Missing transverse energy $E_T^\mathrm{miss}$ [GeV]")
    ax.set_ylabel("Density")
    _set_title(ax, "Missing Transverse Energy")
    ax.legend(prop={"size": 10})
    plt.tight_layout()
    return fig


def plot_paper_kinematic_distributions(
    histograms: dict[str, np.ndarray],
    data_level: str,
    reconstructed_series=None,
):
    plot_utils.set_mpl_style()
    _validate_data_level(data_level)
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        bins = histograms[f"{feature_name}_bins"]
        original = histograms[f"{feature_name}_orig_counts"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                original,
                histograms[f"{feature_name}_reco_counts"],
            )
        else:
            _hist_fill(
                axis,
                bins,
                original,
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
            )
            for data, label, color in reconstructed_series:
                _hist_step(
                    axis,
                    bins,
                    data[f"{feature_name}_reco_counts"],
                    label=label,
                    color=color,
                )
        if feature_name == "pT":
            axis.set_xscale("log")
        axis.set_xlabel(feature_label)
        axis.set_ylabel("Density")
        _set_title(axis, f"{feature_label} distribution")
        axis.legend(prop={"size": 10})

    _set_suptitle(fig, f"{data_level.capitalize()} kinematics: original vs. reconstructed")
    plt.tight_layout()
    return fig


def plot_paper_kinematic_differences(
    histograms: dict[str, np.ndarray],
    data_level: str,
    difference_series=None,
):
    plot_utils.set_mpl_style()
    _validate_data_level(data_level)
    difference_series = difference_series or [(histograms, None, RESIDUAL_COLOR)]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        plot_physical_residual_histogram(
            axis,
            histograms[f"{feature_name}_diff_bins"],
            [
                (data[f"{feature_name}_diff_counts"], label, color)
                for data, label, color in difference_series
            ],
            xlabel=KINEMATIC_RESIDUAL_LABELS[feature_name],
            title=f"{feature_label} residuals",
        )

    _set_suptitle(fig, f"{data_level.capitalize()} kinematic residuals")
    plt.tight_layout()
    return fig


def paper_reconstruction_plots(
    histograms: dict[str, np.ndarray],
    data_level: str,
) -> dict[str, object]:
    return {
        "paper_kinematic_distributions": plot_paper_kinematic_distributions(
            histograms,
            data_level,
        ),
        "paper_kinematic_differences": plot_paper_kinematic_differences(
            histograms,
            data_level,
        ),
    }


def physical_reconstruction_plots(
    feature_names: list[str],
    mse_per_feature: np.ndarray,
    histograms: dict[str, np.ndarray],
    data_level: str = "particle",
    include_all_ratios: bool = False,
) -> dict[str, object]:
    """Build ORBIT paper and exploratory single-run reconstruction figures."""
    figures = paper_reconstruction_plots(histograms, data_level)

    if "jet_pt_resolution_counts" in histograms:
        fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
        _hist_step(
            ax,
            histograms["jet_pt_resolution_bins"],
            histograms["jet_pt_resolution_counts"],
            color=RESIDUAL_COLOR,
        )
        ax.axvline(
            0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        ax.set_xlabel(
            r"Fractional $p_T$ Resolution: "
            r"$(p_T^\mathrm{reco} - p_T^\mathrm{true}) / p_T^\mathrm{true}$"
        )
        ax.set_ylabel("Number of Jets")
        _set_title(ax, "Jet Transverse Momentum Recovery")
        figures["jet_pt_resolution"] = fig

    figures["kinematics"] = plot_physical_feature_histograms(
        histograms,
        feature_names,
        mse_per_feature=mse_per_feature,
        include_all_ratios=include_all_ratios,
        title=f"{data_level.capitalize()} Kinematics: Original vs. Reconstructed",
    )
    if "energy_bins" in histograms:
        figures["energy_residuals"] = plot_energy_histograms(
            histograms,
            include_ratio=include_all_ratios,
            title=f"{data_level.capitalize()} Energy: Original vs. Reconstructed (m=0)",
        )
    if data_level == "particle" and "missing_et_bins" in histograms:
        figures["paper_missing_transverse_energy"] = plot_missing_transverse_energy(
            histograms,
        )
    if data_level == "particle" and "jet_mass_orig_counts" in histograms:
        fig, axes = plt.subplots(1, 3, figsize=SUBSTRUCTURE_SIMPLE_FIGSIZE)
        _set_suptitle(fig, "Jet Substructure", fontsize=16)
        _plot_original_reconstructed_histograms(
            axes[0],
            histograms["jet_mass_bins"],
            histograms["jet_mass_orig_counts"],
            histograms["jet_mass_reco_counts"],
        )
        _set_title(axes[0], "Jet Mass")
        axes[0].set_xlabel("Jet Mass [GeV]")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        plot_physical_residual_histogram(
            axes[1],
            histograms["jet_mass_diff_bins"],
            _single_run_residual_series(histograms["jet_mass_diff_counts"]),
            xlabel=r"$m^\mathrm{reco} - m^\mathrm{orig}$ [GeV]",
            title="Jet Mass Residuals",
        )
        plot_physical_residual_histogram(
            axes[2],
            histograms["tau32_diff_bins"],
            _single_run_residual_series(histograms["tau32_diff_counts"]),
            xlabel=r"$\tau_{32}^\mathrm{reco} - \tau_{32}^\mathrm{orig}$",
            title=r"$\tau_{32}$ Residuals",
        )
        plt.tight_layout()
        figures["jet_substructure"] = fig

    return figures


def plot_codebook_histogram(code_idx: np.ndarray, num_codes: int | None = None):
    plot_utils.set_mpl_style()
    codes = np.asarray(code_idx).reshape(-1)
    codes = codes[np.isfinite(codes)]
    fig, ax = plt.subplots(figsize=(8, 5))
    xlabel = "Token ID"
    if codes.size == 0:
        ax.hist(codes, bins=np.arange(2) - 0.5, color=RECONSTRUCTED_COLOR, histtype="step")
    else:
        unique_codes, counts = np.unique(codes.astype(np.int64), return_counts=True)
        dense_safe = (
            num_codes is not None
            and num_codes <= 4096
            and unique_codes.max(initial=0) <= 4096
        )
        if dense_safe:
            bins = np.arange(int(unique_codes.max()) + 2) - 0.5
            ax.hist(codes, bins=bins, color=RECONSTRUCTED_COLOR, histtype="step")
        else:
            order = np.argsort(counts)[::-1]
            max_bars = 100
            selected = order[:max_bars]
            ax.bar(
                np.arange(len(selected)),
                counts[selected],
                color=RECONSTRUCTED_COLOR,
                alpha=HISTOGRAM_FILL_ALPHA,
            )
            ax.set_xticks([])
            xlabel = f"Active token ID, top {len(selected)} by count"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    utilization_denominator = num_codes or max(len(np.unique(codes)), 1)
    utilization = len(np.unique(codes)) / utilization_denominator
    _set_title(ax, f"Codebook utilization: {utilization:.3f}")
    fig.tight_layout()
    return fig


def attention_delta_eta_phi_figure(
    weights: torch.Tensor,
    x: torch.Tensor,
    valid: torch.Tensor,
    title: str,
    exclude_self: bool = False,
):
    """Plot attention versus angular distance.

    For absolute ORBIT features this expects feature order
    `[eta_scaled, cos(phi), sin(phi), pt_transformed]`.
    """
    attn = weights.mean(dim=1)
    eta = x[..., 0] * 3.0
    phi = torch.atan2(x[..., 2], x[..., 1])

    deta = eta[:, :, None] - eta[:, None, :]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi

    pair_mask = valid[:, :, None] & valid[:, None, :]
    if exclude_self:
        self_mask = torch.eye(pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device)
        pair_mask = pair_mask & ~self_mask[None, :, :]
    if not pair_mask.any():
        return None

    deta_np = deta[pair_mask].detach().cpu().numpy()
    dphi_np = dphi[pair_mask].detach().cpu().numpy()
    weight_np = attn[pair_mask].detach().cpu().numpy()

    weight_sum, eta_edges, phi_edges = np.histogram2d(
        deta_np,
        dphi_np,
        bins=(60, 64),
        range=((-6.0, 6.0), (-math.pi, math.pi)),
        weights=weight_np,
    )
    pair_count, _, _ = np.histogram2d(deta_np, dphi_np, bins=(eta_edges, phi_edges))
    hist = np.divide(
        weight_sum,
        pair_count,
        out=np.zeros_like(weight_sum),
        where=pair_count > 0,
    )

    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=ATTENTION_DELTA_FIGSIZE)
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
        aspect="auto",
    )
    _set_title(ax, title)
    ax.set_xlabel(r"$\Delta\eta = \eta_\mathrm{query} - \eta_\mathrm{key}$")
    ax.set_ylabel(r"$\Delta\phi = \phi_\mathrm{query} - \phi_\mathrm{key}$")
    fig.colorbar(im, ax=ax, label="mean attention weight per pair")
    return fig


def attention_map_figure(matrix, title: str):
    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=ATTENTION_MAP_FIGSIZE)
    im = ax.imshow(matrix, vmin=0.0, vmax=max(float(np.max(matrix)), 1e-6), aspect="auto")
    _set_title(ax, title)
    ax.set_xlabel("key token")
    ax.set_ylabel("query token")
    fig.colorbar(im, ax=ax)
    return fig


def close_figure(fig) -> None:
    plt.close(fig)


def save_figures(figures: dict[str, object], output_dir: str | Path, suffix: str = "png") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(output_path / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def plot_multirun_feature_histograms(
    runs: list[dict],
    feature_names: list[str],
    title: str = "Reconstruction comparison",
):
    """Overlay reconstructed feature histograms from several runs."""
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    reference = runs[0]["histograms"]
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        _hist_fill(
            ax,
            reference[f"{clean_name}_bins"],
            reference[f"{clean_name}_orig_counts"],
            label="Original",
            color=TRUTH_REFERENCE_COLOR,
        )
        for j, run in enumerate(runs):
            histograms = run["histograms"]
            _hist_step(
                ax,
                histograms[f"{clean_name}_bins"],
                histograms[f"{clean_name}_reco_counts"],
                label=run["label"],
                color=RUN_COLORS[j % len(RUN_COLORS)],
            )
        _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(_feature_label(feature_name))
        ax.set_ylabel("Density")
        ax.legend(prop={"size": 8})
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_multirun_residual_histograms(
    runs: list[dict],
    feature_names: list[str],
    title: str = "Reconstruction residual comparison",
):
    """Overlay reconstructed-minus-original residuals from several runs."""
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        for j, run in enumerate(runs):
            histograms = run["histograms"]
            _hist_step(
                ax,
                histograms[f"{clean_name}_diff_bins"],
                histograms[f"{clean_name}_diff_counts"],
                label=run["label"],
                color=RUN_COLORS[j % len(RUN_COLORS)],
            )
        ax.axvline(
            0.0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(f"Reco - original {_feature_label(feature_name)}")
        ax.set_ylabel("Density")
        ax.legend(prop={"size": 8})
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_multirun_metric(
    records: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    log_x: bool = True,
    log_y: bool = False,
):
    """Plot a scalar metric against total codebook size."""
    plot_utils.set_mpl_style()
    usable_records = [
        record
        for record in records
        if record.get(metric) is not None and record.get("total_codebook_size") is not None
    ]
    if not usable_records:
        return None

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    for i, record in enumerate(usable_records):
        ax.scatter(
            record["total_codebook_size"],
            record[metric],
            label=record["label"],
            color=RUN_COLORS[i % len(RUN_COLORS)],
            marker=SCATTER_MARKERS[i % len(SCATTER_MARKERS)],
        )
    if log_x:
        ax.set_xscale("log")
    if log_y and all(record[metric] > 0 for record in usable_records):
        ax.set_yscale("log")
    ax.set_xlabel("Total codebook size")
    ax.set_ylabel(ylabel)
    _set_title(ax, title)
    ax.legend(prop={"size": 8})
    fig.tight_layout()
    return fig
