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
REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_STYLE = "--"
REFERENCE_LINE_ALPHA = 0.5
RATIO_YLIM = (0.5, 1.5)
ATTENTION_DELTA_FIGSIZE = (6, 5)
ATTENTION_MAP_FIGSIZE = (5, 4)
SCATTER_FIGSIZE = (8, 6)

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


def plot_codebook_histogram(code_idx: np.ndarray, num_codes: int | None = None):
    plot_utils.set_mpl_style()
    codes = np.asarray(code_idx).reshape(-1)
    codes = codes[np.isfinite(codes)]
    if codes.size == 0:
        bins = np.arange(2) - 0.5
    else:
        max_code = int(codes.max())
        bins = np.arange(max_code + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(codes, bins=bins, color=RECONSTRUCTED_COLOR, histtype="step")
    ax.set_xlabel("Token ID")
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
