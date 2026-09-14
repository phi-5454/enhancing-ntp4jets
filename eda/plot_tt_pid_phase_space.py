#!/usr/bin/env python3
"""Compare hadron and charged-lepton phase-space occupancy in canonical ttbar data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from gabbro.plotting.utils import set_mpl_style

from probe_tt_pid_composition import CHANNEL_SAMPLES, manifest_files, pid_classes


PDG_COLUMN = "L1T_PUPPIPart_PID"
CHARGE_COLUMN = "L1T_PUPPIPart_Charge"
PUPPIW_COLUMN = "L1T_PUPPIPart_PuppiW"
PT_COLUMN = "L1T_PUPPIPart_PT"
ETA_COLUMN = "L1T_PUPPIPart_Eta"
PHI_COLUMN = "L1T_PUPPIPart_Phi"
HADRON_PIDS = (0, 2, 3)
LEPTON_PIDS = (4, 5, 6, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train_val", "test"), default="train_val")
    parser.add_argument("--events-per-channel", type=int, default=2_000)
    parser.add_argument("--puppiw-min", type=float, default=0.05)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="orbit-tokenizer")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group", default="orbit-tt-pid-phase-space")
    parser.add_argument("--wandb-entity")
    return parser.parse_args()


def load_env_file() -> None:
    env_file = os.environ.get("GABBRO_ENV_FILE")
    if not env_file:
        return
    path = Path(os.path.expandvars(os.path.expanduser(env_file)))
    if not path.is_file():
        raise FileNotFoundError(f"GABBRO_ENV_FILE does not exist: {path}")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ[key] = os.path.expandvars(value)


def sample_channel(
    files: list[Path], events: int, puppiw_min: float, max_particles: int
) -> dict[str, np.ndarray]:
    collected = {"hadron": [[], [], []], "lepton": [[], [], []]}
    columns = [PDG_COLUMN, CHARGE_COLUMN, PUPPIW_COLUMN, PT_COLUMN, ETA_COLUMN, PHI_COLUMN]
    events_read = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        missing = set(columns) - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=2_048):
            arrays = {name: batch.column(name).to_pylist() for name in columns}
            for pdg_values, charge_values, weight_values, pt_values, eta_values, phi_values in zip(
                arrays[PDG_COLUMN],
                arrays[CHARGE_COLUMN],
                arrays[PUPPIW_COLUMN],
                arrays[PT_COLUMN],
                arrays[ETA_COLUMN],
                arrays[PHI_COLUMN],
            ):
                pdg = np.asarray(pdg_values, dtype=np.int64)
                charge = np.asarray(charge_values, dtype=np.float64)
                weight = np.asarray(weight_values, dtype=np.float64)
                pt = np.asarray(pt_values, dtype=np.float64)
                eta = np.asarray(eta_values, dtype=np.float64)
                phi = np.asarray(phi_values, dtype=np.float64)
                pid = pid_classes(pdg, charge)
                valid = (
                    np.isfinite(pt)
                    & (pt > 0)
                    & np.isfinite(eta)
                    & np.isfinite(phi)
                    & np.isfinite(weight)
                    & (weight > puppiw_min)
                )
                selected = np.flatnonzero(valid)[:max_particles]
                for category, classes in (("hadron", HADRON_PIDS), ("lepton", LEPTON_PIDS)):
                    keep = selected[np.isin(pid[selected], classes)]
                    collected[category][0].extend(eta[keep])
                    collected[category][1].extend(phi[keep])
                    collected[category][2].extend(pt[keep])
                events_read += 1
                if events_read >= events:
                    return {
                        category: np.column_stack(values).astype(np.float32)
                        for category, values in collected.items()
                    }
    if events_read < events:
        raise RuntimeError(f"Requested {events} events but found only {events_read}")
    raise AssertionError("unreachable")


def merge_channels(channels: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        category: np.concatenate([values[category] for values in channels.values()])
        for category in ("hadron", "lepton")
    }


def histogram_inputs(data: dict[str, np.ndarray]) -> tuple[dict, dict]:
    eta_bins = np.linspace(-4.5, 4.5, 73)
    phi_bins = np.linspace(-np.pi, np.pi, 65)
    log_pt_bins = np.linspace(np.log10(0.25), np.log10(500.0), 67)
    bins = {"eta": eta_bins, "phi": phi_bins, "log_pt": log_pt_bins}
    values = {
        category: {
            "eta": particles[:, 0],
            "phi": particles[:, 1],
            "log_pt": np.log10(np.clip(particles[:, 2], 1e-6, None)),
        }
        for category, particles in data.items()
    }
    return bins, values


def plot_1d(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    set_mpl_style()
    bins, values = histogram_inputs(data)
    labels = {
        "eta": r"$\eta$",
        "phi": r"$\phi$",
        "log_pt": r"$p_T$ [GeV]",
    }
    colors = {"hadron": "#3f90da", "lepton": "#e76300"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    saved = {}
    for axis, variable in zip(axes, ("eta", "phi", "log_pt")):
        for category in ("hadron", "lepton"):
            counts, edges = np.histogram(values[category][variable], bins=bins[variable])
            widths = np.diff(edges)
            density = counts / max(counts.sum(), 1) / widths
            axis.stairs(
                density,
                edges if variable != "log_pt" else 10**edges,
                label=f"{category.capitalize()} (n={len(values[category][variable]):,})",
                color=colors[category],
                linewidth=2,
            )
            saved[f"{category}_{variable}_counts"] = counts
        if variable == "log_pt":
            axis.set_xscale("log")
        axis.set(xlabel=labels[variable], ylabel="Normalized density")
        axis.legend(frameon=False, fontsize=9)
        saved[f"{variable}_bins"] = bins[variable]
    figure.suptitle(r"Canonical $t\bar{t}$ training mixture: selected particle phase space")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return saved


def _normalized_hist2d(x, y, x_bins, y_bins) -> np.ndarray:
    counts, _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins))
    return counts / max(counts.sum(), 1)


def plot_2d(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    set_mpl_style()
    bins, values = histogram_inputs(data)
    pairs = (
        ("eta", "log_pt", r"$\eta$", r"$p_T$ [GeV]"),
        ("phi", "log_pt", r"$\phi$", r"$p_T$ [GeV]"),
        ("eta", "phi", r"$\eta$", r"$\phi$"),
    )
    histograms = {}
    for category in ("hadron", "lepton"):
        for x_name, y_name, _, _ in pairs:
            histograms[f"{category}_{x_name}_{y_name}"] = _normalized_hist2d(
                values[category][x_name], values[category][y_name], bins[x_name], bins[y_name]
            )
    positive = np.concatenate([hist[hist > 0] for hist in histograms.values()])
    norm = mcolors.LogNorm(vmin=max(float(np.quantile(positive, 0.02)), 1e-7),
                           vmax=float(np.max(positive)))
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    image = None
    for row, category in enumerate(("hadron", "lepton")):
        for column, (x_name, y_name, x_label, y_label) in enumerate(pairs):
            x_edges = bins[x_name]
            y_edges = bins[y_name]
            display_y_edges = 10**y_edges if y_name == "log_pt" else y_edges
            image = axes[row, column].pcolormesh(
                x_edges,
                display_y_edges,
                histograms[f"{category}_{x_name}_{y_name}"].T,
                cmap="viridis",
                norm=norm,
                shading="auto",
            )
            if y_name == "log_pt":
                axes[row, column].set_yscale("log")
            axes[row, column].set(
                xlabel=x_label,
                ylabel=y_label,
                title=f"{category.capitalize()} normalized occupancy",
            )
    figure.colorbar(image, ax=axes, label="Fraction of category per bin", shrink=0.9)
    figure.suptitle(r"Canonical $t\bar{t}$ training mixture: two-dimensional occupancy")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {**histograms, **{f"{name}_bins": edges for name, edges in bins.items()}}


def plot_lepton_fraction(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    set_mpl_style()
    bins, values = histogram_inputs(data)
    pairs = (
        ("eta", "log_pt", r"$\eta$", r"$p_T$ [GeV]"),
        ("phi", "log_pt", r"$\phi$", r"$p_T$ [GeV]"),
        ("eta", "phi", r"$\eta$", r"$\phi$"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    saved = {}
    image = None
    for axis, (x_name, y_name, x_label, y_label) in zip(axes, pairs):
        hadrons, _, _ = np.histogram2d(
            values["hadron"][x_name], values["hadron"][y_name],
            bins=(bins[x_name], bins[y_name]),
        )
        leptons, _, _ = np.histogram2d(
            values["lepton"][x_name], values["lepton"][y_name],
            bins=(bins[x_name], bins[y_name]),
        )
        total = hadrons + leptons
        fraction = np.divide(leptons, total, out=np.full_like(total, np.nan), where=total >= 10)
        y_edges = 10**bins[y_name] if y_name == "log_pt" else bins[y_name]
        image = axis.pcolormesh(
            bins[x_name], y_edges, fraction.T, cmap="magma", vmin=0.0, vmax=0.5,
            shading="auto",
        )
        if y_name == "log_pt":
            axis.set_yscale("log")
        axis.set(xlabel=x_label, ylabel=y_label, title="Lepton fraction")
        saved[f"lepton_fraction_{x_name}_{y_name}"] = fraction
        saved[f"total_{x_name}_{y_name}"] = total
    figure.colorbar(image, ax=axes, label=r"$N_\ell/(N_\ell+N_\mathrm{had})$", shrink=0.9)
    figure.suptitle(r"Canonical $t\bar{t}$ training mixture (bins with $N\geq10$)")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return saved


def upload_to_wandb(args: argparse.Namespace, outputs: list[Path], summary: dict) -> str:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.output_dir.name,
        group=args.wandb_group,
        job_type="eda",
        config={
            "split": args.split,
            "events_per_channel": args.events_per_channel,
            "puppiw_min": args.puppiw_min,
            "max_particles": args.max_particles,
        },
    )
    try:
        run.log({f"tt_pid_phase_space/{path.stem}": wandb.Image(str(path)) for path in outputs})
        run.log(
            {
                "tt_pid_phase_space/counts": wandb.Table(
                    columns=["category", "particles", "fraction"],
                    data=[
                        [name, values["particles"], values["fraction"]]
                        for name, values in summary["categories"].items()
                    ],
                )
            }
        )
        artifact = wandb.Artifact(f"tt-pid-phase-space-{run.id}", type="eda")
        for path in outputs + [args.output_dir / "tt_pid_phase_space_histograms.npz",
                               args.output_dir / "tt_pid_phase_space_summary.json"]:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        return run.url
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    if args.events_per_channel < 1 or args.max_particles < 1:
        raise ValueError("--events-per-channel and --max-particles must be positive")
    load_env_file()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest_dir.expanduser().resolve()
    channels = {
        channel: sample_channel(
            manifest_files(manifest_dir / f"{sample}_{args.split}.txt"),
            args.events_per_channel,
            args.puppiw_min,
            args.max_particles,
        )
        for channel, sample in CHANNEL_SAMPLES.items()
    }
    data = merge_channels(channels)
    outputs = [
        args.output_dir / "tt_pid_phase_space_1d.png",
        args.output_dir / "tt_pid_phase_space_2d.png",
        args.output_dir / "tt_pid_phase_space_lepton_fraction.png",
    ]
    histograms = {}
    histograms.update(plot_1d(data, outputs[0]))
    histograms.update(plot_2d(data, outputs[1]))
    histograms.update(plot_lepton_fraction(data, outputs[2]))
    np.savez_compressed(args.output_dir / "tt_pid_phase_space_histograms.npz", **histograms)
    total = sum(len(values) for values in data.values())
    summary = {
        "split": args.split,
        "events_per_channel": args.events_per_channel,
        "events_total": args.events_per_channel * len(CHANNEL_SAMPLES),
        "puppiw_min": args.puppiw_min,
        "max_particles": args.max_particles,
        "categories": {
            name: {"particles": len(values), "fraction": len(values) / total}
            for name, values in data.items()
        },
        "lepton_to_hadron": len(data["lepton"]) / len(data["hadron"]),
    }
    (args.output_dir / "tt_pid_phase_space_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if args.wandb:
        print(f"W&B: {upload_to_wandb(args, outputs, summary)}")


if __name__ == "__main__":
    main()
