#!/usr/bin/env python3
"""Compare selected Z-candidate muons with their inclusive particle phase space."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from gabbro.plotting.utils import set_mpl_style

from probe_tt_pid_composition import pid_classes


PARTICLE_COLUMNS = (
    "L1T_PUPPIPart_PID",
    "L1T_PUPPIPart_Charge",
    "L1T_PUPPIPart_PuppiW",
    "L1T_PUPPIPart_PT",
    "L1T_PUPPIPart_Eta",
    "L1T_PUPPIPart_Phi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--puppiw-min", type=float, default=0.05)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="orbit-tokenizer")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group", default="orbit-z-mumu-phase-space")
    parser.add_argument("--wandb-entity")
    return parser.parse_args()


def load_env_file() -> None:
    env_file = os.environ.get("GABBRO_ENV_FILE")
    if not env_file:
        return
    path = Path(os.path.expandvars(os.path.expanduser(env_file)))
    if not path.is_file():
        raise FileNotFoundError(path)
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


def read_candidates(path: Path) -> tuple[dict[Path, set[int]], np.ndarray]:
    requested_rows: dict[Path, set[int]] = defaultdict(set)
    candidates = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            requested_rows[Path(row["source_file"])].add(int(row["source_row"]))
            for prefix in ("original_muon", "original_antimuon"):
                values = [float(row[f"{prefix}_{name}"]) for name in ("eta", "phi", "pt")]
                if np.all(np.isfinite(values)):
                    candidates.append(values)
    if not requested_rows:
        raise ValueError(f"Candidate CSV has no event rows: {path}")
    return requested_rows, np.asarray(candidates, dtype=np.float32)


def read_inclusive_particles(
    requested_rows: dict[Path, set[int]], puppiw_min: float, max_particles: int
) -> tuple[np.ndarray, np.ndarray]:
    all_particles: list[list[float]] = []
    all_muons: list[list[float]] = []
    for path, row_numbers in requested_rows.items():
        parquet_file = pq.ParquetFile(path)
        missing = set(PARTICLE_COLUMNS) - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        table = parquet_file.read(columns=list(PARTICLE_COLUMNS))
        arrays = {name: table[name].to_pylist() for name in PARTICLE_COLUMNS}
        for row_number in sorted(row_numbers):
            if row_number >= table.num_rows:
                raise IndexError(f"Row {row_number} is outside {path} ({table.num_rows} rows)")
            pdg = np.asarray(arrays[PARTICLE_COLUMNS[0]][row_number], dtype=np.int64)
            charge = np.asarray(arrays[PARTICLE_COLUMNS[1]][row_number], dtype=np.float64)
            weight = np.asarray(arrays[PARTICLE_COLUMNS[2]][row_number], dtype=np.float64)
            pt = np.asarray(arrays[PARTICLE_COLUMNS[3]][row_number], dtype=np.float64)
            eta = np.asarray(arrays[PARTICLE_COLUMNS[4]][row_number], dtype=np.float64)
            phi = np.asarray(arrays[PARTICLE_COLUMNS[5]][row_number], dtype=np.float64)
            valid = (
                np.isfinite(pt)
                & (pt > 0)
                & np.isfinite(eta)
                & np.isfinite(phi)
                & np.isfinite(weight)
                & (weight > puppiw_min)
            )
            selected = np.flatnonzero(valid)[:max_particles]
            particles = np.column_stack((eta[selected], phi[selected], pt[selected]))
            all_particles.extend(particles.tolist())
            classes = pid_classes(pdg, charge)
            all_muons.extend(particles[np.isin(classes[selected], (6, 7))].tolist())
    return np.asarray(all_particles, dtype=np.float32), np.asarray(all_muons, dtype=np.float32)


def phase_space_inputs(data: dict[str, np.ndarray]) -> tuple[dict, dict]:
    bins = {
        "eta": np.linspace(-4.5, 4.5, 73),
        "phi": np.linspace(-np.pi, np.pi, 65),
        "log_pt": np.linspace(np.log10(0.25), np.log10(500.0), 67),
    }
    values = {
        name: {
            "eta": particles[:, 0],
            "phi": particles[:, 1],
            "log_pt": np.log10(np.clip(particles[:, 2], 1e-6, None)),
        }
        for name, particles in data.items()
    }
    return bins, values


def plot_marginals(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    set_mpl_style()
    bins, values = phase_space_inputs(data)
    labels = {"eta": r"$\eta$", "phi": r"$\phi$", "log_pt": r"$p_T$ [GeV]"}
    colors = {"all_particles": "#777777", "all_muons": "#3f90da", "z_muons": "#e76300"}
    names = {
        "all_particles": "All input particles",
        "all_muons": "All input muons",
        "z_muons": r"Selected $Z$ muons",
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    saved = {}
    for axis, variable in zip(axes, ("eta", "phi", "log_pt")):
        for category in ("all_particles", "all_muons", "z_muons"):
            counts, edges = np.histogram(values[category][variable], bins=bins[variable])
            density = counts / max(counts.sum(), 1) / np.diff(edges)
            axis.stairs(
                density,
                edges if variable != "log_pt" else 10**edges,
                color=colors[category],
                linewidth=2,
                label=f"{names[category]} (n={len(data[category]):,})",
            )
            saved[f"{category}_{variable}_counts"] = counts
        if variable == "log_pt":
            axis.set_xscale("log")
        axis.set(xlabel=labels[variable], ylabel="Normalized density")
        axis.legend(frameon=False, fontsize=9)
        saved[f"{variable}_bins"] = bins[variable]
    figure.suptitle(r"$ZZ\to\mathrm{leptons}$: selected $Z\to\mu\mu$ candidate phase space")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return saved


def plot_eta_pt(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    set_mpl_style()
    eta_bins = np.linspace(-4.5, 4.5, 73)
    pt_bins = np.geomspace(0.25, 500.0, 67)
    histograms = {}
    for category in ("all_muons", "z_muons"):
        hist, _, _ = np.histogram2d(
            data[category][:, 0], data[category][:, 2], bins=(eta_bins, pt_bins)
        )
        histograms[category] = hist / max(hist.sum(), 1)
    positive = np.concatenate([hist[hist > 0] for hist in histograms.values()])
    norm = mcolors.LogNorm(vmin=max(float(np.quantile(positive, 0.02)), 1e-7), vmax=positive.max())
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = None
    for axis, category, title in zip(
        axes,
        ("all_muons", "z_muons"),
        ("All input muons", r"Selected $Z$ muons"),
    ):
        image = axis.pcolormesh(
            eta_bins, pt_bins, histograms[category].T, shading="auto", cmap="viridis", norm=norm
        )
        axis.set_yscale("log")
        axis.set(xlabel=r"$\eta$", ylabel=r"$p_T$ [GeV]", title=title)
    figure.colorbar(image, ax=axes, label="Fraction of population per bin")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {
        "eta_bins": eta_bins,
        "pt_bins": pt_bins,
        "all_muons_eta_pt": histograms["all_muons"],
        "z_muons_eta_pt": histograms["z_muons"],
    }


def upload_to_wandb(args: argparse.Namespace, outputs: list[Path], summary: dict) -> str:
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.output_dir.name,
        group=args.wandb_group,
        job_type="eda",
        config=summary,
    )
    run.log({f"z_mumu_phase_space/{path.stem}": wandb.Image(str(path)) for path in outputs})
    artifact = wandb.Artifact(f"{run.name}-histograms", type="orbit-eda")
    for path in args.output_dir.iterdir():
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    url = run.url
    wandb.finish()
    return url


def main() -> None:
    args = parse_args()
    load_env_file()
    if args.max_particles < 1:
        raise ValueError("--max-particles must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_rows, z_muons = read_candidates(args.candidates_csv)
    all_particles, all_muons = read_inclusive_particles(
        requested_rows, args.puppiw_min, args.max_particles
    )
    data = {"all_particles": all_particles, "all_muons": all_muons, "z_muons": z_muons}
    marginal_path = args.output_dir / "z_mumu_phase_space_marginals.png"
    occupancy_path = args.output_dir / "z_mumu_phase_space_eta_pt.png"
    histograms = plot_marginals(data, marginal_path)
    histograms.update(plot_eta_pt(data, occupancy_path))
    np.savez_compressed(args.output_dir / "z_mumu_phase_space_histograms.npz", **histograms)
    summary = {
        "candidate_csv": str(args.candidates_csv.resolve()),
        "events": sum(len(rows) for rows in requested_rows.values()),
        "puppiw_min": args.puppiw_min,
        "max_particles": args.max_particles,
        "counts": {name: len(values) for name, values in data.items()},
    }
    (args.output_dir / "z_mumu_phase_space_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {marginal_path}")
    print(f"Wrote {occupancy_path}")
    if args.wandb:
        print(f"W&B: {upload_to_wandb(args, [marginal_path, occupancy_path], summary)}")


if __name__ == "__main__":
    main()
