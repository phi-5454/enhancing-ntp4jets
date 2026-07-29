#!/usr/bin/env python3
"""Render one ORBIT particle event and its FastJet jets in eta--phi space.

Example
-------
uv run --locked python scripts/visualize_orbit_event.py sample.parquet 12 \
    --radius 0.8 --output event_12.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pyarrow.parquet as pq
import vector

from gabbro.plotting.utils import set_mpl_style
from gabbro.plotting.orbit import event_marker_areas


vector.register_awkward()


PARTICLE_COLUMNS = (
    "L1T_PUPPIPart_Eta",
    "L1T_PUPPIPart_Phi",
    "L1T_PUPPIPart_PT",
)
PUPPI_WEIGHT_COLUMN = "L1T_PUPPIPart_PuppiW"
PHI_LIMITS = (-np.pi, np.pi)


def _marker_areas(pt: np.ndarray) -> np.ndarray:
    """Return bounded, logarithmically scaled scatter-marker areas in points squared."""
    return event_marker_areas(pt)


def _read_event(
    path: Path,
    event_index: int,
    puppi_weight_min: float | None,
    max_particles: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Read and select one particle event from an ORBIT-format Parquet file."""
    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema_arrow.names)
    missing_columns = sorted(set(PARTICLE_COLUMNS) - schema_names)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required ORBIT particle columns: {', '.join(missing_columns)}"
        )
    if not 0 <= event_index < parquet_file.metadata.num_rows:
        raise IndexError(
            f"event index {event_index} is outside [0, {parquet_file.metadata.num_rows - 1}]"
        )

    columns = list(PARTICLE_COLUMNS)
    has_puppi_weights = PUPPI_WEIGHT_COLUMN in schema_names
    if has_puppi_weights:
        columns.append(PUPPI_WEIGHT_COLUMN)
    table = pq.read_table(path, columns=columns).slice(event_index, 1)

    eta = np.asarray(table[PARTICLE_COLUMNS[0]][0].as_py(), dtype=float)
    phi = np.asarray(table[PARTICLE_COLUMNS[1]][0].as_py(), dtype=float)
    pt = np.asarray(table[PARTICLE_COLUMNS[2]][0].as_py(), dtype=float)
    selection = np.isfinite(eta) & np.isfinite(phi) & np.isfinite(pt) & (pt > 0.0)
    if has_puppi_weights and puppi_weight_min is not None:
        puppi_weight = np.asarray(table[PUPPI_WEIGHT_COLUMN][0].as_py(), dtype=float)
        selection &= np.isfinite(puppi_weight) & (puppi_weight >= puppi_weight_min)
    selected_eta, selected_phi, selected_pt = eta[selection], phi[selection], pt[selection]
    if max_particles is not None:
        selected_eta = selected_eta[:max_particles]
        selected_phi = selected_phi[:max_particles]
        selected_pt = selected_pt[:max_particles]
    wrapped_phi = np.arctan2(np.sin(selected_phi), np.cos(selected_phi))
    return selected_eta, wrapped_phi, selected_pt


def _cluster_jets(
    eta: np.ndarray,
    phi: np.ndarray,
    pt: np.ndarray,
    radius: float,
    algorithm: str,
    min_jet_pt: float,
):
    """Cluster massless particles and return inclusive FastJet jets for one event."""
    if not len(pt):
        return []
    algorithm_map = {"kt": fastjet.kt_algorithm, "antikt": fastjet.antikt_algorithm}
    particles = ak.zip(
        {"pt": [pt], "eta": [eta], "phi": [phi], "mass": [np.zeros_like(pt)]},
        with_name="Momentum4D",
    )
    jet_definition = fastjet.JetDefinition(algorithm_map[algorithm], radius)
    cluster = fastjet.ClusterSequence(particles, jet_definition)
    jets = cluster.inclusive_jets(min_pt=min_jet_pt)[0]
    order = np.argsort(-np.asarray(jets.pt))
    return [(float(jets.eta[i]), float(jets.phi[i]), float(jets.pt[i])) for i in order]


def _add_wrapped_jet_circle(ax, eta: float, phi: float, radius: float, color: str) -> None:
    """Draw the jet boundary, including copies visible across the phi wrap."""
    for wrapped_phi in (phi - 2 * np.pi, phi, phi + 2 * np.pi):
        ax.add_patch(
            Circle((eta, wrapped_phi), radius, fill=False, color=color, lw=1.8, alpha=0.9)
        )


def plot_event(
    eta: np.ndarray,
    phi: np.ndarray,
    pt: np.ndarray,
    jets: list[tuple[float, float, float]],
    radius: float,
    algorithm: str,
    event_index: int,
):
    """Create the event-display figure."""
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    scatter = ax.scatter(
        eta,
        phi,
        s=_marker_areas(pt),
        c=np.log10(pt),
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.25,
        zorder=2,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$\log_{10}(p_{\mathrm{T}} / \mathrm{GeV})$")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    jet_handles = []
    for index, (jet_eta, jet_phi, jet_pt) in enumerate(jets, start=1):
        color = colors[(index - 1) % len(colors)]
        _add_wrapped_jet_circle(ax, jet_eta, jet_phi, radius, color)
        ax.scatter(jet_eta, jet_phi, marker="x", s=80, color=color, linewidths=2.0, zorder=4)
        ax.annotate(
            f"J{index}: {jet_pt:.1f} GeV",
            (jet_eta, jet_phi),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize="small",
            zorder=5,
        )
        jet_handles.append(Line2D([0], [0], color=color, marker="x", lw=1.8, label=f"jet {index}"))

    size_pts = np.array([1.0, 10.0, 100.0])
    size_handles = [
        plt.scatter([], [], s=_marker_areas(np.array([value]))[0], color="gray", alpha=0.8,
                    edgecolors="black", linewidths=0.25, label=f"{value:g} GeV")
        for value in size_pts
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title=r"particle $p_{\mathrm{T}}$",
        loc="upper right",
    )
    ax.add_artist(size_legend)
    if jet_handles:
        ax.legend(handles=jet_handles, title="clustered jets", loc="lower right")

    eta_padding = max(radius, 0.2)
    if len(eta):
        ax.set_xlim(float(np.min(eta) - eta_padding), float(np.max(eta) + eta_padding))
    else:
        ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(*PHI_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(
        f"ORBIT event {event_index}: {len(pt)} particles, {len(jets)} {algorithm} jets "
        f"($R={radius:g}$)"
    )
    ax.grid(alpha=0.25)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_file", type=Path, help="ORBIT particle Parquet file")
    parser.add_argument("event_index", type=int, help="Zero-based event index in the file")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG or PDF path")
    parser.add_argument(
        "--radius", type=float, default=0.8, help="FastJet radius R (default: 0.8)"
    )
    parser.add_argument("--algorithm", choices=("kt", "antikt"), default="kt")
    parser.add_argument(
        "--mode",
        choices=("standard", "full-event"),
        default="standard",
        help="Standard uses PuppiW >= 0.05 and 128 particles; full-event uses no weight cut and 500.",
    )
    parser.add_argument(
        "--min-particle-pt", type=float, default=0.0, help="Particle pT threshold [GeV]"
    )
    parser.add_argument("--min-jet-pt", type=float, default=0.0, help="Jet pT threshold [GeV]")
    parser.add_argument(
        "--puppi-weight-min",
        type=float,
        default=None,
        help="Override the standard-mode PUPPI threshold; ignored by full-event mode.",
    )
    parser.add_argument("--max-particles", type=int, help="Override the mode's particle cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.min_particle_pt < 0 or args.min_jet_pt < 0:
        raise ValueError("pT thresholds must be non-negative")
    if args.max_particles is not None and args.max_particles < 1:
        raise ValueError("--max-particles must be positive")
    puppi_weight_min = (
        None
        if args.mode == "full-event"
        else (0.05 if args.puppi_weight_min is None else args.puppi_weight_min)
    )
    max_particles = args.max_particles or (500 if args.mode == "full-event" else 128)
    eta, phi, pt = _read_event(
        args.parquet_file,
        args.event_index,
        puppi_weight_min,
        max_particles,
    )
    particle_selection = pt >= args.min_particle_pt
    eta, phi, pt = eta[particle_selection], phi[particle_selection], pt[particle_selection]
    jets = _cluster_jets(eta, phi, pt, args.radius, args.algorithm, args.min_jet_pt)
    figure = plot_event(eta, phi, pt, jets, args.radius, args.algorithm, args.event_index)
    output = args.output or Path(f"orbit_event_{args.event_index}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved event display to {output}")


if __name__ == "__main__":
    main()
