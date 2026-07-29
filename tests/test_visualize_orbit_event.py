"""Focused tests for the ORBIT event-display helpers."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "visualize_orbit_event.py"
SPEC = importlib.util.spec_from_file_location("visualize_orbit_event", SCRIPT_PATH)
event_display = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(event_display)


def test_read_cluster_and_plot_event(tmp_path):
    parquet_path = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_Eta": [[0.0, 0.1, 2.0]],
                "L1T_PUPPIPart_Phi": [[3.12, -3.12, 0.0]],
                "L1T_PUPPIPart_PT": [[100.0, 20.0, 5.0]],
                "L1T_PUPPIPart_PuppiW": [[1.0, 1.0, 0.01]],
            }
        ),
        parquet_path,
    )

    eta, phi, pt = event_display._read_event(parquet_path, 0, puppi_weight_min=0.05)
    assert np.allclose(eta, [0.0, 0.1])
    assert len(phi) == len(pt) == 2
    assert np.all(event_display._marker_areas(pt) > 0)

    full_eta, _, full_pt = event_display._read_event(
        parquet_path,
        0,
        puppi_weight_min=None,
        max_particles=3,
    )
    assert np.allclose(full_eta, [0.0, 0.1, 2.0])
    assert np.allclose(full_pt, [100.0, 20.0, 5.0])

    jets = event_display._cluster_jets(eta, phi, pt, 0.8, "kt", 0.0)
    figure = event_display.plot_event(eta, phi, pt, jets, 0.8, "kt", event_index=0)
    output = tmp_path / "event.png"
    figure.savefig(output)
    plt.close(figure)
    assert output.stat().st_size > 0
