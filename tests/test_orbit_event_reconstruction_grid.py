"""Tests for matched ORBIT validation event displays."""

import matplotlib.pyplot as plt
import numpy as np

from gabbro.plotting.orbit import (
    plot_event_reconstruction_grid,
    reconstruction_loss_metrics,
)


def test_plot_event_reconstruction_grid(tmp_path):
    original = np.array(
        [
            [[0.0, 0.1, 10.0], [0.2, -0.3, 2.0]],
            [[-0.4, 1.0, 20.0], [0.0, 0.0, 0.0]],
            [[0.7, -2.0, 5.0], [0.8, -1.9, 1.0]],
        ]
    )
    reconstructed = original.copy()
    reconstructed[..., 0] += 0.02
    mask = np.array([[True, True], [True, False], [True, True]])

    figure = plot_event_reconstruction_grid(
        [
            ("ggHbb", original, reconstructed, mask),
            ("minbias", original, reconstructed, mask),
        ],
        events_per_group=3,
    )
    output = tmp_path / "event_reconstruction_grid.png"
    figure.savefig(output)
    plt.close(figure)

    assert len(figure.axes) == 7  # Six event axes and one shared colorbar.
    assert len(figure.legends) == 1
    legend_handles = figure.legends[0].legend_handles
    assert [handle.get_marker() for handle in legend_handles] == ["o", "x"]
    assert legend_handles[1].get_color() == "black"
    assert legend_handles[1].get_markersize() == 9
    assert output.stat().st_size > 0


def test_reconstruction_loss_metrics_match_particle_and_value_reductions():
    original = np.zeros((2, 2, 2))
    reconstructed = np.array(
        [
            [[1.0, -1.0], [10.0, 10.0]],
            [[2.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = np.array([[True, False], [True, False]])

    metrics = reconstruction_loss_metrics(original, reconstructed, mask, "l1")

    assert metrics["loss_reco"] == 2.0
    assert metrics["loss_reco_l1"] == 2.0
    assert metrics["loss_reco_l2"] == 3.0
    assert metrics["loss_reco_l1_per_value"] == 1.0
    assert metrics["loss_reco_l2_per_value"] == 1.5
