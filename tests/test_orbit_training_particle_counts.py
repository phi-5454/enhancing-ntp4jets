"""Tests for the bounded training-start particle multiplicity plot."""

from types import SimpleNamespace

import numpy as np
import torch

from gabbro.callbacks.orbit_plotting_callback import OrbitPlottingCallback
from gabbro.plotting.orbit import plot_particle_count_histograms


def test_training_particle_count_sample_is_bounded_and_class_aware():
    batches = [
        {
            "part_mask": torch.tensor(
                [[True, True, False], [True, True, True], [True, False, False]]
            ),
            "jet_type_labels": torch.tensor([0, 1, 0]),
        }
    ]
    datamodule = SimpleNamespace(
        hparams=SimpleNamespace(class_to_label={"ggHbb": 0, "minbias": 1}),
        train_dataloader=lambda: batches,
    )
    trainer = SimpleNamespace(datamodule=datamodule)

    counts = OrbitPlottingCallback._sample_training_particle_counts(trainer, max_events=2)

    assert counts["ggHbb"].tolist() == [2]
    assert counts["minbias"].tolist() == [3]
    assert sum(len(values) for values in counts.values()) == 2


def test_particle_count_figure_contains_joint_and_classwise_panels():
    figure = plot_particle_count_histograms(
        {"ggHbb": np.asarray([2, 3]), "minbias": np.asarray([1, 2])}
    )

    assert [axis.get_title() for axis in figure.axes] == [
        "Joint training sample",
        "Training sample by class",
    ]
