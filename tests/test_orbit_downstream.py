"""Focused tests for paired downstream physics-fidelity helpers."""

import json
import runpy

import awkward as ak
import numpy as np
import pytest

from gabbro.data.orbit_downstream import (
    FIVE_CLASS_GROUPS,
    FIVE_CLASS_TO_LABEL,
    PairedOrbitDataModule,
    physical_particles_to_classifier_features,
)
from gabbro.models.orbit_event_classifier import classification_metrics


def test_five_class_taxonomy_and_feature_contract():
    assert list(FIVE_CLASS_GROUPS) == ["QCD", "tt", "VJets", "VV", "ggHbb"]
    assert [len(processes) for processes in FIVE_CLASS_GROUPS.values()] == [2, 3, 7, 9, 1]
    assert FIVE_CLASS_TO_LABEL["ggHbb"] == 4

    particles = ak.Array([[[0.3, np.pi / 2, 10.0], [-0.6, 0.0, 20.0]]])
    pid = ak.Array([[1, 7]])
    features, mask = physical_particles_to_classifier_features(particles, pid, 4)
    assert features.shape == (1, 4, 12)
    assert mask.tolist() == [[True, True, False, False]]
    assert np.allclose(features[0, 0, :4], [0.1, 0.0, 1.0, np.log(10.0) - 1.8])
    assert features[0, 0, 5] == 1
    assert features[0, 1, 11] == 1
    assert np.count_nonzero(features[0, 2:].numpy()) == 0


def test_classification_metrics_are_perfect_for_perfect_logits():
    targets = np.repeat(np.arange(5), 3)
    logits = np.full((len(targets), 5), -10.0)
    logits[np.arange(len(targets)), targets] = 10.0
    metrics = classification_metrics(targets, logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_auroc"] == 1.0
    assert np.array_equal(metrics["confusion_matrix"], np.eye(5, dtype=int) * 3)


def test_paired_data_module_derives_full_event_length_and_batch_size(tmp_path):
    metadata = {
        "class_to_label": FIVE_CLASS_TO_LABEL,
        "sequence_type": "particle_full",
        "max_sequence_length": 500,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    module = PairedOrbitDataModule(str(tmp_path), num_workers=0)
    assert module.hparams.max_sequence_length == 500
    assert module.hparams.batch_size == 16

    with pytest.raises(ValueError, match="does not match paired data"):
        PairedOrbitDataModule(str(tmp_path), max_sequence_length=128, num_workers=0)


def test_higgs_truth_selector_uses_direct_bb_daughters():
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    truth = ak.Record(
        {
            "Gen_Part_PT": [100.0, 110.0, 50.0, 45.0],
            "Gen_Part_Eta": [0.0, 0.1, 0.2, -0.2],
            "Gen_Part_Phi": [0.0, 0.1, 0.2, -0.2],
            "Gen_Part_Mass": [125.0, 125.0, 4.8, 4.8],
            "Gen_Part_PID": [25, 25, 5, -5],
            "Gen_Part_D1": [-1, 2, -1, -1],
            "Gen_Part_D2": [-1, 3, -1, -1],
        }
    )
    higgs, b, bbar = namespace["decaying_higgs"](truth)
    assert higgs[2] == 110.0
    assert {b[2], bbar[2]} == {45.0, 50.0}


def test_higgs_candidates_are_stable_for_perfect_reconstruction():
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    particles = np.array(
        [
            [0.10, 0.10, 180.0],
            [0.12, 0.12, 80.0],
            [-0.35, -0.30, 170.0],
            [-0.38, -0.28, 70.0],
        ]
    )
    b = np.array([0.1, 0.1, 200.0, 4.8])
    bbar = np.array([-0.35, -0.3, 190.0, 4.8])
    higgs = np.array([-0.1, -0.1, 390.0, 125.0])
    resolved = namespace["resolved_candidate"](particles, b, bbar)
    boosted = namespace["boosted_candidate"](particles, higgs, b, bbar)
    assert resolved is not None
    assert boosted is not None
    assert resolved == namespace["resolved_candidate"](particles.copy(), b, bbar)
    assert boosted == namespace["boosted_candidate"](particles.copy(), higgs, b, bbar)
