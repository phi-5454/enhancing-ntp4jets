"""Focused tests for the restartable joint 5M campaign plot workflow."""

import json

import numpy as np
import pytest
import yaml

pytest.importorskip("mplhep")

from scripts.build_orbit_5m_campaign_comparison import load_registry, plot_jets
from scripts.evaluate_orbit_5m_plot_cache import load_exported_config, split_quota


def test_split_quota_preserves_balanced_event_budget():
    assert split_quota(10_000, 3) == [3334, 3333, 3333]
    assert sum(split_quota(2000, 9)) == 2000


def test_wandb_config_unwraps_nested_hydra_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"value": "{'sequence_type': 'particle_full', 'max_sequence_length': 500}"},
                "seed": {"value": 12345},
            }
        )
    )
    config = load_exported_config(path)
    assert config.data.sequence_type == "particle_full"
    assert config.data.max_sequence_length == 500
    assert config.seed == 12345


def test_registry_assigns_eight_unique_styles(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [{"id": str(i)} for i in range(8)]}))
    models = load_registry(path)
    assert len({model["color"] for model in models}) == 8
    assert len({model["marker"] for model in models}) == 8


def test_jet_plot_accepts_unconstrained_hungarian_cache(tmp_path):
    models = []
    suite_dir = tmp_path / "ttbar"
    suite_dir.mkdir()
    for index in range(8):
        model = {
            "id": f"m{index}",
            "label": f"model {index}",
            "color": f"C{index}",
            "linestyle": "-",
        }
        models.append(model)
        path = suite_dir / f"m{index}.npz"
        np.savez_compressed(
            path,
            jet_delta_pt=np.linspace(-0.2, 0.2, 20),
            jet_delta_eta=np.linspace(-0.1, 0.1, 20),
            jet_delta_phi=np.linspace(-0.1, 0.1, 20),
            jet_match_dr=np.linspace(0.0, 2.0, 20),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "jet_pairing": (
                        "Hungarian minimum total delta-R, no delta-R rejection"
                    )
                }
            )
        )
    figure = plot_jets(tmp_path, "ttbar", models)
    assert figure._suptitle is None
    assert len(figure.axes) == 3
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
