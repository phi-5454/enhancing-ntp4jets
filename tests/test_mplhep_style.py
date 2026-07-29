"""Tests for the shared mplhep-based physics plotting style."""

import sys
from types import SimpleNamespace

import matplotlib as mpl

from gabbro.plotting import utils


def test_set_mpl_style_uses_cms_and_preserves_project_palette(monkeypatch):
    calls = []
    fake_hep = SimpleNamespace(
        style=SimpleNamespace(CMS=object(), use=lambda style: calls.append(style))
    )
    monkeypatch.setitem(sys.modules, "mplhep", fake_hep)

    returned = utils.set_mpl_style()

    assert returned is fake_hep
    assert calls == [fake_hep.style.CMS]
    assert mpl.rcParams["axes.prop_cycle"] == utils.params_to_update["axes.prop_cycle"]
