#!/usr/bin/env python
"""Run the raw highest-pT Higgs-mass control in the local FastJet environment."""

from pathlib import Path
import runpy

import vector
from matplotlib.axes import Axes


vector.register_awkward()
_original_set = Axes.set


def _set_with_valid_higgs_title(self, *args, **kwargs):
    if "title" in kwargs:
        kwargs["title"] = r"Resolved $H \to b\bar{b}$"
    return _original_set(self, *args, **kwargs)


Axes.set = _set_with_valid_higgs_title
runpy.run_path(
    str(Path(__file__).with_name("evaluate_orbit_higgs_mass_raw.py")),
    run_name="__main__",
)
