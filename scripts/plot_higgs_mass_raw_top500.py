#!/usr/bin/env python
"""Run the raw highest-pT Higgs-mass control with vector behavior registered."""

from pathlib import Path
import runpy

import vector  # noqa: F401  # Registers Momentum4D behavior used by FastJet.


runpy.run_path(
    str(Path(__file__).with_name("evaluate_orbit_higgs_mass_raw.py")),
    run_name="__main__",
)
