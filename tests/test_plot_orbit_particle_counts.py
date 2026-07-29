"""Tests for the standalone ORBIT particle-count plotting helpers."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_orbit_particle_counts.py"
SPEC = importlib.util.spec_from_file_location("plot_orbit_particle_counts", SCRIPT_PATH)
count_plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(count_plots)


def test_counts_and_histogram(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_PT": [[10.0, 3.0, 0.0], [5.0]],
                "L1T_PUPPIPart_PuppiW": [[0.9, 0.05, 1.0], [0.2]],
            }
        ),
        parquet_path,
    )
    total, valid, has_weights = count_plots.particle_counts([parquet_path], 10, 0.05)
    assert total.tolist() == [3, 1]
    assert valid.tolist() == [1, 1]
    assert has_weights

    output = tmp_path / "counts.png"
    count_plots.plot_histograms(
        {"ggHbb": total, "minbias": valid},
        {"ggHbb": valid, "minbias": total},
        output,
        bins=10,
        puppi_weight_min=0.05,
    )
    assert output.stat().st_size > 0


def test_uses_dataloader_manifest_names(tmp_path):
    for name in ("ggHbb_train_val.txt", "minbias_train_val.txt"):
        (tmp_path / name).write_text("/tmp/example.parquet\n")
    manifests = count_plots.manifests_for_split(tmp_path, "train-val")
    assert manifests["ggHbb"].name == "ggHbb_train_val.txt"
    assert manifests["minbias"].name == "minbias_train_val.txt"
