import numpy as np
import pytest

from gabbro.utils.orbit_binary import (
    read_ragged_binary,
    unsigned_dtype_for_size,
    write_ragged_binary,
)


@pytest.mark.parametrize(
    ("size", "dtype"),
    [(1, "|u1"), (256, "|u1"), (257, "<u2"), (65536, "<u2"), (65537, "<u4")],
)
def test_unsigned_dtype_for_codebook_size(size, dtype):
    assert unsigned_dtype_for_size(size).str == dtype


def test_ragged_feature_round_trip(tmp_path):
    events = [
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        np.array([[9, 10, 11, 12]], dtype=np.float32),
    ]
    path = tmp_path / "features.bin"
    write_ragged_binary(
        path,
        events,
        dtype="<f4",
        width=4,
        metadata={"kind": "original_parquet_particles"},
    )

    header, loaded = read_ragged_binary(path)

    assert header["event_count"] == 2
    assert header["kind"] == "original_parquet_particles"
    assert header["dtype"] == "<f4"
    for actual, expected in zip(loaded, events):
        np.testing.assert_array_equal(actual, expected)


def test_ragged_token_round_trip(tmp_path):
    events = [np.array([0, 2047], dtype=np.uint16), np.array([7], dtype=np.uint16)]
    path = tmp_path / "tokens.bin"
    write_ragged_binary(
        path,
        events,
        dtype=unsigned_dtype_for_size(2048),
        metadata={"kind": "quantized_token_ids", "num_codes": 2048},
    )

    header, loaded = read_ragged_binary(path)

    assert header["dtype"] == "<u2"
    assert header["num_codes"] == 2048
    for actual, expected in zip(loaded, events):
        np.testing.assert_array_equal(actual, expected)


def test_ragged_writer_rejects_wrong_width(tmp_path):
    with pytest.raises(ValueError, match="event shape"):
        write_ragged_binary(
            tmp_path / "bad.bin",
            [np.zeros((3, 3), dtype=np.float32)],
            dtype="<f4",
            width=4,
            metadata={},
        )
