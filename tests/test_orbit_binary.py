import numpy as np
import pytest

from gabbro.utils.orbit_binary import (
    bits_for_size,
    pack_unsigned_values,
    read_packed_ragged_binary,
    read_ragged_binary,
    unpack_unsigned_values,
    unsigned_dtype_for_size,
    write_packed_ragged_binary,
    write_ragged_binary,
)
from gabbro.utils.orbit_firmware import (
    ANGLE_LSB,
    PT_LSB_GEV,
    PUPPI_COMMON_BITS,
    quantize_puppi_common,
    unpack_puppi_common,
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


@pytest.mark.parametrize(("size", "bits"), [(1, 1), (2, 1), (256, 8), (4096, 12)])
def test_bits_for_size(size, bits):
    assert bits_for_size(size) == bits


def test_dense_unsigned_packing_round_trip():
    values = np.array([0, 1, 4095, 17, 2048], dtype=np.uint16)
    payload = pack_unsigned_values(values, 12)
    assert len(payload) == 8
    np.testing.assert_array_equal(unpack_unsigned_values(payload, len(values), 12), values)


def test_packed_ragged_round_trip(tmp_path):
    events = [np.array([0, 4095, 7]), np.array([], dtype=np.uint16), np.array([12])]
    path = tmp_path / "packed.bin"
    write_packed_ragged_binary(
        path,
        events,
        bits_per_value=12,
        metadata={"kind": "vq_tokens", "num_codes": 4096},
    )
    header, loaded = read_packed_ragged_binary(path)
    assert header["bits_per_value"] == 12
    assert header["num_codes"] == 4096
    for actual, expected in zip(loaded, events):
        np.testing.assert_array_equal(actual, expected)


def test_puppi_common_golden_layout_and_round_trip():
    features = np.array(
        [[10 * PT_LSB_GEV, -2 * ANGLE_LSB, 3 * ANGLE_LSB]], dtype=np.float64
    )
    words = quantize_puppi_common(features, np.array([6]))
    expected = 10 | (((1 << 12) - 2) << 14) | (3 << 26) | (6 << 37)
    assert int(words[0]) == expected
    assert int(words[0]).bit_length() <= PUPPI_COMMON_BITS
    fields = unpack_puppi_common(words)
    assert {name: int(values[0]) for name, values in fields.items()} == {
        "pt": 10,
        "eta": -2,
        "phi": 3,
        "pid": 6,
    }


def test_puppi_common_rounds_saturates_and_wraps_phi():
    features = np.array([[1e9, -1e9, np.pi + ANGLE_LSB]], dtype=np.float64)
    fields = unpack_puppi_common(quantize_puppi_common(features, np.array([7])))
    assert int(fields["pt"][0]) == (1 << 14) - 1
    assert int(fields["eta"][0]) == -(1 << 11)
    assert int(fields["phi"][0]) == -719
    assert int(fields["pid"][0]) == 7
