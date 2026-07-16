from __future__ import annotations

import daft
import pytest

from daft_rotation import matrix_to_quat

# Matrices are FLAT, row-major, 9 elements. A nested [[...], [...], [...]] row
# does not cast: Daft raises "Cannot cast List to FixedSizeList because not all
# elements have sizes: 9". See probe/GROUND-TRUTH.md.
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

FSL4 = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
QUAT_XYZW = daft.DataType.extension("daft_rotation.quaternion", FSL4, "xyzw")
QUAT_WXYZ = daft.DataType.extension("daft_rotation.quaternion", FSL4, "wxyz")


def _matrices(rows):
    df = daft.from_pydict({"m": rows})
    return df.select(df["m"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))))


def test_matrix_to_quat_output_is_extension_typed(sess):
    """The output column must carry its convention in its dtype.

    This is the load-bearing claim of the whole design and nothing in Daft
    tests it, so pin it here.

    Compare dtypes with ==, never str(): DataType's Display renders
    Extension[name; storage] and omits the metadata entirely, so both
    conventions stringify identically. Equality does compare the metadata.
    """
    df = _matrices([IDENTITY])
    out = df.select(matrix_to_quat(df["m"], order="xyzw"))
    assert out.schema()["m"].dtype == QUAT_XYZW


def test_matrix_to_quat_order_travels_in_the_dtype(sess):
    df = _matrices([IDENTITY])
    xyzw = df.select(matrix_to_quat(df["m"], order="xyzw")).schema()["m"].dtype
    wxyz = df.select(matrix_to_quat(df["m"], order="wxyz")).schema()["m"].dtype
    assert xyzw == QUAT_XYZW
    assert wxyz == QUAT_WXYZ
    assert xyzw != wxyz, "the two conventions must be distinguishable"


def test_extension_storage_is_transparent_to_consumers(sess):
    """.to_pydict() must yield plain lists, so the tag costs consumers nothing."""
    df = _matrices([IDENTITY])
    out = df.select(matrix_to_quat(df["m"], order="xyzw")).to_pydict()
    row = out["m"][0]
    assert isinstance(row, list), f"expected a plain list, got {type(row)}"
    assert len(row) == 4


def test_identity_matrix_gives_the_identity_quaternion(sess):
    df = _matrices([IDENTITY])
    out = df.select(matrix_to_quat(df["m"], order="xyzw")).to_pydict()
    x, y, z, w = out["m"][0]
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert abs(w) == pytest.approx(1.0, abs=1e-9)


def test_wxyz_puts_the_scalar_first(sess):
    df = _matrices([IDENTITY])
    out = df.select(matrix_to_quat(df["m"], order="wxyz")).to_pydict()
    w, x, y, z = out["m"][0]
    assert abs(w) == pytest.approx(1.0, abs=1e-9)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_non_rotation_gives_null(sess):
    junk = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = _matrices([junk])
    out = df.select(matrix_to_quat(df["m"], order="xyzw")).to_pydict()
    assert out["m"][0] is None


def test_invalid_order_fails_in_python(sess):
    """An unrecognised order must never reach Rust."""
    df = _matrices([IDENTITY])
    with pytest.raises(ValueError, match="wxzy"):
        matrix_to_quat(df["m"], order="wxzy")
