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


from daft_rotation import quat, quat_inverse, quat_multiply

I_XYZW = [0.0, 0.0, 0.0, 1.0]


def _quats(rows, order=None):
    df = daft.from_pydict({"q": rows})
    df = df.select(df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)))
    if order is not None:
        df = df.select(quat(df["q"], order))
    return df


def test_typed_column_needs_no_order_argument(sess):
    df = _quats([I_XYZW], order="xyzw")
    out = df.select(quat_inverse(df["q"])).to_pydict()
    assert out["q"][0] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9)


def test_plain_column_without_order_is_rejected_at_plan_time(sess):
    df = _quats([I_XYZW])
    with pytest.raises(Exception, match="infer_quat_order"):
        df.select(quat_inverse(df["q"])).collect()


def test_plain_column_with_order_works(sess):
    df = _quats([I_XYZW])
    out = df.select(quat_inverse(df["q"], order="xyzw")).to_pydict()
    assert out["q"][0] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9)


def test_order_disagreeing_with_the_dtype_is_rejected(sess):
    df = _quats([I_XYZW], order="xyzw")
    with pytest.raises(Exception, match="wxyz"):
        df.select(quat_inverse(df["q"], order="wxyz")).collect()


def test_mixed_conventions_compose_correctly(sess):
    """An Eigen wxyz column and a ROS xyzw column must compose.

    Both normalise to canonical (w,x,y,z) on the way in, so this is a real
    capability rather than an accident.
    """
    half_z_xyzw = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    half_z_wxyz = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
    df = daft.from_pydict({"a": [half_z_xyzw], "b": [half_z_wxyz]})
    fsl = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
    df = df.select(df["a"].cast(fsl), df["b"].cast(fsl))
    df = df.select(quat(df["a"], "xyzw"), quat(df["b"], "wxyz"))
    out = df.select(quat_multiply(df["a"], df["b"])).to_pydict()
    # Two quarter turns about z compose to a half turn about z.
    x, y, z, w = out["a"][0]
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert abs(z) == pytest.approx(1.0, abs=1e-9)
    assert w == pytest.approx(0.0, abs=1e-9)


def test_output_takes_the_first_typed_arguments_order(sess):
    half_z_xyzw = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    half_z_wxyz = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
    df = daft.from_pydict({"a": [half_z_xyzw], "b": [half_z_wxyz]})
    fsl = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
    df = df.select(df["a"].cast(fsl), df["b"].cast(fsl))
    df = df.select(quat(df["a"], "xyzw"), quat(df["b"], "wxyz"))
    # Compare with ==, not str(): Display omits the metadata.
    dtype = df.select(quat_multiply(df["a"], df["b"])).schema()["a"].dtype
    assert dtype == QUAT_XYZW, f"expected the left argument's order, got {dtype}"


def test_multiply_broadcasts_a_single_row(sess):
    df = daft.from_pydict({"a": [I_XYZW, I_XYZW, I_XYZW]})
    fsl = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
    df = df.select(df["a"].cast(fsl))
    df = df.select(quat(df["a"], "xyzw"))
    lit = daft.lit(I_XYZW).cast(fsl)
    out = df.select(quat_multiply(df["a"], quat(lit, "xyzw"))).to_pydict()
    assert len(out["a"]) == 3


def test_null_row_propagates(sess):
    df = daft.from_pydict({"q": [I_XYZW, None]})
    fsl = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
    df = df.select(df["q"].cast(fsl))
    df = df.select(quat(df["q"], "xyzw"))
    out = df.select(quat_inverse(df["q"])).to_pydict()
    assert out["q"][1] is None


def test_value_row_after_a_null_row_is_intact(sess):
    """A null row must not shift the rows after it.

    FixedSizeListBuilder::finish asserts values.len() == len * list_len, so a
    wrong child count on the null path panics rather than corrupting silently.
    Every other fixture here is a single row, so nothing exercises a value row
    following a null one.
    """
    junk = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = _matrices([junk, IDENTITY])
    out = df.select(matrix_to_quat(df["m"], order="xyzw")).to_pydict()
    assert out["m"][0] is None
    x, y, z, w = out["m"][1]
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert abs(w) == pytest.approx(1.0, abs=1e-9)


def test_broadcast_preserves_values_not_just_length(sess):
    """Broadcasting must deliver the right values, not merely the right count.

    Multiplying identity by identity cannot distinguish a broadcast bug that
    returns wrong-but-4-wide data, so use a quarter turn about z whose product
    with each row is checkable.
    """
    quarter_z = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    fsl = daft.DataType.fixed_size_list(daft.DataType.float64(), 4)
    df = daft.from_pydict({"a": [quarter_z, quarter_z]})
    df = df.select(df["a"].cast(fsl))
    df = df.select(quat(df["a"], "xyzw"))
    lit = quat(daft.lit(quarter_z).cast(fsl), "xyzw")
    out = df.select(quat_multiply(df["a"], lit)).to_pydict()
    assert len(out["a"]) == 2
    # Two quarter turns about z compose to a half turn about z, on every row.
    for row in out["a"]:
        x, y, z, w = row
        assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert abs(z) == pytest.approx(1.0, abs=1e-9)
        assert w == pytest.approx(0.0, abs=1e-9)


# test_non_broadcastable_lengths_are_rejected is NOT drivable through the
# public API. daft.DataFrame["col"] resolves purely by column name, not by
# the DataFrame instance the reference was taken from: quat(a["q"], ...) and
# quat(b["q"], ...) both print as col(q), so evaluating them inside
# a.select(...) makes both operands resolve against a's own "q" column,
# silently ignoring b's data rather than raising or truncating (verified by
# hand: the result equals identity*identity, never touching b's rows).
# Daft's DataFrame model also guarantees every column in one .select() call
# shares that DataFrame's row count, so there is no way to hand a single
# kernel invocation two same-named columns of genuinely different length
# (neither equal nor 1) through the public API. The requirement itself
# (broadcast_len's mismatch arm must error, not truncate) is pinned directly
# in Rust instead: see ffi::tests::mismatched_lengths_are_rejected.


def test_functions_are_unavailable_without_the_extension():
    """A session that never loaded the extension must not resolve our functions."""
    from daft.session import Session

    s = Session()
    with s:
        df = daft.from_pydict({"q": [[0.0, 0.0, 0.0, 1.0]]})
        df = df.select(df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)))
        with pytest.raises(Exception, match="not found"):
            df.select(quat_inverse(df["q"], order="xyzw")).collect()
