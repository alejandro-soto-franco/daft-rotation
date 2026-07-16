from __future__ import annotations

import math

import daft
import numpy as np
import pytest

from daft_rotation import quat, quat_rotate, quat_to_matrix, rot6d_to_matrix, rotation_geodesic_angle

# Row-major, flattened. Casting a *nested* list of lists directly to
# Tensor[Float64, (3, 3)] fails on this Daft build with
# "Cannot cast List to FixedSizeList because not all elements have sizes: 9"
# (see probe/GROUND-TRUTH.md); Daft's own tests/expressions/test_rotation.py
# flattens matrices the same way before casting to a tensor column.
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_geodesic_angle_of_identity_with_itself_is_zero(sess):
    df = daft.from_pydict({"a": [IDENTITY], "b": [IDENTITY]})
    df = df.select(
        df["a"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
        df["b"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
    )
    out = df.select(rotation_geodesic_angle(df["a"], df["b"])).to_pydict()
    assert out["a"][0] == pytest.approx(0.0, abs=1e-9)


def test_geodesic_angle_recovers_a_quarter_turn(sess):
    quarter = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    df = daft.from_pydict({"a": [quarter], "b": [IDENTITY]})
    df = df.select(
        df["a"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
        df["b"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
    )
    out = df.select(rotation_geodesic_angle(df["a"], df["b"])).to_pydict()
    assert out["a"][0] == pytest.approx(math.pi / 2, abs=1e-9)


def test_geodesic_angle_is_null_outside_so3(sess):
    junk = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = daft.from_pydict({"a": [junk], "b": [IDENTITY]})
    df = df.select(
        df["a"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
        df["b"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
    )
    out = df.select(rotation_geodesic_angle(df["a"], df["b"])).to_pydict()
    assert out["a"][0] is None


def test_geodesic_angle_broadcasts_a_literal_against_a_trajectory(sess):
    """A length-1 literal must broadcast against a longer column.

    Every other multi-argument function (quat_multiply, quat_rotate) broadcasts
    a length-1 argument against a longer one; rotation_geodesic_angle must too,
    since comparing a trajectory against one reference pose is its natural use.
    """
    quarter = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    mat3 = daft.DataType.tensor(daft.DataType.float64(), (3, 3))
    df = daft.from_pydict({"a": [IDENTITY, quarter]})
    df = df.select(df["a"].cast(mat3))
    identity_lit = daft.lit(IDENTITY).cast(mat3)

    out = df.select(rotation_geodesic_angle(df["a"], identity_lit)).to_pydict()

    # Each row's angle against the identity is that row's own rotation angle:
    # the identity row is 0, the quarter turn is pi/2.
    assert out["a"] == pytest.approx([0.0, math.pi / 2], abs=1e-9)


I_XYZW = [0.0, 0.0, 0.0, 1.0]
MAT3 = daft.DataType.tensor(daft.DataType.float64(), (3, 3))


def test_matrix_outputs_are_real_tensors(sess):
    """The output must be Tensor[Float64, [3,3]], not a flat list of 9.

    Only the arrow.fixed_shape_tensor extension tag makes it so; storage
    alone yields FixedSizeList[Float64, 9]. Without the tag this passes
    every numeric test while silently returning the wrong type, so assert
    the dtype directly.
    """
    df = daft.from_pydict({"r": [[2.0, 0.0, 0.0, 1.0, 3.0, 0.0]]})
    df = df.select(df["r"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 6)))
    assert df.select(rot6d_to_matrix(df["r"])).schema()["r"].dtype == MAT3


def test_rot6d_to_matrix_orthonormalises(sess):
    df = daft.from_pydict({"r": [[2.0, 0.0, 0.0, 1.0, 3.0, 0.0]]})
    df = df.select(df["r"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 6)))
    out = df.select(rot6d_to_matrix(df["r"])).to_pydict()
    m = out["r"][0].tolist()
    assert m[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_rot6d_to_matrix_rejects_parallel_vectors(sess):
    df = daft.from_pydict({"r": [[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]})
    df = df.select(df["r"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 6)))
    out = df.select(rot6d_to_matrix(df["r"])).to_pydict()
    assert out["r"][0] is None


def test_quat_to_matrix_of_identity_is_the_identity(sess):
    df = daft.from_pydict({"q": [I_XYZW]})
    df = df.select(df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)))
    df = df.select(quat(df["q"], "xyzw"))
    out = df.select(quat_to_matrix(df["q"])).to_pydict()
    # pytest.approx rejects a nested plain-list actual/expected pair outright
    # ("does not support nested data structures"); wrapping the expected value
    # in a numpy array routes the comparison through its numpy path instead,
    # which does handle 2D arrays, without changing what is being asserted.
    assert out["q"][0].tolist() == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), abs=1e-9
    )


def test_quat_rotate_preserves_norm(sess):
    quarter_z = [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
    df = daft.from_pydict({"q": [quarter_z], "v": [[1.0, 0.0, 0.0]]})
    df = df.select(
        df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)),
        df["v"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 3)),
    )
    df = df.select(quat(df["q"], "xyzw"), df["v"])
    out = df.select(quat_rotate(df["q"], df["v"])).to_pydict()
    # A quarter turn about z sends x to y.
    assert out["q"][0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_quat_rotate_needs_an_order_for_a_plain_column(sess):
    df = daft.from_pydict({"q": [I_XYZW], "v": [[1.0, 0.0, 0.0]]})
    df = df.select(
        df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)),
        df["v"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 3)),
    )
    with pytest.raises(Exception, match="infer_quat_order"):
        df.select(quat_rotate(df["q"], df["v"])).collect()
