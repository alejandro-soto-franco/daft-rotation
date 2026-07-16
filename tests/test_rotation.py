from __future__ import annotations

import math

import daft
import pytest

from daft_rotation import rotation_geodesic_angle

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
