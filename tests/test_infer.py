from __future__ import annotations

import math
import random

import daft
import pytest

from daft_rotation import infer_quat_order


def _small_rotations(n, order):
    """Near-identity rotations, the common case in consecutive frames."""
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        angle = rng.uniform(-0.05, 0.05)
        axis = [rng.gauss(0, 1) for _ in range(3)]
        norm = math.sqrt(sum(a * a for a in axis))
        axis = [a / norm for a in axis]
        s = math.sin(angle / 2)
        w = math.cos(angle / 2)
        x, y, z = (a * s for a in axis)
        rows.append([x, y, z, w] if order == "xyzw" else [w, x, y, z])
    return daft.from_pydict({"q": rows})


def test_infers_xyzw_from_near_identity_rotations():
    df = _small_rotations(500, "xyzw")
    report = infer_quat_order(df, "q")
    assert report.likely == "xyzw"
    assert report.confidence == "high"


def test_infers_wxyz_from_near_identity_rotations():
    df = _small_rotations(500, "wxyz")
    report = infer_quat_order(df, "q")
    assert report.likely == "wxyz"
    assert report.confidence == "high"


def test_reports_low_confidence_for_uniform_rotations():
    """Large, uniformly distributed rotations make the slots look alike.

    The tool must say so rather than pick.
    """
    rng = random.Random(1)
    rows = []
    for _ in range(500):
        v = [rng.gauss(0, 1) for _ in range(4)]
        norm = math.sqrt(sum(c * c for c in v))
        rows.append([c / norm for c in v])
    df = daft.from_pydict({"q": rows})
    report = infer_quat_order(df, "q")
    assert report.confidence == "low"


def test_flags_a_column_that_is_not_unit_quaternions():
    df = daft.from_pydict({"q": [[1.0, 2.0, 3.0, 4.0]] * 10})
    report = infer_quat_order(df, "q")
    assert report.unit_norm is False
    assert "not unit" in str(report).lower()


def test_report_renders_both_slots():
    df = _small_rotations(100, "xyzw")
    text = str(infer_quat_order(df, "q"))
    assert "slot 0" in text and "slot 3" in text
