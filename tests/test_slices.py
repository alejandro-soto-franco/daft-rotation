from __future__ import annotations

import daft
import pytest

from daft_rotation import quat, quat_inverse


def _fsl4():
    return daft.DataType.fixed_size_list(daft.DataType.float64(), 4)


def test_sliced_input_reads_the_right_rows(sess):
    """A non-zero array offset must not shift the values read.

    The kernel must reach rows through arrow accessors that compose the
    array's own offset. Indexing a flat child buffer with row*width silently
    reads a window shifted by offset*width, which returns wrong rotations
    rather than an error. This is the bug dvector's float path still has.
    """
    quats = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
    ]
    df = daft.from_pydict({"q": quats, "i": list(range(len(quats)))})
    df = df.select(df["q"].cast(_fsl4()), df["i"])
    df = df.select(quat(df["q"], "xyzw"), df["i"])

    # Filtering produces a sliced array with a non-zero offset downstream.
    tail = df.where(df["i"] >= 3)
    out = tail.select(quat_inverse(tail["q"])).to_pydict()

    # The inverse of a unit quaternion is its conjugate: negate x, y, z.
    expected = [[-x, -y, -z, w] for x, y, z, w in quats[3:]]
    assert len(out["q"]) == 3
    for got, want in zip(out["q"], expected):
        assert got == pytest.approx(want, abs=1e-9)


def test_limit_offset_slice_reads_the_right_rows(sess):
    quats = [[float(i), 0.0, 0.0, 1.0] for i in range(10)]
    df = daft.from_pydict({"q": quats, "i": list(range(10))})
    df = df.select(df["q"].cast(_fsl4()), df["i"])
    df = df.select(quat(df["q"], "xyzw"), df["i"])

    tail = df.where(df["i"] >= 7)
    out = tail.select(quat_inverse(tail["q"])).to_pydict()
    # Each row's x component is its index; the inverse negates it (after
    # dividing by the squared norm), so the sign and ordering both pin the offset.
    assert len(out["q"]) == 3
    for got, i in zip(out["q"], [7, 8, 9]):
        n2 = float(i) ** 2 + 1.0
        assert got == pytest.approx([-i / n2, 0.0, 0.0, 1.0 / n2], abs=1e-9)
