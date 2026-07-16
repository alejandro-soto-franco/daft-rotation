"""End-to-end tests that a sliced/filtered column reads the correct rows.

These tests push a `FixedSizeListArray` through a Daft `.where()` filter,
which produces an array carrying a non-zero offset, and assert that the
kernel returns the correct value for every row. They do not pin the
"row * width" indexing defect once believed to lurk in `FixedRows::get`,
and Daft's own `examples/dvector/src/vectors.rs` does not carry that defect
either.

Finding, verified against `arrow-array-57.3.0/src/array/fixed_size_list_array.rs`:
`impl From<ArrayData> for FixedSizeListArray` (~line 442) folds the parent's
offset into the child at construction
(`data.child_data()[0].slice(data.offset() * size, data.len() * size)`),
`offset()` is hard-coded to return 0 (~line 502), and `value_offset_at(i)` is
plain `i * value_length` with no offset term. So `values()[i * width ..]` and
`value(i)` compute the same thing on this arrow-rs version: no test built
against the public API can distinguish manual `row * width` arithmetic from
the `value(i)` accessor here.

`FixedRows` still uses `value(i)` rather than manual arithmetic, because
`value(i)` is correct whether or not arrow-rs folds the offset eagerly,
whereas manual arithmetic on `values()` silently depends on that folding
behaviour, an implementation detail rather than part of arrow-rs's public
contract. These tests stay useful as end-to-end coverage of sliced/filtered
input, and as a guard against a future arrow-rs that stops folding offsets
eagerly.
"""

from __future__ import annotations

import daft
import pytest

from daft_rotation import quat, quat_inverse


def _fsl4():
    return daft.DataType.fixed_size_list(daft.DataType.float64(), 4)


def test_sliced_input_reads_the_right_rows(sess):
    """A non-zero array offset must not shift the values read.

    Filtering with `.where()` produces a sliced array with a non-zero offset
    downstream; this checks that every row still resolves to its correct
    quaternion end to end. See the module docstring for why this cannot
    distinguish `value(i)` from manual `row * width` arithmetic on the
    pinned arrow-rs version, and why `FixedRows` uses `value(i)` regardless.
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
    """A non-zero offset from `.where()` must preserve per-row sign and order.

    Companion to `test_sliced_input_reads_the_right_rows`: each row bakes in
    its own integer index, so both the sign and the ordering of the returned
    values pin the offset, not just its magnitude. As with that test, this
    cannot distinguish `value(i)` from manual `row * width` arithmetic on the
    pinned arrow-rs version; see the module docstring.
    """
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
