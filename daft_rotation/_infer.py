"""Discovering a quaternion column's component order from its values.

This is a diagnostic, not a constructor. It reports what the data suggests and
returns to the caller, who then declares the convention with ``quat``. Nothing
here feeds a default: a heuristic that silently picked a convention would be the
very failure the typed dtype exists to prevent.

The discriminator is a two-way comparison. Slots 1 and 2 hold vector components
under both conventions (y,z for xyzw; x,y for wxyz), so only slots 0 and 3 differ
and exactly one of them is the scalar. In near-identity rotations the scalar
concentrates near |1| while the vector parts sit near 0, which separates them.
Large, uniformly distributed rotations do not separate, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import daft


@dataclass
class OrderReport:
    """What a sample of a quaternion column suggests about its component order."""

    likely: str | None
    confidence: str
    slot0_mean: float
    slot3_mean: float
    slot0_var: float
    slot3_var: float
    unit_norm: bool
    sampled: int

    def __str__(self) -> str:
        lines = [
            f"sampled {self.sampled} rows",
            f"  slot 0: mean|v|={self.slot0_mean:.4f} var={self.slot0_var:.4f}",
            f"  slot 3: mean|v|={self.slot3_mean:.4f} var={self.slot3_var:.4f}",
        ]
        if not self.unit_norm:
            lines.append(
                "  sum of squares is not 1: this column is not unit quaternions, "
                "so component order is not the first problem to solve"
            )
        if self.likely is None:
            lines.append(
                "  likely: undetermined (confidence: low)\n"
                "  neither slot concentrates near |1|, which happens when rotations "
                "are large and uniformly distributed. Read the dataset's "
                "documentation instead."
            )
        else:
            scalar_slot = 3 if self.likely == "xyzw" else 0
            lines.append(f"  likely: {self.likely} (confidence: {self.confidence})")
            lines.append(
                f"  basis: slot {scalar_slot} concentrates near |1|, consistent with "
                "a scalar component over near-identity frames"
            )
            lines.append(f'  declare it with: quat(col, "{self.likely}")')
        return "\n".join(lines)


def infer_quat_order(df: daft.DataFrame, column: str, n: int = 1000) -> OrderReport:
    """Reports which component order a quaternion column likely uses.

    Samples the column and compares slots 0 and 3, the only two that differ
    between the conventions. Returns a report; it never changes the column and
    never picks a convention on your behalf. Declare the result with ``quat``.

    Args:
        df (DataFrame): The frame holding the column.
        column (str): Name of a ``FixedSizeList[Float64, 4]`` column.
        n (int, default=1000): How many rows to sample.

    Returns:
        OrderReport: The statistics, a likely order, and a confidence level.
            ``likely`` is None when the sample does not separate the slots.
    """
    import numpy as np

    rows = df.select(daft.col(column)).limit(n).to_pydict()[column]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise ValueError(f"column {column!r} has no non-null rows to sample")

    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(
            f"column {column!r} must hold 4 components per row, got shape {a.shape}"
        )

    abs_mean = np.abs(a).mean(axis=0)
    var = a.var(axis=0)
    sq = (a**2).sum(axis=1)
    unit_norm = bool(np.allclose(sq, 1.0, atol=1e-3))

    s0, s3 = float(abs_mean[0]), float(abs_mean[3])
    # The scalar slot must both dominate the other candidate and sit near 1.
    # Requiring only the first would "identify" a scalar in uniform data.
    hi, lo = max(s0, s3), min(s0, s3)
    separated = hi > 0.9 and lo < 0.3

    if not separated:
        likely, confidence = None, "low"
    else:
        likely = "xyzw" if s3 > s0 else "wxyz"
        confidence = "high" if hi > 0.95 and lo < 0.15 else "medium"

    return OrderReport(
        likely=likely,
        confidence=confidence,
        slot0_mean=s0,
        slot3_mean=s3,
        slot0_var=float(var[0]),
        slot3_var=float(var[3]),
        unit_norm=unit_norm,
        sampled=len(rows),
    )
