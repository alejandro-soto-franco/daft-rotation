"""Quaternion, 6D, and SO(3) rotation expressions for Daft.

Rotations are the most common non-scalar quantity in robot trajectory, camera
extrinsic, and hand-tracking datasets, and Daft stores them without semantics.
These expressions operate on them natively, so composing, inverting, and
comparing rotations does not require a Python UDF.

Quaternion component order is never assumed. Declare it once with ``quat``, or
pass ``order=`` per call, or discover it with ``infer_quat_order``.
"""

from __future__ import annotations

import daft
from daft.expressions import Expression

__all__ = ["rotation_geodesic_angle"]


def rotation_geodesic_angle(left: Expression, right: Expression) -> Expression:
    """Angle in radians of the relative rotation between two 3x3 rotation matrices.

    Computes ``arccos((trace(A @ B.T) - 1) / 2)``, the geodesic distance on SO(3),
    which lies in ``[0, pi]``. This is the bi-invariant Riemannian metric, so it is
    symmetric and unchanged by rotating both arguments.

    Both arguments are tested for membership in ``SO(3)``, since the angle is only
    defined between rotations: a row where either is not a rotation produces null
    rather than a plausible angle. Rounding is absorbed by clamping the cosine
    into ``[-1, 1]``.

    Applied to consecutive frames of a trajectory, it measures how much a body
    turned between them.

    Args:
        left (Tensor[Float64, [3, 3]] Expression): The first rotation matrix.
        right (Tensor[Float64, [3, 3]] Expression): The second rotation matrix.

    Returns:
        Expression (Float64 Expression): The angle in radians.
    """
    return daft.get_function("rotation_geodesic_angle", left, right)
