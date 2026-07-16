"""Quaternion, 6D, and SO(3) rotation expressions for Daft.

Rotations are the most common non-scalar quantity in robot trajectory, camera
extrinsic, and hand-tracking datasets, and Daft stores them without semantics.
These expressions operate on them natively, so composing, inverting, and
comparing rotations does not require a Python UDF.

Quaternion component order is never assumed. Declare it once with ``quat``, or
pass ``order=`` per call, or discover it with ``infer_quat_order``.
"""

from __future__ import annotations

from typing import Literal

import daft
from daft.expressions import Expression

__all__ = ["matrix_to_quat", "rotation_geodesic_angle"]

QuatOrder = Literal["xyzw", "wxyz"]

_ORDERS = ("xyzw", "wxyz")


def _check_order(order: str) -> str:
    """Validate an order string before it can reach a function name.

    Raising here means an unrecognised convention never becomes a failed
    function lookup, whose error would name a function the user never typed.
    """
    if order not in _ORDERS:
        raise ValueError(
            f"component order {order!r} is not recognised; expected one of {_ORDERS}"
        )
    return order


def matrix_to_quat(matrix: Expression, *, order: QuatOrder) -> Expression:
    """Converts a 3x3 rotation matrix into a quaternion.

    Uses Shepperd's method, which stays numerically stable near a half turn where
    the naive trace formula loses precision. The sign is not canonicalised, since a
    quaternion and its negation denote the same rotation.

    The matrix is tested for membership in ``SO(3)`` first, so one that is scaled,
    is a shear, is a reflection, or holds a non-finite value produces null rather
    than a plausible quaternion.

    The result carries ``order`` in its dtype, so downstream calls need not repeat it.

    Args:
        matrix (Tensor[Float64, [3, 3]] Expression): The rotation matrix.
        order (str): Component order of the result, either "xyzw" or "wxyz".
            Required: there is no default convention.

    Returns:
        Expression (Quaternion Expression): The quaternion, extension-typed.
    """
    return daft.get_function(f"rotation_matrix_to_quat_{_check_order(order)}", matrix)


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
