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

__all__ = [
    "QuatOrder",
    "matrix_to_quat",
    "quat",
    "quat_inverse",
    "quat_multiply",
    "quat_rotate",
    "quat_to_matrix",
    "rot6d_to_matrix",
    "rotation_geodesic_angle",
]

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


def quat(expr: Expression, order: QuatOrder) -> Expression:
    """Declares a quaternion column's component order.

    Attaches the convention to the column's dtype, so downstream calls need not
    repeat it and a mismatched pair is rejected rather than silently composed.

    This is a cast, not a kernel: it reinterprets the column's type and does not
    touch the values. The storage remains ``FixedSizeList[Float64, 4]``, so
    ``to_pydict`` still yields plain lists.

    Args:
        expr (FixedSizeList[Float64, 4] Expression): The quaternion column.
        order (str): Component order, either "xyzw" (ROS, tf2, scipy) or "wxyz"
            (Eigen, MuJoCo, Isaac Sim). Required: there is no default convention.
            If you do not know it, see ``infer_quat_order``.

    Returns:
        Expression (Quaternion Expression): The same values, extension-typed.
    """
    return expr.cast(
        daft.DataType.extension(
            "daft_rotation.quaternion",
            daft.DataType.fixed_size_list(daft.DataType.float64(), 4),
            _check_order(order),
        )
    )


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


def _fn(base: str, order: QuatOrder | None) -> str:
    """Pick the registered name for a call.

    ``order=None`` means "read it from the column's dtype", never "assume a
    default", so it maps to the generic registration whose return_field
    rejects an untyped argument.
    """
    return base if order is None else f"{base}_{_check_order(order)}"


def quat_multiply(left: Expression, right: Expression, *, order: QuatOrder | None = None) -> Expression:
    """Hamilton product of two quaternions.

    The result applies ``right`` first and then ``left``, matching composition of
    the corresponding rotation matrices. Neither input is normalised. Rows holding
    a non-finite value produce null.

    Arguments may use different conventions: each is read according to its own
    declared order. The result carries the left argument's order.

    Args:
        left (Quaternion Expression): The left quaternion.
        right (Quaternion Expression): The right quaternion.
        order (str, optional): Component order, "xyzw" or "wxyz". Omit for columns
            that declare their own via ``quat``. Required for plain columns; there
            is no default convention.

    Returns:
        Expression (Quaternion Expression): The product, extension-typed.
    """
    return daft.get_function(_fn("rotation_quat_multiply", order), left, right)


def quat_inverse(quaternion: Expression, *, order: QuatOrder | None = None) -> Expression:
    """Inverse of a quaternion, the conjugate divided by the squared norm.

    For unit quaternions this is the conjugate, and it undoes the rotation. Zero
    quaternions, and rows holding a non-finite value, produce null.

    Args:
        quaternion (Quaternion Expression): The quaternion.
        order (str, optional): Component order, "xyzw" or "wxyz". Omit for columns
            that declare their own via ``quat``. Required for plain columns; there
            is no default convention.

    Returns:
        Expression (Quaternion Expression): The inverse, extension-typed.
    """
    return daft.get_function(_fn("rotation_quat_inverse", order), quaternion)


def rot6d_to_matrix(rot6d: Expression) -> Expression:
    """Converts the 6D continuous rotation representation into a 3x3 rotation matrix.

    The six values are the first two columns of the matrix. They are orthonormalised
    by Gram-Schmidt, and the third column is their cross product, so the result always
    lies in SO(3). This is the representation of Zhou et al. (2019), widely used as a
    network output because it is continuous, unlike quaternions and Euler angles.

    Rows whose two vectors are zero or parallel, or which hold a non-finite value,
    produce null, since they determine no rotation.

    Args:
        rot6d (FixedSizeList[Float64, 6] Expression): The 6D rotation representation.

    Returns:
        Expression (Tensor[Float64, [3, 3]] Expression): The rotation matrix.
    """
    return daft.get_function("rotation_rot6d_to_matrix", rot6d)


def quat_to_matrix(quaternion: Expression, *, order: QuatOrder | None = None) -> Expression:
    """Converts a quaternion into a 3x3 rotation matrix.

    The quaternion is normalised first. Zero quaternions, and rows holding a
    non-finite value, produce null.

    Args:
        quaternion (Quaternion Expression): The quaternion.
        order (str, optional): Component order, "xyzw" or "wxyz". Omit for columns
            that declare their own via ``quat``. Required for plain columns; there
            is no default convention.

    Returns:
        Expression (Tensor[Float64, [3, 3]] Expression): The rotation matrix.
    """
    return daft.get_function(_fn("rotation_quat_to_matrix", order), quaternion)


def quat_rotate(
    quaternion: Expression, vector: Expression, *, order: QuatOrder | None = None
) -> Expression:
    """Rotates a 3-vector by a quaternion.

    The quaternion is normalised first, so the length of the vector is preserved.
    Zero quaternions, and rows where the quaternion or the vector holds a
    non-finite value, produce null.

    Args:
        quaternion (Quaternion Expression): The quaternion.
        vector (FixedSizeList[Float64, 3] Expression): The vector to rotate.
        order (str, optional): Component order of the quaternion, "xyzw" or "wxyz".
            Omit for columns that declare their own via ``quat``. Required for plain
            columns; there is no default convention.

    Returns:
        Expression (FixedSizeList[Float64, 3] Expression): The rotated vector.
    """
    return daft.get_function(_fn("rotation_quat_rotate", order), quaternion, vector)
