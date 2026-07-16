# daft-rotation

Native quaternion, 6D, and SO(3) rotation expressions for the Daft query engine.

## Why

Rotations are the most common non-scalar quantity in robot trajectory, camera extrinsic, and hand-joint-orientation datasets. Daft stores them as columns of floats without semantics. This extension operates on them natively, so composing, inverting, and comparing rotations does not require a Python user-defined function.

## Install

Currently, this package builds only against a local Daft checkout:

```bash
git clone https://github.com/alejandrosotofranco/daft-rotation.git
cd daft-rotation
pip install -e .
```

This is necessary because the published `daft-ext` SDK has not yet stabilised its API. Once the extension system reaches stability, standard `pip install daft-rotation` will work.

## Example

```python
import daft
from daft_rotation import quat, quat_multiply

# Load a table with quaternion columns in xyzw order
df = daft.read_parquet("robot_trajectories.parquet")

# Declare the convention once
df = df.with_column("joint_quat", quat(df["joint_quat"], "xyzw"))

# Compose rotations (left is applied after right)
df = df.with_column(
    "composed",
    quat_multiply(df["joint_quat"], df["tool_offset_quat"])
)

# Retrieve the result
df.select("composed").show()
```

## Quaternion component order

Two conventions exist in the wild:
- **xyzw** (ROS, tf2, scipy): stores x, y, z, w
- **wxyz** (Eigen, MuJoCo, Isaac Sim): stores w, x, y, z

Because both quaternion arrays are mathematically indistinguishable until they are interpreted, feeding one convention to code expecting the other silently produces plausible, wrong rotations that pass every numerical test. This library **never assumes a convention**.

Instead, you declare it once per column:

```python
# Declare at load
df = df.with_column("q", quat(df["q"], "xyzw"))

# Or per call (useful for raw columns)
df = df.with_column("rotated", quat_rotate(df["raw_q"], df["v"], order="wxyz"))
```

A column without a declared order is an error, not a guess. If you do not know which convention your data uses, inspect it first:

```python
from daft_rotation import infer_quat_order

report = infer_quat_order(df, "q")
print(report)
```

Output:

```
OrderReport(
    column='q',
    n_samples=1000,
    n_valid=987,
    hypothesis='xyzw',
    evidence='0.98 (unit quaternions), score_xyzw=0.997 > score_wxyz=0.003'
)
```

The library reports the evidence, never decides. You then declare it explicitly.

## API

| Function | Purpose |
|----------|---------|
| `quat(col, order)` | Declare a column's component order by casting to an extension dtype. |
| `infer_quat_order(df, col, n=1000)` | Inspect a column and return evidence for its likely convention. |
| `matrix_to_quat(m, *, order)` | Convert a 3×3 rotation matrix to a quaternion. |
| `quat_multiply(a, b, *, order=None)` | Hamilton product (quaternion composition). |
| `quat_inverse(q, *, order=None)` | Quaternion inverse (rotation undo). |
| `quat_to_matrix(q, *, order=None)` | Convert a quaternion to a 3×3 rotation matrix. |
| `quat_rotate(q, v, *, order=None)` | Rotate a 3-vector by a quaternion. |
| `rot6d_to_matrix(r)` | Convert 6D rotation representation to a 3×3 matrix. |
| `rotation_geodesic_angle(a, b)` | Angle (radians) of relative rotation between two 3×3 matrices. |

The `order` parameter is optional for functions ending in `_multiply`, `_inverse`, `_to_matrix`, or `_rotate`: pass `None` (default) to read the convention from the column's dtype, or a string to override. For matrices and raw columns, `order` is required.

## Status

Daft's extension system is experimental. This library is a native extension and shares that status. Report issues on [GitHub](https://github.com/alejandrosotofranco/daft-rotation).
