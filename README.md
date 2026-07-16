# daft-rotation

Native quaternion, 6D, and SO(3) rotation expressions for the Daft query engine.

## Why

Rotations are the most common non-scalar quantity in robot trajectory, camera extrinsic, and hand-joint-orientation datasets. Daft stores them as columns of floats without semantics. This extension operates on them natively, so composing, inverting, and comparing rotations does not require a Python user-defined function.

## Install

Currently, this package builds only against a local Daft checkout, and it must be a **sibling directory** of `daft-rotation`: `Cargo.toml` depends on `../daft/src/daft-ext` and `pyproject.toml`'s `[tool.uv.sources]` depends on `../daft`, both relative paths.

```bash
git clone https://github.com/eventual-inc/daft.git
git clone https://github.com/alejandrosotofranco/daft-rotation.git
cd daft-rotation
uv sync
```

Use `uv sync`, not `pip install -e .`: `pyproject.toml` declares `dependencies = ["daft"]`, a bare PyPI requirement, and the `[tool.uv.sources]` entry that redirects it to the sibling checkout is a uv-only override that plain `pip` never reads. Under `pip install -e .`, `daft` resolves from PyPI instead, leaving you with PyPI's Python API running against a cdylib compiled against `../daft/src/daft-ext`, a version mismatch between the two. `uv sync` honours `[tool.uv.sources]` and installs the sibling `../daft` editable, leaving `daft` and `daft-rotation` as true siblings under the same parent directory. This is necessary because the published `daft-ext` SDK has not yet stabilised its API and lacks the prelude helpers and the `daft_extension` macro this crate depends on. Once the extension system reaches stability, standard `pip install daft-rotation` will work.

## Example

```python
import daft
from daft.session import Session

import daft_rotation
from daft_rotation import quat, quat_multiply

# Kernel-backed functions dispatch through daft.get_function, which
# requires the extension to be loaded into an active session first.
session = Session()
session.load_extension(daft_rotation)

with session:
    # A table with quaternion columns in xyzw order
    df = daft.from_pydict(
        {
            "joint_quat": [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
            ],
            "tool_offset_quat": [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    )

    # Declare the convention once per column
    df = df.with_column("joint_quat", quat(df["joint_quat"], "xyzw"))
    df = df.with_column("tool_offset_quat", quat(df["tool_offset_quat"], "xyzw"))

    # Compose rotations (left is applied after right)
    df = df.with_column(
        "composed",
        quat_multiply(df["joint_quat"], df["tool_offset_quat"]),
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
import daft
from daft_rotation import infer_quat_order

# A handful of near-identity frames (small rotations), xyzw order
rows = [
    [0.000859, -0.017163, -0.001102, 0.999852],
    [-0.003873, -0.000581, 0.002694, 0.999989],
    [0.00095, 0.000549, 0.000408, 0.999999],
    [-0.000403, -0.009408, -0.008655, 0.999918],
    [0.018999, -0.002063, 0.007379, 0.99979],
    [-0.003576, 0.008788, -0.006483, 0.999934],
    [-0.000598, -0.000439, 0.001179, 0.999999],
    [-0.013483, -0.018839, 0.00274, 0.999728],
    [-0.002524, 0.007006, -0.009378, 0.999928],
    [-0.000477, -0.002282, 0.00449, 0.999987],
    [0.024916, -0.001006, -0.000421, 0.999689],
    [0.010463, 0.005531, 0.014249, 0.999828],
]
df = daft.from_pydict({"q": rows})

report = infer_quat_order(df, "q")
print(report)
```

Output:

```
sampled 12 rows
  slot 0: mean|v|=0.0068 var=0.0001
  slot 3: mean|v|=0.9999 var=0.0000
  likely: xyzw (confidence: high)
  basis: slot 3 concentrates near |1|, consistent with a scalar component over near-identity frames
  declare it with: quat(col, "xyzw")
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

The `order` parameter is optional for `quat_multiply`, `quat_inverse`, `quat_to_matrix`, and `quat_rotate`: pass `None` (default) to read the convention from the column's dtype, or a string to override. For `matrix_to_quat`, and for a plain (untyped) column passed to any of the four, `order` is required.

## Status

Daft's extension system is experimental. This library is a native extension and shares that status. Report issues on [GitHub](https://github.com/alejandrosotofranco/daft-rotation).
