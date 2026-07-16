"""Records the exact arrow types Daft produces, so kernels can match them.

Not part of the package. Run it, paste the output into probe/GROUND-TRUTH.md.

This script builds the tensor column from a flat 9-element row because
casting a *nested* list of lists (``[[1,0,0],[0,1,0],[0,0,1]]``) directly to
``Tensor[Float64, (3, 3)]`` raises ``DaftError::ComputeError Cannot cast List
to FixedSizeList because not all elements have sizes: 9`` on this Daft
build. Daft's own ``tests/expressions/test_rotation.py`` casts from a
*flattened* 9-element row instead (``m.flatten().tolist()`` before
``.cast(MAT3)``), so the tensor column below is built the same way to match
the interface Daft actually accepts. The 4-wide quaternion cast has no such
restriction.
"""

from __future__ import annotations

import daft

df = daft.from_pydict(
    {
        "q": [[0.0, 0.0, 0.0, 1.0]],
        "m": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
    }
)
df = df.select(
    df["q"].cast(daft.DataType.fixed_size_list(daft.DataType.float64(), 4)),
    df["m"].cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3))),
)

print("daft schema:")
print(df.schema())

arrow = df.to_arrow()
print("\narrow schema:")
print(arrow.schema)
for field in arrow.schema:
    print(f"\nfield {field.name!r}:")
    print(f"  type      = {field.type!r}")
    print(f"  nullable  = {field.nullable}")

    t = field.type
    if hasattr(t, "value_field"):
        vf = t.value_field
        print(f"  child name     = {vf.name!r}")
        print(f"  child type     = {vf.type!r}")
        print(f"  child nullable = {vf.nullable}")

    # Tensor lowers to a pyarrow *extension type* (arrow.fixed_shape_tensor),
    # not a plain FixedSizeList: `value_field` lives on its `storage_type`,
    # not on the extension type itself.
    if hasattr(t, "extension_name"):
        st = t.storage_type
        print(f"  extension_name = {t.extension_name!r}")
        print(f"  storage_type   = {st!r}")
        if hasattr(st, "value_field"):
            svf = st.value_field
            print(f"  storage child name     = {svf.name!r}")
            print(f"  storage child type     = {svf.type!r}")
            print(f"  storage child nullable = {svf.nullable}")
