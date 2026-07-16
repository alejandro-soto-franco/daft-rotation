# Ground truth: Daft's arrow types

## arrow types as of 2026-07-16

Recorded by running `probe/dump_schema.py` against the `daft` 0.3.0-dev0
checkout at `/home/alejandrosotofranco/daft`, with
`/home/alejandrosotofranco/.venvs/global/bin/python`.

### Why this script flattens the matrix

An earlier version of `probe/dump_schema.py` cast a **nested** list of lists
(`[[1,0,0],[0,1,0],[0,0,1]]`) directly to `Tensor[Float64, (3, 3)]`. On this
Daft build that raises, at `to_arrow()` time:

```
daft.exceptions.DaftCoreException: DaftError::ComputeError Cannot cast List
to FixedSizeList because not all elements have sizes: 9
```

This reproduces identically whether the nested list appears in the probe
script or in `tests/test_rotation.py` (verified by running the exact
`IDENTITY`/`.cast(daft.DataType.tensor(...))` snippet standalone). Daft's own
`daft/tests/expressions/test_rotation.py` (which defines the real
`rotation_geodesic_angle` etc. that this crate's functions are meant to
match) never casts a nested list either; it always flattens first:
`m.flatten().tolist()` before `.cast(MAT3)`.

`probe/dump_schema.py` builds the tensor column from a flat 9-element row
instead of a nested 3x3 list (the `q` / `FixedSizeList[Float64, 4]` cast has
no such restriction). `tests/test_rotation.py` follows the same shape:
`IDENTITY`, the quarter-turn matrix, and the non-rotation "junk" matrix are
all flat, row-major, 9-element lists.

### Verbatim probe output

```
daft schema:
╭─────────────┬─────────────────────────╮
│ Column Name ┆ DType                   │
╞═════════════╪═════════════════════════╡
│ q           ┆ List[Float64; 4]        │
├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ m           ┆ Tensor[Float64; [3, 3]] │
╰─────────────┴─────────────────────────╯


arrow schema:
q: fixed_size_list<item: double>[4]
  child 0, item: double
m: extension<arrow.fixed_shape_tensor[value_type=double, shape=[3,3]]>

field 'q':
  type      = FixedSizeListType(fixed_size_list<item: double>[4])
  nullable  = True
  child name     = 'item'
  child type     = DataType(double)
  child nullable = True

field 'm':
  type      = FixedShapeTensorType(extension<arrow.fixed_shape_tensor[value_type=double, shape=[3,3]]>)
  nullable  = True
  extension_name = 'arrow.fixed_shape_tensor'
  storage_type   = FixedSizeListType(fixed_size_list<item: double>[9])
  storage child name     = 'item'
  storage child type     = DataType(double)
  storage child nullable = True
```

(Note: Daft's own schema printer displays `FixedSizeList[Float64, N]` as
`List[Float64; N]` — this is Daft's display convention, not evidence that the
underlying arrow type is a variable-length `List`. The arrow schema section,
and the `to_arrow()` pyarrow types below it, are the ground truth.)

### What this means for `src/ffi.rs`

- **`FixedSizeList[Float64, 4]`** (the `q` quaternion column in the probe;
  used by the quaternion kernels): child field name is `"item"`, child nullable `True`,
  outer field nullable `True`. This is arrow-rs's own default child name for
  `FixedSizeListArray`, so `FixedRows` needs no special-casing for it.

- **`Tensor[Float64, [3, 3]]`**: on the **pyarrow side**, this is not a plain
  `FixedSizeList` — it is an Arrow **extension type**
  (`arrow.fixed_shape_tensor`) whose `storage_type` is
  `FixedSizeList<item: double>[9]` (child name `"item"`, child nullable
  `True`, outer field nullable `True`).

  The open question is what the **arrow-rs** side of the
  `daft-ext` FFI boundary (`import_array`/`import_field` in
  `daft-ext/src/helpers.rs`, via `arrow_array_57::ffi::from_ffi` /
  `Field::try_from`) sees for this column, since arrow-rs's `DataType` enum
  has no `Extension` variant — extension type identity in the C Data
  Interface travels as schema *metadata* (`ARROW:extension:name` /
  `ARROW:extension:metadata`), not as part of the format string, and
  arrow-rs's core `DataType` model does not represent it separately from its
  storage type.

  This was settled empirically, not by reading arrow-rs source: with
  `crate::ffi::FixedRows::new(array, 9, "...")` wired into
  `rotation_geodesic_angle` (`src/functions.rs`) and built into the cdylib,
  `tests/test_rotation.py`'s three tests — using genuine
  `Tensor[Float64, [3,3]]`-typed columns end to end through
  `df.select(...).cast(...)` and the registered Daft function — **passed
  without `FixedRows::new` ever rejecting the array**. If the arrow-rs side
  saw anything other than `FixedSizeList[Float64, 9]` (a struct, a nested
  list, a different width), the `downcast_ref::<FixedSizeListArray>()` or the
  `value_length() != 9` check in `FixedRows::new` would have produced a
  `TypeError` and failed the test at `.collect()`/`.to_pydict()`, not at
  cast time. This is conclusive: **`Tensor[Float64, [3, 3]]` lowers to a
  plain `FixedSizeListArray` of width 9 and `Float64` values at the arrow-rs
  FFI boundary daft-ext uses.** The `arrow.fixed_shape_tensor` extension
  identity is transparent to arrow-rs at this boundary; it is not exposed as
  a distinct `DataType`.

  Conclusion: **no tensor-specific row accessor is needed.** `FixedRows::get_vec`
  with `width = 9` is correct and is what `src/functions.rs` uses.

### Practical implication for constructing tensor-typed test data

Any test or example that builds a `Tensor[Float64, [3, 3]]` column from
Python must supply a **flat 9-element list per row**, then
`.cast(daft.DataType.tensor(daft.DataType.float64(), (3, 3)))`. Supplying a
nested `[[...], [...], [...]]` per row raises `DaftError::ComputeError
Cannot cast List to FixedSizeList because not all elements have sizes: 9` at
`.collect()`/`.to_arrow()`/`.to_pydict()` time. This applies to any
3x3-matrix fixture.
