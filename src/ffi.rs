//! Shared plumbing between the FFI boundary and `crate::math`.
//!
//! Row access goes through arrow-rs accessors that compose the array's own
//! offset. Never index a child buffer with `row * width` arithmetic: a sliced
//! input carries a non-zero offset and that arithmetic silently reads the
//! wrong window. See tests/test_slices.py.

use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array,
    builder::{FixedSizeListBuilder, Float64Builder},
    cast::AsArray,
};
use arrow_schema::{DataType, Field};
use daft_ext::prelude::{DaftError, DaftResult};

/// Read-only view over a fixed-size list column of `Float64`, by row.
pub(crate) struct FixedRows<'a> {
    array: &'a FixedSizeListArray,
    width: usize,
}

impl<'a> FixedRows<'a> {
    /// Validate that `array` is `FixedSizeList[Float64, width]` and wrap it.
    pub(crate) fn new(array: &'a ArrayRef, width: usize, what: &str) -> DaftResult<Self> {
        let fsl = array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                DaftError::TypeError(format!(
                    "{what}: expected FixedSizeList[Float64, {width}], got {:?}",
                    array.data_type()
                ))
            })?;
        if fsl.value_length() as usize != width {
            return Err(DaftError::TypeError(format!(
                "{what}: expected {width} components per row, got {}",
                fsl.value_length()
            )));
        }
        if !matches!(fsl.values().data_type(), DataType::Float64) {
            return Err(DaftError::TypeError(format!(
                "{what}: expected Float64 components, got {:?}",
                fsl.values().data_type()
            )));
        }
        Ok(Self { array: fsl, width })
    }

    pub(crate) fn len(&self) -> usize {
        self.array.len()
    }

    /// The `i`th row, or `None` when the row is null or holds a null component.
    ///
    /// Parent offsets are folded into child values at construction, so `value(i)`
    /// correctly handles sliced parents. Do not index `self.array.values()` with
    /// `i * width` arithmetic; it re-reads the unfolded buffer with a shifted window.
    pub(crate) fn get(&self, i: usize) -> Option<[f64; 4]> {
        assert_eq!(self.width, 4, "get() is the 4-wide specialisation");
        if self.array.is_null(i) {
            return None;
        }
        let row = self.array.value(i);
        let row: &Float64Array = row.as_primitive();
        if row.null_count() > 0 {
            return None;
        }
        Some([row.value(0), row.value(1), row.value(2), row.value(3)])
    }

    /// The `i`th row as a slice-backed `Vec`, or `None` when null.
    ///
    /// Used for widths other than 4. Allocates per row; acceptable here because
    /// every caller immediately consumes it into fixed-size mathematics.
    pub(crate) fn get_vec(&self, i: usize) -> Option<Vec<f64>> {
        if self.array.is_null(i) {
            return None;
        }
        let row = self.array.value(i);
        let row: &Float64Array = row.as_primitive();
        if row.null_count() > 0 {
            return None;
        }
        Some((0..self.width).map(|j| row.value(j)).collect())
    }
}

/// A plain nullable `Float64` output field.
pub(crate) fn float64_field(name: &str) -> Field {
    Field::new(name, DataType::Float64, true)
}

/// The storage type of a quaternion column.
///
/// The child field name and nullability must match what Daft produces, or
/// `DataArray::from_arrow` rejects the array. Taken from probe/GROUND-TRUTH.md,
/// which records child name "item", child nullable true, outer nullable true.
pub(crate) fn quat_storage() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 4)
}

/// The output length for a set of argument lengths.
///
/// A length-1 argument broadcasts against a longer one, matching how Daft
/// treats a literal. Any other disagreement is an error.
pub(crate) fn broadcast_len(lens: &[usize], what: &str) -> DaftResult<usize> {
    let n = lens.iter().copied().max().unwrap_or(0);
    for &l in lens {
        if l != n && l != 1 {
            return Err(DaftError::RuntimeError(format!(
                "{what}: cannot broadcast lengths {lens:?}"
            )));
        }
    }
    Ok(n)
}

impl FixedRows<'_> {
    /// The `i`th row, treating a length-1 column as a broadcast constant.
    pub(crate) fn get_broadcast(&self, i: usize) -> Option<[f64; 4]> {
        self.get(if self.len() == 1 { 0 } else { i })
    }
}

/// The storage type of a 3x3 rotation matrix column: the FixedSizeList that
/// backs the arrow.fixed_shape_tensor extension type.
pub(crate) fn tensor_storage() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 9)
}

/// A `Tensor[Float64, [3, 3]]` output field.
///
/// The extension tag is what makes this a tensor rather than a flat list of 9:
/// Daft's decoder reads ARROW:extension:name and the JSON shape metadata and
/// rebuilds DataType::FixedShapeTensor. Storage alone is not enough.
pub(crate) fn tensor3x3_field(name: &str) -> Field {
    let mut md = HashMap::new();
    md.insert(
        "ARROW:extension:name".to_string(),
        "arrow.fixed_shape_tensor".to_string(),
    );
    md.insert(
        "ARROW:extension:metadata".to_string(),
        r#"{"shape":[3,3]}"#.to_string(),
    );
    Field::new(name, tensor_storage(), true).with_metadata(md)
}

/// The storage type of a 3-vector column. A plain fixed-size list, not an
/// extension type: a 3-vector carries no shape metadata and needs no tag.
pub(crate) fn vec3_storage() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 3)
}

impl FixedRows<'_> {
    /// The `i`th row as a `Vec`, treating a length-1 column as a broadcast constant.
    pub(crate) fn get_vec_broadcast(&self, i: usize) -> Option<Vec<f64>> {
        self.get_vec(if self.len() == 1 { 0 } else { i })
    }
}

/// Append one row to a fixed-size list builder, or a null row.
///
/// The null path must append exactly `width` child values before `append(false)`:
/// `FixedSizeListBuilder::finish` asserts `values.len() == len * list_len`, so a wrong
/// count panics there rather than corrupting later rows. Centralising it means that
/// contract is honoured in one place instead of six.
pub(crate) fn append_row(
    builder: &mut FixedSizeListBuilder<Float64Builder>,
    row: Option<impl IntoIterator<Item = f64>>,
    width: usize,
) {
    match row {
        Some(values) => {
            let mut n = 0;
            for v in values {
                builder.values().append_value(v);
                n += 1;
            }
            debug_assert_eq!(n, width, "append_row: row had {n} values, expected {width}");
            builder.append(true);
        }
        None => {
            for _ in 0..width {
                builder.values().append_null();
            }
            builder.append(false);
        }
    }
}
