use std::{ffi::CStr, sync::Arc};

use arrow_array::{ArrayRef, Float64Array};
use daft_ext::prelude::{ArrowData, ArrowSchema, DaftError, DaftResult, DaftScalarFunction, export_array, export_field, import_array, import_field};

use crate::{ffi::{FixedRows, float64_field}, math};

pub(crate) struct GeodesicAngle;

impl DaftScalarFunction for GeodesicAngle {
    fn name(&self) -> &CStr {
        c"rotation_geodesic_angle"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 2 {
            return Err(DaftError::TypeError(format!(
                "rotation_geodesic_angle expects 2 arguments, got {}",
                args.len()
            )));
        }
        let field = import_field(&args[0])?;
        export_field(&float64_field(field.name()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        let mut it = args.into_iter();
        let a = import_array(it.next().expect("arity checked in return_field"))?;
        let b = import_array(it.next().expect("arity checked in return_field"))?;

        let out = geodesic_kernel(&a, &b)?;
        export_array(out, "rotation_geodesic_angle")
    }
}

fn geodesic_kernel(a: &ArrayRef, b: &ArrayRef) -> DaftResult<ArrayRef> {
    // Width comes from probe/GROUND-TRUTH.md. If a 3x3 tensor is not a
    // FixedSizeList[Float64, 9], replace FixedRows with the TensorRows
    // accessor written in Step 3.
    let a = FixedRows::new(a, 9, "rotation_geodesic_angle: left")?;
    let b = FixedRows::new(b, 9, "rotation_geodesic_angle: right")?;
    if a.len() != b.len() {
        // DaftError has exactly two variants, TypeError and RuntimeError.
        // There is no ValueError.
        return Err(DaftError::RuntimeError(format!(
            "rotation_geodesic_angle: length mismatch, {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let out: Float64Array = (0..a.len())
        .map(|i| match (a.get_vec(i), b.get_vec(i)) {
            (Some(x), Some(y)) => math::geodesic_angle(&x, &y),
            _ => None,
        })
        .collect();
    Ok(Arc::new(out))
}
