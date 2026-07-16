use std::{ffi::CStr, sync::Arc};

use arrow_array::{
    ArrayRef, Float64Array,
    builder::{FixedSizeListBuilder, Float64Builder},
};
use daft_ext::prelude::{ArrowData, ArrowSchema, DaftError, DaftResult, DaftScalarFunction, export_array, export_field, import_array, import_field};

use crate::{
    ffi::{FixedRows, float64_field, quat_storage},
    math::{self, QuatOrder},
    order::{quat_field, resolve_order},
};

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

/// A 3x3 rotation matrix to a quaternion, in the order the variant names.
pub(crate) struct MatrixToQuat(pub(crate) QuatOrder);

impl DaftScalarFunction for MatrixToQuat {
    fn name(&self) -> &CStr {
        match self.0 {
            QuatOrder::Xyzw => c"rotation_matrix_to_quat_xyzw",
            QuatOrder::Wxyz => c"rotation_matrix_to_quat_wxyz",
        }
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 1 {
            return Err(DaftError::TypeError(format!(
                "matrix_to_quat expects 1 argument, got {}",
                args.len()
            )));
        }
        let field = import_field(&args[0])?;
        // No quaternion input to read a convention from, so the order always
        // comes from the name. resolve_order is not consulted.
        export_field(&quat_field(field.name(), self.0, quat_storage()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        let m = import_array(args.into_iter().next().expect("arity checked in return_field"))?;
        let rows = FixedRows::new(&m, 9, "matrix_to_quat")?;

        let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 4);
        for i in 0..rows.len() {
            match rows.get_vec(i).and_then(|m| {
                let m: [f64; 9] = m.try_into().ok()?;
                // None for a matrix outside SO(3), which is how null propagates.
                math::mat_to_quat(m)
            }) {
                Some(q) => {
                    for c in self.0.write(q) {
                        builder.values().append_value(c);
                    }
                    builder.append(true);
                }
                None => {
                    for _ in 0..4 {
                        builder.values().append_null();
                    }
                    builder.append(false);
                }
            }
        }
        let out: ArrayRef = Arc::new(builder.finish());
        // export_array drops metadata, which is irrelevant: the host wraps the
        // result with return_field's field and checks only the physical type.
        export_array(out, "matrix_to_quat")
    }
}

/// Quaternion inverse: `conjugate(q) / |q|^2`, null for a zero or non-finite norm.
pub(crate) struct QuatInverse(pub(crate) Option<QuatOrder>);

impl DaftScalarFunction for QuatInverse {
    fn name(&self) -> &CStr {
        match self.0 {
            None => c"rotation_quat_inverse",
            Some(QuatOrder::Xyzw) => c"rotation_quat_inverse_xyzw",
            Some(QuatOrder::Wxyz) => c"rotation_quat_inverse_wxyz",
        }
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 1 {
            return Err(DaftError::TypeError(format!(
                "quat_inverse expects 1 argument, got {}",
                args.len()
            )));
        }
        let field = import_field(&args[0])?;
        let order = resolve_order(&field, self.0)?;
        export_field(&quat_field(field.name(), order, quat_storage()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        let mut it = args.into_iter();
        let (schema, data) = {
            let d = it.next().expect("arity checked");
            (import_field(&d.schema)?, d)
        };
        let order = resolve_order(&schema, self.0)?;
        let q = import_array(data)?;
        let rows = FixedRows::new(&q, 4, "quat_inverse")?;

        let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 4);
        for i in 0..rows.len() {
            match rows.get(i).and_then(|raw| math::inverse(order.read(&raw))) {
                Some(r) => {
                    for c in order.write(r) {
                        builder.values().append_value(c);
                    }
                    builder.append(true);
                }
                None => {
                    for _ in 0..4 {
                        builder.values().append_null();
                    }
                    builder.append(false);
                }
            }
        }
        export_array(Arc::new(builder.finish()), "quat_inverse")
    }
}

/// Hamilton product of two quaternions, each read according to its own convention.
pub(crate) struct QuatMultiply(pub(crate) Option<QuatOrder>);

impl DaftScalarFunction for QuatMultiply {
    fn name(&self) -> &CStr {
        match self.0 {
            None => c"rotation_quat_multiply",
            Some(QuatOrder::Xyzw) => c"rotation_quat_multiply_xyzw",
            Some(QuatOrder::Wxyz) => c"rotation_quat_multiply_wxyz",
        }
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 2 {
            return Err(DaftError::TypeError(format!(
                "quat_multiply expects 2 arguments, got {}",
                args.len()
            )));
        }
        let left = import_field(&args[0])?;
        let right = import_field(&args[1])?;
        // Resolve both so a bad right-hand argument is reported at plan time too.
        let left_order = resolve_order(&left, self.0)?;
        let _right_order = resolve_order(&right, self.0)?;
        export_field(&quat_field(left.name(), left_order, quat_storage()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        let mut it = args.into_iter();
        let a_data = it.next().expect("arity checked");
        let b_data = it.next().expect("arity checked");
        let a_order = resolve_order(&import_field(&a_data.schema)?, self.0)?;
        let b_order = resolve_order(&import_field(&b_data.schema)?, self.0)?;

        let a = import_array(a_data)?;
        let b = import_array(b_data)?;
        let ar = FixedRows::new(&a, 4, "quat_multiply: left")?;
        let br = FixedRows::new(&b, 4, "quat_multiply: right")?;
        let n = crate::ffi::broadcast_len(&[ar.len(), br.len()], "quat_multiply")?;

        let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), 4);
        for i in 0..n {
            let product = match (ar.get_broadcast(i), br.get_broadcast(i)) {
                (Some(x), Some(y)) => Some(math::multiply(a_order.read(&x), b_order.read(&y))),
                _ => None,
            };
            match product {
                Some(p) => {
                    // Output carries the left argument's convention.
                    for c in a_order.write(p) {
                        builder.values().append_value(c);
                    }
                    builder.append(true);
                }
                None => {
                    for _ in 0..4 {
                        builder.values().append_null();
                    }
                    builder.append(false);
                }
            }
        }
        export_array(Arc::new(builder.finish()), "quat_multiply")
    }
}
