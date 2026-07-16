use std::sync::Arc;

use daft_ext::prelude::*;

mod ffi;
mod functions;
mod math;
mod order;

#[daft_extension]
struct RotationExtension;

impl DaftExtension for RotationExtension {
    fn install(session: &mut dyn DaftSession) {
        session.define_function(Arc::new(functions::GeodesicAngle));
        session.define_function(Arc::new(functions::MatrixToQuat(math::QuatOrder::Xyzw)));
        session.define_function(Arc::new(functions::MatrixToQuat(math::QuatOrder::Wxyz)));
    }
}
