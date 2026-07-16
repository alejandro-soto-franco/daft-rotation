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

        for order in [None, Some(math::QuatOrder::Xyzw), Some(math::QuatOrder::Wxyz)] {
            session.define_function(Arc::new(functions::QuatInverse(order)));
            session.define_function(Arc::new(functions::QuatMultiply(order)));
        }

        session.define_function(Arc::new(functions::Rot6dToMatrix));
        for order in [None, Some(math::QuatOrder::Xyzw), Some(math::QuatOrder::Wxyz)] {
            session.define_function(Arc::new(functions::QuatToMatrix(order)));
            session.define_function(Arc::new(functions::QuatRotate(order)));
        }
    }
}
