//! Where a quaternion column's component order comes from.
//!
//! Two sources, never a default. A typed column carries its order in Arrow
//! extension metadata; a plain column takes it from the registered function
//! name the caller chose. A column with neither is an error, because guessing
//! is the failure this library exists to prevent: xyzw and wxyz data are
//! indistinguishable elementwise, so a wrong guess returns plausible,
//! silently incorrect rotations.

use std::collections::HashMap;

use arrow_schema::{DataType, Field};
use daft_ext::prelude::{DaftError, DaftResult};

use crate::math::QuatOrder;

/// Extension name for a quaternion column. Must match `daft_rotation/__init__.py`.
pub(crate) const QUAT_EXT_NAME: &str = "daft_rotation.quaternion";

pub(crate) const EXT_NAME_KEY: &str = "ARROW:extension:name";
pub(crate) const EXT_META_KEY: &str = "ARROW:extension:metadata";

/// Whether `field` is a quaternion column produced by this library.
pub(crate) fn is_quat_ext(field: &Field) -> bool {
    field.metadata().get(EXT_NAME_KEY).map(String::as_str) == Some(QUAT_EXT_NAME)
}

/// Build a quaternion output field carrying its order.
///
/// `export_field` preserves this metadata; `export_array` would drop it, which
/// is why output typing goes through `return_field`.
pub(crate) fn quat_field(name: &str, order: QuatOrder, storage: DataType) -> Field {
    let mut md = HashMap::new();
    md.insert(EXT_NAME_KEY.to_string(), QUAT_EXT_NAME.to_string());
    md.insert(EXT_META_KEY.to_string(), order_str(order).to_string());
    Field::new(name, storage, true).with_metadata(md)
}

/// Resolve the component order of a quaternion argument.
///
/// `from_name` is `Some` when the caller invoked an order-carrying function
/// name (`..._xyzw` / `..._wxyz`) and `None` for the generic name.
pub(crate) fn resolve_order(field: &Field, from_name: Option<QuatOrder>) -> DaftResult<QuatOrder> {
    if !is_quat_ext(field) {
        // Not ours. Any metadata it carries belongs to another library and is
        // not ours to interpret, so the name is the only source.
        return from_name.ok_or_else(|| {
            DaftError::TypeError(format!(
                "column '{}' is an untyped quaternion, so its component order is unknown. \
                 Pass order=\"xyzw\" or order=\"wxyz\", or declare it once with \
                 quat(col, order). If you do not know the convention, \
                 infer_quat_order(df, \"{}\") reports which it likely is.",
                field.name(),
                field.name()
            ))
        });
    }

    let raw = field.metadata().get(EXT_META_KEY).ok_or_else(|| {
        DaftError::TypeError(format!(
            "column '{}' is tagged {QUAT_EXT_NAME} but records no component order. \
             Add ARROW:extension:metadata of \"xyzw\" or \"wxyz\", or build the \
             field with quat(col, order).",
            field.name()
        ))
    })?;

    let from_meta = QuatOrder::parse(raw).ok_or_else(|| {
        DaftError::TypeError(format!(
            "column '{}' records component order '{raw}', which is neither \
             \"xyzw\" nor \"wxyz\"",
            field.name()
        ))
    })?;

    match from_name {
        None => Ok(from_meta),
        Some(named) if named == from_meta => Ok(from_meta),
        Some(named) => Err(DaftError::TypeError(format!(
            "column '{}' declares component order {}, but {} was requested. \
             Drop the order argument to use the column's own, or cast the column \
             if it is mislabelled.",
            field.name(),
            order_str(from_meta),
            order_str(named),
        ))),
    }
}

fn order_str(o: QuatOrder) -> &'static str {
    match o {
        QuatOrder::Xyzw => "xyzw",
        QuatOrder::Wxyz => "wxyz",
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_schema::{DataType, Field};

    use super::*;
    use crate::math::QuatOrder;

    fn storage() -> DataType {
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 4)
    }

    fn plain() -> Field {
        Field::new("q", storage(), true)
    }

    fn typed(order: &str) -> Field {
        let mut md = HashMap::new();
        md.insert(EXT_NAME_KEY.to_string(), QUAT_EXT_NAME.to_string());
        md.insert(EXT_META_KEY.to_string(), order.to_string());
        plain().with_metadata(md)
    }

    #[test]
    fn typed_column_supplies_its_own_order() {
        assert_eq!(resolve_order(&typed("xyzw"), None).unwrap(), QuatOrder::Xyzw);
        assert_eq!(resolve_order(&typed("wxyz"), None).unwrap(), QuatOrder::Wxyz);
    }

    #[test]
    fn name_supplies_order_for_a_plain_column() {
        assert_eq!(
            resolve_order(&plain(), Some(QuatOrder::Wxyz)).unwrap(),
            QuatOrder::Wxyz
        );
    }

    #[test]
    fn plain_column_without_an_order_is_rejected() {
        let err = resolve_order(&plain(), None).unwrap_err().to_string();
        // The message must tell the user how to proceed, since this is the
        // DROID case: a float64[4] column whose convention nobody recorded.
        assert!(err.contains("infer_quat_order"), "unhelpful error: {err}");
        assert!(err.contains("order"), "unhelpful error: {err}");
    }

    #[test]
    fn agreeing_order_is_allowed() {
        assert_eq!(
            resolve_order(&typed("xyzw"), Some(QuatOrder::Xyzw)).unwrap(),
            QuatOrder::Xyzw
        );
    }

    #[test]
    fn disagreeing_order_is_an_error_naming_both() {
        let err = resolve_order(&typed("xyzw"), Some(QuatOrder::Wxyz))
            .unwrap_err()
            .to_string();
        // Assert on the actual phrases in their roles, not just presence of the
        // two words: a message with xyzw and wxyz swapped would tell the user
        // the exact opposite of the truth and must fail this test.
        assert!(
            err.contains("declares component order xyzw"),
            "message must say the column declares xyzw: {err}"
        );
        assert!(
            err.contains("wxyz was requested"),
            "message must say wxyz was requested: {err}"
        );
    }

    #[test]
    fn unparseable_metadata_is_an_error_not_a_fallback() {
        // A column tagged with our extension name but a corrupt order must fail
        // loudly. Falling back to a default here would reintroduce the silent
        // guess the whole design exists to prevent.
        let err = resolve_order(&typed("wxzy"), None).unwrap_err().to_string();
        assert!(err.contains("wxzy"), "{err}");
    }

    #[test]
    fn extension_name_from_another_library_is_not_ours() {
        let mut md = HashMap::new();
        md.insert(EXT_NAME_KEY.to_string(), "someone_else.quaternion".to_string());
        md.insert(EXT_META_KEY.to_string(), "xyzw".to_string());
        let foreign = plain().with_metadata(md);
        assert!(!is_quat_ext(&foreign));
        // Treated as plain: its metadata is not ours to interpret.
        assert!(resolve_order(&foreign, None).is_err());
        assert_eq!(
            resolve_order(&foreign, Some(QuatOrder::Xyzw)).unwrap(),
            QuatOrder::Xyzw
        );
    }

    #[test]
    fn our_extension_name_without_metadata_is_an_error() {
        let mut md = HashMap::new();
        md.insert(EXT_NAME_KEY.to_string(), QUAT_EXT_NAME.to_string());
        let tagged = plain().with_metadata(md);
        assert!(resolve_order(&tagged, None).is_err());
    }

    #[test]
    fn metadata_without_extension_name_is_not_ours() {
        // The metadata key alone, without the extension-name key, must never
        // confer an order: a field like this behaves exactly like a plain
        // column, because metadata is only ours to interpret once the name
        // marks the field as ours.
        let mut md = HashMap::new();
        md.insert(EXT_META_KEY.to_string(), "xyzw".to_string());
        let f = plain().with_metadata(md);
        assert!(!is_quat_ext(&f));

        let err = resolve_order(&f, None).unwrap_err().to_string();
        assert!(err.contains("infer_quat_order"), "{err}");

        assert_eq!(
            resolve_order(&f, Some(QuatOrder::Wxyz)).unwrap(),
            QuatOrder::Wxyz
        );
    }

    #[test]
    fn metadata_order_matching_is_case_insensitive() {
        // QuatOrder::parse lowercases before matching, so upper-case metadata
        // still resolves correctly. This is not a guess: the value is still
        // one of the two known orders, just spelled differently.
        assert_eq!(resolve_order(&typed("XYZW"), None).unwrap(), QuatOrder::Xyzw);
    }

    #[test]
    fn metadata_order_matching_is_not_trimmed() {
        // QuatOrder::parse does not trim, so a leading space makes the value
        // unrecognised and must fail closed rather than silently match.
        let err = resolve_order(&typed(" xyzw"), None).unwrap_err().to_string();
        assert!(err.contains(" xyzw"), "{err}");
    }

    #[test]
    fn quat_field_round_trips_through_resolve() {
        for order in [QuatOrder::Xyzw, QuatOrder::Wxyz] {
            let f = quat_field("out", order, storage());
            assert!(is_quat_ext(&f));
            assert_eq!(resolve_order(&f, None).unwrap(), order);
        }
    }
}
