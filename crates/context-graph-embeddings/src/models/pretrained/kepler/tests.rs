//! Tests for KeplerModel.

use super::*;

#[test]
fn test_kepler_dimension() {
    assert_eq!(KEPLER_DIMENSION, 768);
}

#[test]
fn test_encode_entity_with_type() {
    let text = KeplerModel::encode_entity("Paris", Some("location"));
    assert_eq!(text, "[LOCATION] Paris");
}
