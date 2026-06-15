//! Text encoding and projection tests.

use super::*;

#[test]
fn test_encode_text_produces_correct_dimension() {
    let model = HdcModel::default_model();
    let hv = model.encode_text("hello world");
    assert_eq!(hv.len(), HDC_DIMENSION);
}
