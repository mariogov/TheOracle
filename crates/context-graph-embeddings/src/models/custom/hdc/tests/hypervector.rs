//! Random hypervector generation tests.

use super::*;

#[test]
fn test_random_hypervector_dimension() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(42);
    assert_eq!(hv.len(), HDC_DIMENSION);
}
