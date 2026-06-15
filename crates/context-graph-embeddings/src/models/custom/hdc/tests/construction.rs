//! Construction and initialization tests for HdcModel.

use super::*;

#[test]
fn test_new_with_valid_params() {
    let model = HdcModel::new(3, 0xDEAD_BEEF).unwrap();
    assert_eq!(model.ngram_size(), 3);
    assert_eq!(model.seed(), 0xDEAD_BEEF);
    assert!(model.is_initialized());
}
