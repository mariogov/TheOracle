//! Bind, bundle, permute, and similarity operation tests.

use super::*;

#[test]
fn test_bind_self_inverse() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(1);
    let b = model.random_hypervector(2);
    let bound = HdcModel::bind(&a, &b);
    let unbound = HdcModel::bind(&bound, &b);
    assert_eq!(a, unbound, "A ^ B ^ B should equal A");
}
