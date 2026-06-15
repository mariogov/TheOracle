//! Construction and initialization tests for TemporalPeriodicModel.

use crate::models::custom::temporal_periodic::TemporalPeriodicModel;
use crate::traits::EmbeddingModel;

#[test]
fn test_new_creates_initialized_model() {
    let model = TemporalPeriodicModel::new();

    assert!(
        model.is_initialized(),
        "Custom model must be initialized immediately"
    );
    assert_eq!(model.periods.len(), 5, "Must have 5 periods");
}
