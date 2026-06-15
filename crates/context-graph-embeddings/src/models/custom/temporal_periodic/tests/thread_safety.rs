//! Thread safety and constants tests for TemporalPeriodicModel.

use crate::models::custom::temporal_periodic::TemporalPeriodicModel;

#[test]
fn test_model_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TemporalPeriodicModel>();
}
