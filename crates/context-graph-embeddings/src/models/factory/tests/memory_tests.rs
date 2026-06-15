//! Memory estimation and thread safety tests.

use std::path::PathBuf;

use crate::config::GpuConfig;
use crate::models::factory::DefaultModelFactory;
use crate::traits::ModelFactory;
use crate::types::ModelId;

#[test]
fn test_estimate_memory_all_nonzero() {
    let factory = DefaultModelFactory::new(PathBuf::from("./models"), GpuConfig::default());

    for model_id in ModelId::all() {
        let estimate = factory.estimate_memory(*model_id);
        assert!(
            estimate > 0,
            "Memory estimate for {:?} should be > 0",
            model_id
        );
    }
}

#[test]
fn test_factory_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DefaultModelFactory>();
}
