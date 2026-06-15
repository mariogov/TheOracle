//! Tests for sparse projection module.

use std::path::Path;

use super::error::ProjectionError;
use super::types::ProjectionMatrix;

#[test]
fn test_expected_shape_constants() {
    assert_eq!(ProjectionMatrix::EXPECTED_SHAPE, (30522, 1536));
    assert_eq!(ProjectionMatrix::input_dimension(), 30522);
    assert_eq!(ProjectionMatrix::output_dimension(), 1536);
}

#[test]
fn test_load_missing_file() {
    let result = ProjectionMatrix::load(Path::new("/nonexistent/path/that/does/not/exist"));

    assert!(result.is_err(), "load() must return Err for missing file");

    let err = result.unwrap_err();

    assert!(
        matches!(err, ProjectionError::MatrixMissing { .. }),
        "Error must be MatrixMissing variant, got: {:?}",
        err
    );

    let msg = format!("{}", err);
    assert!(
        msg.contains("EMB-E006"),
        "Error must contain code EMB-E006, got: {}",
        msg
    );
}
