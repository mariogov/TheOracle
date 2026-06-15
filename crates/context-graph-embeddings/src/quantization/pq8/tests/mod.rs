//! Tests for PQ-8 quantization.
//!
//! Test modules:
//! - `encoder_tests` - Basic encoder functionality and error handling
//! - `training_tests` - Codebook training with k-means
//! - `persistence_tests` - Codebook save/load
//! - `dimension_tests` - Constitution-specified dimensions (768, 1024, 1536)

mod dimension_tests;
mod encoder_tests;
mod persistence_tests;
mod training_tests;
