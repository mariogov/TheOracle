//! Tests for PQ-8 codebook training.
//!
//! Tests for:
//! - Synthetic embedding generation
//! - Codebook training with k-means
//! - Training error handling
//! - Trained codebook quality

use crate::quantization::pq8::encoder::PQ8Encoder;
use crate::quantization::pq8::training::generate_realistic_embeddings;
use crate::quantization::pq8::types::{PQ8QuantizationError, NUM_CENTROIDS, NUM_SUBVECTORS};
use crate::quantization::types::{PQ8Codebook, QuantizationMethod};
use std::sync::Arc;

#[test]
fn test_generate_realistic_embeddings() {
    let samples = generate_realistic_embeddings(100, 256, 42);
    assert_eq!(samples.len(), 100);
    assert_eq!(samples[0].len(), 256);

    // Verify normalization - each vector should have L2 norm ~= 1.0
    for (i, sample) in samples.iter().enumerate() {
        let norm: f32 = sample.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Sample {} has norm {} instead of ~1.0",
            i,
            norm
        );
    }

    // Verify no NaN/Inf
    for sample in &samples {
        for &v in sample {
            assert!(!v.is_nan(), "Found NaN in generated embedding");
            assert!(!v.is_infinite(), "Found Infinity in generated embedding");
        }
    }
}

#[test]
fn test_codebook_training_basic() {
    // Train with minimum required samples (256)
    let samples = generate_realistic_embeddings(300, 256, 42);
    let codebook = PQ8Codebook::train(&samples, None).expect("training should succeed");

    assert_eq!(codebook.embedding_dim, 256);
    assert_eq!(codebook.num_subvectors, NUM_SUBVECTORS);
    assert_eq!(codebook.num_centroids, NUM_CENTROIDS);
    assert_eq!(codebook.centroids.len(), NUM_SUBVECTORS);

    // Each subvector should have 256 centroids
    for sv_centroids in &codebook.centroids {
        assert_eq!(sv_centroids.len(), NUM_CENTROIDS);
        // Each centroid should have subvector_dim = 256/8 = 32 elements
        for centroid in sv_centroids {
            assert_eq!(centroid.len(), 256 / NUM_SUBVECTORS);
        }
    }
}

#[test]
fn test_codebook_training_insufficient_samples() {
    let samples = generate_realistic_embeddings(100, 256, 42); // < 256 samples
    let result = PQ8Codebook::train(&samples, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        PQ8QuantizationError::InsufficientSamples { required, provided } => {
            assert_eq!(required, NUM_CENTROIDS);
            assert_eq!(provided, 100);
        }
        e => panic!("Expected InsufficientSamples, got {:?}", e),
    }
}

#[test]
fn test_codebook_training_dimension_mismatch() {
    let mut samples = generate_realistic_embeddings(300, 256, 42);
    samples[50] = vec![0.1; 128]; // Wrong dimension

    let result = PQ8Codebook::train(&samples, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        PQ8QuantizationError::SampleDimensionMismatch {
            sample_idx,
            expected,
            got,
        } => {
            assert_eq!(sample_idx, 50);
            assert_eq!(expected, 256);
            assert_eq!(got, 128);
        }
        e => panic!("Expected SampleDimensionMismatch, got {:?}", e),
    }
}

#[test]
fn test_codebook_training_nan_in_sample() {
    let mut samples = generate_realistic_embeddings(300, 256, 42);
    samples[10][5] = f32::NAN;

    let result = PQ8Codebook::train(&samples, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        PQ8QuantizationError::ContainsNaN { index } => {
            assert_eq!(index, 5);
        }
        e => panic!("Expected ContainsNaN, got {:?}", e),
    }
}

#[test]
fn test_trained_codebook_quantization_roundtrip() {
    // Train codebook on clustered synthetic data
    let training_samples = generate_realistic_embeddings(1000, 256, 42);
    let codebook = PQ8Codebook::train(&training_samples, None).expect("training");

    // Create encoder with trained codebook
    let trained_encoder = PQ8Encoder::with_codebook(Arc::new(codebook));

    // Also create default encoder to compare
    let default_encoder = PQ8Encoder::new(256);

    // Test embeddings from same seed (overlap with training clusters)
    let test_samples = generate_realistic_embeddings(50, 256, 42);

    let mut trained_total = 0.0f32;
    let mut default_total = 0.0f32;

    for sample in &test_samples {
        // Test with trained codebook
        let quantized = trained_encoder.quantize(sample).expect("quantize");
        let reconstructed = trained_encoder.dequantize(&quantized).expect("dequantize");

        assert_eq!(reconstructed.len(), sample.len());
        assert_eq!(quantized.data.len(), 8);
        assert_eq!(quantized.method, QuantizationMethod::PQ8);

        // Compute cosine similarity for trained
        let dot: f32 = sample
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = sample.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let trained_cosine = dot / (norm_a * norm_b);
        trained_total += trained_cosine;

        // Test with default codebook for comparison
        let default_q = default_encoder.quantize(sample).expect("quantize default");
        let default_r = default_encoder
            .dequantize(&default_q)
            .expect("dequantize default");
        let dot: f32 = sample
            .iter()
            .zip(default_r.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_b: f32 = default_r.iter().map(|x| x * x).sum::<f32>().sqrt();
        let default_cosine = dot / (norm_a * norm_b);
        default_total += default_cosine;
    }

    let trained_avg = trained_total / test_samples.len() as f32;
    let _default_avg = default_total / test_samples.len() as f32;

    // Key assertion: trained codebook must produce meaningful results
    assert!(
        trained_avg > 0.4,
        "Trained codebook avg cosine {} too low - training may be broken",
        trained_avg
    );
}

#[test]
fn test_trained_codebook_recall_verification() {
    // Train on dataset
    let training_samples = generate_realistic_embeddings(2000, 256, 42);
    let codebook = PQ8Codebook::train(&training_samples, None).expect("training");
    let trained_encoder = PQ8Encoder::with_codebook(Arc::new(codebook));

    // Default encoder for comparison
    let default_encoder = PQ8Encoder::new(256);

    // Test samples
    let test_samples = generate_realistic_embeddings(100, 256, 42);

    let mut trained_total = 0.0f32;
    let mut default_total = 0.0f32;
    let mut min_trained = f32::MAX;

    for sample in &test_samples {
        // Trained codebook
        let quantized = trained_encoder.quantize(sample).expect("quantize");
        let reconstructed = trained_encoder.dequantize(&quantized).expect("dequantize");

        let dot: f32 = sample
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = sample.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let trained_cosine = dot / (norm_a * norm_b);
        trained_total += trained_cosine;
        min_trained = min_trained.min(trained_cosine);

        // Default codebook
        let q = default_encoder.quantize(sample).expect("q");
        let r = default_encoder.dequantize(&q).expect("r");
        let dot: f32 = sample.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
        let norm_b: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        default_total += dot / (norm_a * norm_b);
    }

    let trained_avg = trained_total / test_samples.len() as f32;
    let _default_avg = default_total / test_samples.len() as f32;

    // Verify trained codebook produces reasonable results
    assert!(
        trained_avg > 0.35,
        "Trained codebook avg {} too low - algorithm may be broken",
        trained_avg
    );

    // Verify no catastrophic failures
    assert!(
        min_trained > 0.2,
        "Min cosine {} indicates catastrophic reconstruction failure",
        min_trained
    );
}
