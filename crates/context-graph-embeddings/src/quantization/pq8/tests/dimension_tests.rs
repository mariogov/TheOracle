//! Dimension-specific tests for PQ-8 quantization.
//!
//! Tests for Constitution-specified embedding dimensions:
//! - 768D (E5_Causal, E10_Multimodal)
//! - 1024D (E1_Semantic)
//! - 1536D (E7_Code)

use crate::quantization::pq8::encoder::PQ8Encoder;
use crate::quantization::pq8::training::generate_realistic_embeddings;
use crate::quantization::types::PQ8Codebook;
use std::sync::Arc;

#[test]
fn test_codebook_768d_causal() {
    // E5_Causal: 768D
    let samples = generate_realistic_embeddings(300, 768, 42);
    let codebook = PQ8Codebook::train(&samples, None).expect("training");

    assert_eq!(codebook.embedding_dim, 768);
    assert_eq!(codebook.centroids[0][0].len(), 768 / 8); // 96

    let encoder = PQ8Encoder::with_codebook(Arc::new(codebook));
    let test = generate_realistic_embeddings(1, 768, 99).remove(0);

    let q = encoder.quantize(&test).expect("quantize");
    let d = encoder.dequantize(&q).expect("dequantize");
    assert_eq!(d.len(), 768);
}

#[test]
fn test_codebook_1024d_semantic() {
    // E1_Semantic: 1024D
    let samples = generate_realistic_embeddings(300, 1024, 42);
    let codebook = PQ8Codebook::train(&samples, None).expect("training");

    assert_eq!(codebook.embedding_dim, 1024);
    assert_eq!(codebook.centroids[0][0].len(), 1024 / 8); // 128

    let encoder = PQ8Encoder::with_codebook(Arc::new(codebook));
    let test = generate_realistic_embeddings(1, 1024, 99).remove(0);

    let q = encoder.quantize(&test).expect("quantize");
    let d = encoder.dequantize(&q).expect("dequantize");
    assert_eq!(d.len(), 1024);
}

#[test]
fn test_codebook_1536d_code() {
    // E7_Code: 1536D
    let samples = generate_realistic_embeddings(300, 1536, 42);
    let codebook = PQ8Codebook::train(&samples, None).expect("training");

    assert_eq!(codebook.embedding_dim, 1536);
    assert_eq!(codebook.centroids[0][0].len(), 1536 / 8); // 192

    let encoder = PQ8Encoder::with_codebook(Arc::new(codebook));
    let test = generate_realistic_embeddings(1, 1536, 99).remove(0);

    let q = encoder.quantize(&test).expect("quantize");
    let d = encoder.dequantize(&q).expect("dequantize");
    assert_eq!(d.len(), 1536);
}
