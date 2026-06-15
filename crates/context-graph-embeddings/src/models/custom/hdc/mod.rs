//! Hyperdimensional Computing (HDC) embedding model (E9).
//!
//! HDC uses high-dimensional binary vectors (hypervectors) for robust, noise-tolerant
//! representations. This implementation uses 10,000-bit binary hypervectors with
//! the `bitvec` crate for efficient bit manipulation.
//!
//! # Core Operations
//!
//! - **Random**: Generate random hypervector with ~50% bits set
//! - **Bind (XOR)**: Associate two concepts (reversible: A ^ B ^ B = A)
//! - **Bundle (Majority)**: Combine multiple vectors preserving similarity
//! - **Permute**: Circular bit shift for positional encoding
//!
//! # Text Encoding
//!
//! Text is encoded via character n-grams with position binding:
//! 1. Generate random hypervector for each character
//! 2. Extract n-grams (default: trigrams) and bind with position vectors
//! 3. Bundle all n-gram vectors into final representation
//!
//! # Projection
//!
//! The 10K-bit binary vector is projected to 1024D floating-point for fusion:
//! - Binary to float: 0 -> -1.0, 1 -> +1.0
//! - L2 normalization applied
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Invalid inputs return errors immediately
//! - **FAIL FAST**: Configuration errors detected at construction
//! - **DETERMINISTIC**: Same input always produces same output (seeded RNG)
//!
//! # Module Structure
//!
//! - `types`: Core constants and type definitions
//! - `operations`: Hypervector operations (bind, bundle, permute, similarity)
//! - `encoding`: Text encoding and float projection
//! - `model`: HdcModel struct and EmbeddingModel implementation

mod encoding;
mod model;
mod operations;
mod types;

// Re-export public API for backwards compatibility
pub use model::HdcModel;
pub use types::{
    Hypervector, DEFAULT_NGRAM_SIZE, DEFAULT_SEED, HDC_DIMENSION, HDC_PROJECTED_DIMENSION,
};

#[cfg(test)]
mod tests;
