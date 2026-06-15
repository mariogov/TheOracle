//! Core types and constants for the HDC (Hyperdimensional Computing) model.
//!
//! This module defines the fundamental constants and type aliases used throughout
//! the HDC implementation.

use bitvec::prelude::*;

/// Native dimension: 10,000 bits.
pub const HDC_DIMENSION: usize = 10_000;

/// Projected dimension for fusion pipeline: 1024.
pub const HDC_PROJECTED_DIMENSION: usize = 1024;

/// Default n-gram size for text encoding.
pub const DEFAULT_NGRAM_SIZE: usize = 3;

/// Seed for deterministic random hypervector generation.
pub const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// A 10,000-bit binary hypervector.
pub type Hypervector = BitVec<u64, Lsb0>;
