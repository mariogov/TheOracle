//! Standalone similarity computation helper functions.
//!
//! These functions compute specific similarity metrics for teleological vectors
//! and are used by the SimilarityComputer for various comparison modes.

use super::super::super::groups::GroupType;
use super::super::super::types::NUM_EMBEDDERS;
use super::super::super::vector::TeleologicalVector;

/// Compute topic profile cosine similarity.
pub fn compute_purpose_similarity(a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
    a.topic_profile.similarity(&b.topic_profile)
}

/// Compute cross-correlation cosine similarity.
pub fn compute_correlation_similarity(a: &TeleologicalVector, b: &TeleologicalVector) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (&av, &bv) in a.cross_correlations.iter().zip(b.cross_correlations.iter()) {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    }
}

/// Compute similarity for specific embedding pairs.
pub fn compute_specific_pairs_similarity(
    a: &TeleologicalVector,
    b: &TeleologicalVector,
    pairs: &[(usize, usize)],
) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }

    let mut sum_sim = 0.0f32;

    for &(i, j) in pairs {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let av = a.get_correlation(lo, hi);
        let bv = b.get_correlation(lo, hi);
        // Absolute difference similarity
        sum_sim += 1.0 - (av - bv).abs();
    }

    sum_sim / pairs.len() as f32
}

/// Compute similarity for specific embedding groups.
pub fn compute_specific_groups_similarity(
    a: &TeleologicalVector,
    b: &TeleologicalVector,
    groups: &[GroupType],
) -> f32 {
    if groups.is_empty() {
        return 0.0;
    }

    let mut sum_sim = 0.0f32;

    for &group in groups {
        let ga = a.group_alignments.get(group);
        let gb = b.group_alignments.get(group);
        sum_sim += 1.0 - (ga - gb).abs();
    }

    sum_sim / groups.len() as f32
}

/// Compute similarity for a single embedder's correlation pattern.
///
/// Compares all 12 cross-correlations that involve the specified embedder.
pub fn compute_single_embedder_pattern_similarity(
    a: &TeleologicalVector,
    b: &TeleologicalVector,
    embedder_idx: usize,
) -> f32 {
    assert!(
        embedder_idx < NUM_EMBEDDERS,
        "FAIL FAST: embedder index {} out of bounds",
        embedder_idx
    );

    let mut sum_sim = 0.0f32;
    let mut count = 0;

    // All pairs involving this embedder
    for other in 0..NUM_EMBEDDERS {
        if other == embedder_idx {
            continue;
        }

        let (lo, hi) = if embedder_idx < other {
            (embedder_idx, other)
        } else {
            (other, embedder_idx)
        };

        let av = a.get_correlation(lo, hi);
        let bv = b.get_correlation(lo, hi);
        sum_sim += 1.0 - (av - bv).abs();
        count += 1;
    }

    if count > 0 {
        sum_sim / count as f32
    } else {
        0.0
    }
}
