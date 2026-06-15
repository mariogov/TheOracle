//! Manual verification tests for RRF edge cases per TASK-LOGIC-011.
//!
//! These tests verify:
//! 1. Empty rankings
//! 2. Document in only one ranking
//! 3. Same document at same rank in all spaces

use context_graph_core::retrieval::AggregationStrategy;
use uuid::Uuid;

/// Edge Case 1: Empty Rankings
/// Input: ranked_lists = []
/// Expected: Returns empty HashMap
#[test]
fn test_rrf_edge_case_empty_rankings() {
    println!("\n=== EDGE CASE 1: Empty Rankings ===");
    println!("INPUT: ranked_lists = []");

    let ranked_lists: Vec<(usize, Vec<Uuid>)> = vec![];

    println!("BEFORE STATE: No documents");
    let scores = AggregationStrategy::aggregate_rrf(&ranked_lists, 60.0);

    println!("AFTER STATE: HashMap with {} entries", scores.len());
    assert_eq!(
        scores.len(),
        0,
        "Empty rankings should return empty HashMap"
    );

    println!("[VERIFIED] Empty rankings returns empty HashMap");
}

/// Edge Case 2: Document in Only One Ranking
/// Input: Doc A in space 0 at rank 0, not in spaces 1-12
/// Expected: RRF(A) = 1/61 (only one contribution)
#[test]
fn test_rrf_edge_case_single_ranking() {
    println!("\n=== EDGE CASE 2: Document in Only One Ranking ===");

    let doc_a = Uuid::new_v4();
    println!("INPUT: Doc {} in space 0 at rank 0 only", doc_a);

    let ranked_lists = vec![
        (0, vec![doc_a]), // Only in space 0
    ];

    println!("BEFORE STATE: Doc absent from aggregate");
    let scores = AggregationStrategy::aggregate_rrf(&ranked_lists, 60.0);

    let score_a = scores.get(&doc_a).unwrap();
    let expected = 1.0 / 61.0; // 1/(60+0+1)

    println!("AFTER STATE: Doc has score {:.10}", score_a);
    println!("EXPECTED: {:.10} (1/61)", expected);

    assert!(
        (score_a - expected).abs() < 0.0001,
        "RRF(A) should be 1/61, got {}",
        score_a
    );

    println!("[VERIFIED] Single ranking contribution = 1/61 = 0.01639344...");
}

/// Edge Case 3: Same Document at Same Rank in All 13 Spaces
/// Input: Doc A at rank 0 in all 13 spaces
/// Expected: RRF(A) = 13 × (1/61) = 0.21311...
#[test]
fn test_rrf_edge_case_all_spaces() {
    println!("\n=== EDGE CASE 3: Document in All 13 Spaces at Rank 0 ===");

    let doc_a = Uuid::new_v4();
    println!("INPUT: Doc {} at rank 0 in all 13 spaces", doc_a);

    // Create rankings for all 13 embedders
    let ranked_lists: Vec<(usize, Vec<Uuid>)> = (0..13).map(|space| (space, vec![doc_a])).collect();

    println!("BEFORE STATE: Doc absent from aggregate");
    let scores = AggregationStrategy::aggregate_rrf(&ranked_lists, 60.0);

    let score_a = scores.get(&doc_a).unwrap();
    let expected = 13.0 / 61.0; // 13 × 1/(60+0+1)

    println!("AFTER STATE: Doc has score {:.10}", score_a);
    println!("EXPECTED: {:.10} (13 × 1/61)", expected);

    assert!(
        (score_a - expected).abs() < 0.0001,
        "RRF(A) should be 13/61, got {}",
        score_a
    );

    println!("[VERIFIED] 13-space ranking contribution = 13/61 = 0.21311475...");
}

/// Verify exact RRF formula: 1/(k + rank + 1)
#[test]
fn test_rrf_formula_verification() {
    println!("\n=== FORMULA VERIFICATION ===");

    // Test with k=60 per Constitution
    let k = 60.0;

    // Rank 0: 1/(60+0+1) = 1/61
    let rank0_contrib = AggregationStrategy::rrf_contribution(0, k);
    let rank0_expected = 1.0 / 61.0;
    assert!((rank0_contrib - rank0_expected).abs() < 0.0001);
    println!(
        "Rank 0: expected={:.10}, actual={:.10}",
        rank0_expected, rank0_contrib
    );

    // Rank 5: 1/(60+5+1) = 1/66
    let rank5_contrib = AggregationStrategy::rrf_contribution(5, k);
    let rank5_expected = 1.0 / 66.0;
    assert!((rank5_contrib - rank5_expected).abs() < 0.0001);
    println!(
        "Rank 5: expected={:.10}, actual={:.10}",
        rank5_expected, rank5_contrib
    );

    // Rank 99: 1/(60+99+1) = 1/160
    let rank99_contrib = AggregationStrategy::rrf_contribution(99, k);
    let rank99_expected = 1.0 / 160.0;
    assert!((rank99_contrib - rank99_expected).abs() < 0.0001);
    println!(
        "Rank 99: expected={:.10}, actual={:.10}",
        rank99_expected, rank99_contrib
    );

    println!("[VERIFIED] RRF formula: 1/(k + rank + 1) with k=60");
}

/// Verify multi-space aggregation with exact expected values
#[test]
fn test_rrf_multispace_exact_values() {
    println!("\n=== MULTI-SPACE EXACT VALUES ===");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    // id1 at ranks [0, 1, 0] across 3 spaces
    // RRF = 1/61 + 1/62 + 1/61 = 0.04888...
    let ranked_lists = vec![
        (0, vec![id1, id2]), // id1=rank0, id2=rank1
        (1, vec![id2, id1]), // id2=rank0, id1=rank1
        (2, vec![id1, id2]), // id1=rank0, id2=rank1
    ];

    let scores = AggregationStrategy::aggregate_rrf(&ranked_lists, 60.0);

    let score1 = scores.get(&id1).unwrap();
    let expected1 = 1.0 / 61.0 + 1.0 / 62.0 + 1.0 / 61.0;

    println!("id1 at ranks [0, 1, 0]:");
    println!("  Expected: {:.10} (1/61 + 1/62 + 1/61)", expected1);
    println!("  Actual:   {:.10}", score1);

    assert!(
        (score1 - expected1).abs() < 0.0001,
        "id1 RRF mismatch: expected {}, got {}",
        expected1,
        score1
    );

    let score2 = scores.get(&id2).unwrap();
    let expected2 = 1.0 / 62.0 + 1.0 / 61.0 + 1.0 / 62.0;

    println!("id2 at ranks [1, 0, 1]:");
    println!("  Expected: {:.10} (1/62 + 1/61 + 1/62)", expected2);
    println!("  Actual:   {:.10}", score2);

    assert!(
        (score2 - expected2).abs() < 0.0001,
        "id2 RRF mismatch: expected {}, got {}",
        expected2,
        score2
    );

    println!("[VERIFIED] Multi-space aggregation matches exact expected values");
}
