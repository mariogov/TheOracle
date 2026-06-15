//! Content storage tests.
//!
//! TASK-CONTENT: Content Storage Tests (Happy Path with Real Data)
//!
//! CRITICAL: Uses #[tokio::test] to prevent zombie runtime threads.
//! DO NOT use tokio::runtime::Runtime::new() in tests.

use super::helpers::{create_fingerprint_for_content, create_real_fingerprint};
use crate::teleological::RocksDbTeleologicalStore;
use context_graph_core::traits::TeleologicalMemoryStore;
use uuid::Uuid;

#[tokio::test]
async fn test_content_store_retrieve_happy_path() {
    println!("=== TEST: store_content / get_content happy path (TASK-CONTENT-007/008) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Define content first (hash must be computed before fingerprint creation)
    let content = "This is test content for happy path validation. \
                   It contains multiple sentences and special characters: \
                   @#$%^&*() äöü 日本語 🦀";

    // Create fingerprint with correct content hash
    let fingerprint = create_fingerprint_for_content(content);
    let fingerprint_id = fingerprint.id;

    // Store the fingerprint
    let stored_id = store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");
    assert_eq!(stored_id, fingerprint_id);

    println!("BEFORE: Storing content ({} bytes)", content.len());

    store
        .store_content(fingerprint_id, content)
        .await
        .expect("Should store content");

    // Retrieve and verify
    let retrieved = store
        .get_content(fingerprint_id)
        .await
        .expect("Should retrieve content")
        .expect("Content should exist");

    println!("AFTER: Retrieved content ({} bytes)", retrieved.len());

    assert_eq!(content, retrieved, "Content should match exactly");

    println!("RESULT: PASS - Content store/retrieve happy path successful");
}

#[tokio::test]
async fn test_content_batch_retrieval() {
    println!("=== TEST: get_content_batch happy path (TASK-CONTENT-008) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Store multiple fingerprints with content
    let mut ids = Vec::new();
    let mut expected_contents = Vec::new();

    for i in 0..5 {
        let content = format!(
            "Batch content item {} with unique text: {}",
            i,
            Uuid::new_v4()
        );
        let fingerprint = create_fingerprint_for_content(&content);
        let id = fingerprint.id;

        store
            .store(fingerprint)
            .await
            .expect("Should store fingerprint");
        store
            .store_content(id, &content)
            .await
            .expect("Should store content");

        ids.push(id);
        expected_contents.push(content);
    }

    // Add one ID that has no content (use any hash since we won't store content)
    let fp_no_content = create_real_fingerprint();
    store
        .store(fp_no_content.clone())
        .await
        .expect("Should store fingerprint without content");
    ids.push(fp_no_content.id);
    expected_contents.push(String::new()); // Placeholder for None

    // Batch retrieve
    let results = store
        .get_content_batch(&ids)
        .await
        .expect("Should batch retrieve content");

    assert_eq!(
        results.len(),
        ids.len(),
        "Result count should match ID count"
    );

    // Verify first 5 have content
    for i in 0..5 {
        assert!(results[i].is_some(), "Content {} should exist", i);
        assert_eq!(
            results[i].as_ref().unwrap(),
            &expected_contents[i],
            "Content {} should match",
            i
        );
    }

    // Last one should be None
    assert!(
        results[5].is_none(),
        "Fingerprint without content should return None"
    );

    println!("RESULT: PASS - Batch content retrieval successful (5 with content, 1 without)");
}

#[tokio::test]
async fn test_content_delete_cascade() {
    println!("=== TEST: delete cascade removes content (TASK-CONTENT-009) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Create content and fingerprint with matching hash
    let content = "Content that will be deleted via cascade";
    let fingerprint = create_fingerprint_for_content(content);
    let id = fingerprint.id;

    store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");
    store
        .store_content(id, content)
        .await
        .expect("Should store content");

    // Verify content exists
    let before = store
        .get_content(id)
        .await
        .expect("Should get content")
        .expect("Content should exist before delete");
    assert_eq!(before, content);

    println!("BEFORE: Content exists ({} bytes)", before.len());

    // Hard delete fingerprint (should cascade to content)
    let deleted = store.delete(id, false).await.expect("Should delete");
    assert!(deleted, "Delete should return true");

    // Content should be gone
    let after = store.get_content(id).await.expect("Should query content");
    assert!(after.is_none(), "Content should be deleted via cascade");

    println!("AFTER: Content deleted via cascade");
    println!("RESULT: PASS - Delete cascade removes content");
}

#[tokio::test]
async fn test_content_soft_delete_preserves_content() {
    println!("=== TEST: soft delete preserves content (TASK-CONTENT-009) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Create content and fingerprint with matching hash
    let content = "Content preserved during soft delete";
    let fingerprint = create_fingerprint_for_content(content);
    let id = fingerprint.id;

    store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");
    store
        .store_content(id, content)
        .await
        .expect("Should store content");

    // Soft delete fingerprint
    let deleted = store.delete(id, true).await.expect("Should soft delete");
    assert!(deleted, "Soft delete should return true");

    // Content should still exist (only hard delete cascades)
    let after = store.get_content(id).await.expect("Should query content");
    assert!(
        after.is_some(),
        "Content should be preserved after soft delete"
    );
    assert_eq!(after.unwrap(), content);

    println!("RESULT: PASS - Soft delete preserves content");
}

#[tokio::test]
async fn test_content_max_size_boundary() {
    println!("=== TEST: content max size boundary (1MB) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Test at exactly 1MB (maximum allowed)
    let max_content = "x".repeat(1_048_576);
    assert_eq!(max_content.len(), 1_048_576, "Should be exactly 1MB");

    // Create fingerprint with correct hash for max content
    let fingerprint = create_fingerprint_for_content(&max_content);
    let id = fingerprint.id;
    store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");

    let result = store.store_content(id, &max_content).await;
    assert!(result.is_ok(), "1MB content should be accepted");

    // Verify retrieval
    let retrieved = store
        .get_content(id)
        .await
        .expect("Should retrieve")
        .expect("Should exist");
    assert_eq!(retrieved.len(), 1_048_576, "Retrieved size should match");

    println!("RESULT: PASS - 1MB content (max boundary) accepted and retrieved");
}

#[tokio::test]
async fn test_content_over_max_size_rejected() {
    println!("=== TEST: content over max size rejected ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Test over 1MB (should fail at size check before hash check)
    let over_max_content = "x".repeat(1_048_577);
    assert_eq!(
        over_max_content.len(),
        1_048_577,
        "Should be 1 byte over 1MB"
    );

    // Create fingerprint with correct hash (not that it matters - size check comes first)
    let fingerprint = create_fingerprint_for_content(&over_max_content);
    let id = fingerprint.id;
    store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");

    let result = store.store_content(id, &over_max_content).await;
    assert!(result.is_err(), "Content over 1MB should be rejected");

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("exceeds maximum") || err_msg.contains("1048576"),
        "Error should mention size limit: {}",
        err_msg
    );

    println!("RESULT: PASS - Content over 1MB correctly rejected");
}

#[tokio::test]
async fn test_content_nonexistent_fingerprint() {
    println!("=== TEST: get_content for non-existent fingerprint returns None ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    let nonexistent_id = Uuid::new_v4();
    let result = store
        .get_content(nonexistent_id)
        .await
        .expect("Should not error, just return None");

    assert!(
        result.is_none(),
        "Non-existent fingerprint should return None"
    );

    println!("RESULT: PASS - Non-existent fingerprint returns None (not error)");
}

#[tokio::test]
async fn test_content_empty_string_allowed() {
    println!("=== TEST: empty string content ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    // Create fingerprint with hash of empty string
    let fingerprint = create_fingerprint_for_content("");
    let id = fingerprint.id;
    store
        .store(fingerprint)
        .await
        .expect("Should store fingerprint");

    // Store empty string - allowed (size 0 < max 1MB, and hash matches)
    let result = store.store_content(id, "").await;
    assert!(
        result.is_ok(),
        "Empty content should be allowed (size validation passes)"
    );

    // Verify retrieval
    let retrieved = store
        .get_content(id)
        .await
        .expect("Should retrieve")
        .expect("Should exist");
    assert_eq!(retrieved, "", "Empty string should round-trip correctly");

    println!("RESULT: PASS - Empty string allowed and retrieved correctly");
}

#[tokio::test]
async fn test_content_unicode_comprehensive() {
    println!("=== TEST: Unicode content comprehensive ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");

    let test_cases = vec![
        "Simple ASCII",
        "Latin-1: café résumé naïve",
        "Greek: Ελληνικά",
        "Russian: Русский",
        "Chinese: 中文测试",
        "Japanese: 日本語テスト",
        "Korean: 한국어 테스트",
        "Arabic: العربية",
        "Hebrew: עברית",
        "Emoji: 🦀🚀💾🔥",
        "Math: ∑∫∂∇∏√",
        "Mixed: Hello 世界 🌍 Привет",
    ];

    for (i, content) in test_cases.iter().enumerate() {
        // Create fingerprint with correct hash for this content
        let fingerprint = create_fingerprint_for_content(content);
        let id = fingerprint.id;
        store
            .store(fingerprint)
            .await
            .expect("Should store fingerprint");

        store
            .store_content(id, content)
            .await
            .unwrap_or_else(|_| panic!("Should store Unicode content case {}", i));

        let retrieved = store
            .get_content(id)
            .await
            .expect("Should retrieve")
            .expect("Should exist");

        assert_eq!(
            *content, retrieved,
            "Unicode case {} should round-trip correctly",
            i
        );
    }

    println!(
        "RESULT: PASS - All {} Unicode test cases passed",
        test_cases.len()
    );
}
