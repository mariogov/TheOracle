//! Content hash and byte size tests for ModelInput.

use crate::types::input::{ImageFormat, ModelInput};

// ============================================================
// CONTENT HASH TESTS (4 tests)
// ============================================================

#[test]
fn test_content_hash_same_for_identical_text_inputs() {
    let input1 = ModelInput::text("test content").unwrap();
    let hash1 = input1.content_hash();
    println!("BEFORE: First hash = {}", hash1);

    let input2 = ModelInput::text("test content").unwrap();
    let hash2 = input2.content_hash();
    println!("AFTER: Second hash = {}", hash2);

    assert_eq!(
        hash1, hash2,
        "Identical inputs must produce identical hashes"
    );
}

#[test]
fn test_content_hash_different_for_different_content() {
    let input1 = ModelInput::text("Hello").unwrap();
    let input2 = ModelInput::text("World").unwrap();
    let input3 = ModelInput::text("hello").unwrap(); // case sensitive

    assert_ne!(input1.content_hash(), input2.content_hash());
    assert_ne!(input1.content_hash(), input3.content_hash());
}

#[test]
fn test_content_hash_includes_instruction_in_hash() {
    let without_inst = ModelInput::text("content").unwrap();
    let with_inst = ModelInput::text_with_instruction("content", "query:").unwrap();
    let with_empty_inst = ModelInput::text_with_instruction("content", "").unwrap();

    // All three should have different hashes
    assert_ne!(without_inst.content_hash(), with_inst.content_hash());
    assert_ne!(without_inst.content_hash(), with_empty_inst.content_hash());
    assert_ne!(with_inst.content_hash(), with_empty_inst.content_hash());
}

#[test]
fn test_content_hash_includes_all_fields_for_each_variant() {
    // Code: different language = different hash
    let code1 = ModelInput::code("code", "rust").unwrap();
    let code2 = ModelInput::code("code", "python").unwrap();
    assert_ne!(code1.content_hash(), code2.content_hash());

    // Image: different format = different hash
    let img1 = ModelInput::image(vec![1, 2, 3], ImageFormat::Png).unwrap();
    let img2 = ModelInput::image(vec![1, 2, 3], ImageFormat::Jpeg).unwrap();
    assert_ne!(img1.content_hash(), img2.content_hash());

    // Audio: different sample_rate = different hash
    let audio1 = ModelInput::audio(vec![1, 2, 3], 16000, 1).unwrap();
    let audio2 = ModelInput::audio(vec![1, 2, 3], 44100, 1).unwrap();
    assert_ne!(audio1.content_hash(), audio2.content_hash());

    // Audio: different channels = different hash
    let audio3 = ModelInput::audio(vec![1, 2, 3], 16000, 2).unwrap();
    assert_ne!(audio1.content_hash(), audio3.content_hash());
}

// ============================================================
// BYTE SIZE TESTS (3 tests)
// ============================================================

#[test]
fn test_byte_size_returns_correct_size_for_text() {
    let content = "Hello, world!";
    let input = ModelInput::text(content).unwrap();
    let size = input.byte_size();
    assert_eq!(size, content.len());

    // With instruction
    let input_with_inst = ModelInput::text_with_instruction("Hello", "query:").unwrap();
    assert_eq!(input_with_inst.byte_size(), "Hello".len() + "query:".len());
}

#[test]
fn test_byte_size_returns_correct_size_for_image_bytes() {
    let bytes = vec![0u8; 1000];
    let input = ModelInput::image(bytes.clone(), ImageFormat::Png).unwrap();
    assert_eq!(input.byte_size(), 1000);
}

#[test]
fn test_byte_size_includes_string_heap_allocation() {
    let code_content = "fn main() {}";
    let language = "rust";
    let input = ModelInput::code(code_content, language).unwrap();

    // Should include both strings
    assert_eq!(input.byte_size(), code_content.len() + language.len());
}

// ============================================================
// EDGE CASE: HASH DETERMINISM ACROSS CALLS
// ============================================================

#[test]
fn test_hash_determinism_multiple_calls() {
    let input = ModelInput::text("deterministic content").unwrap();

    let hash1 = input.content_hash();
    let hash2 = input.content_hash();
    let hash3 = input.content_hash();

    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);
    println!(
        "Hash determinism verified: {} == {} == {}",
        hash1, hash2, hash3
    );
}

// ============================================================
// EDGE CASE: LARGE INPUTS
// ============================================================

#[test]
fn test_large_text_input() {
    let large_content = "x".repeat(100_000);
    let input = ModelInput::text(&large_content).unwrap();
    assert_eq!(input.byte_size(), 100_000);

    // Hash should still work
    let hash = input.content_hash();
    assert_ne!(hash, 0);
}

#[test]
fn test_large_image_input() {
    let large_bytes = vec![0xFFu8; 1_000_000]; // 1MB
    let input = ModelInput::image(large_bytes, ImageFormat::Jpeg).unwrap();
    assert_eq!(input.byte_size(), 1_000_000);

    // Hash should still work
    let hash = input.content_hash();
    assert_ne!(hash, 0);
}
