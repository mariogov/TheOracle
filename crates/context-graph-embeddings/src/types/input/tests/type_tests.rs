//! Type predicate and accessor tests for ModelInput.

use crate::types::input::{ImageFormat, ModelInput};

// ============================================================
// TYPE PREDICATE TESTS (4 tests)
// ============================================================

#[test]
fn test_is_text_returns_true_only_for_text_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(text.is_text());
    assert!(!code.is_text());
    assert!(!image.is_text());
    assert!(!audio.is_text());
}

#[test]
fn test_is_code_returns_true_only_for_code_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_code());
    assert!(code.is_code());
    assert!(!image.is_code());
    assert!(!audio.is_code());
}

#[test]
fn test_is_image_returns_true_only_for_image_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_image());
    assert!(!code.is_image());
    assert!(image.is_image());
    assert!(!audio.is_image());
}

#[test]
fn test_is_audio_returns_true_only_for_audio_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_audio());
    assert!(!code.is_audio());
    assert!(!image.is_audio());
    assert!(audio.is_audio());
}

// ============================================================
// ACCESSOR TESTS (4 tests)
// ============================================================

#[test]
fn test_as_text_returns_none_for_non_text() {
    let code = ModelInput::code("code", "rust").unwrap();
    assert!(code.as_text().is_none());

    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    assert!(image.as_text().is_none());
}

#[test]
fn test_as_code_returns_none_for_non_code() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_code().is_none());

    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();
    assert!(audio.as_code().is_none());
}

#[test]
fn test_as_image_returns_none_for_non_image() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_image().is_none());

    let code = ModelInput::code("code", "rust").unwrap();
    assert!(code.as_image().is_none());
}

#[test]
fn test_as_audio_returns_none_for_non_audio() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_audio().is_none());

    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    assert!(image.as_audio().is_none());
}
