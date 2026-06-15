//! Display and Serde serialization tests for ModelInput.

use crate::types::input::{ImageFormat, ModelInput};

// ============================================================
// DISPLAY AND SERDE TESTS (2 tests)
// ============================================================

#[test]
fn test_display_formatting() {
    let text = ModelInput::text("Hello, world!").unwrap();
    let display = format!("{}", text);
    assert!(display.contains("Text"));
    assert!(display.contains("Hello"));

    let code = ModelInput::code("fn main() {}", "rust").unwrap();
    let display = format!("{}", code);
    assert!(display.contains("Code"));
    assert!(display.contains("rust"));

    let image = ModelInput::image(vec![1, 2, 3], ImageFormat::Png).unwrap();
    let display = format!("{}", image);
    assert!(display.contains("Image"));
    assert!(display.contains("PNG"));
    assert!(display.contains("3 bytes"));

    let audio = ModelInput::audio(vec![1, 2, 3, 4], 44100, 2).unwrap();
    let display = format!("{}", audio);
    assert!(display.contains("Audio"));
    assert!(display.contains("44100"));
    assert!(display.contains("stereo"));
}

#[test]
fn test_serde_round_trip() {
    let text = ModelInput::text("Hello").unwrap();
    let json = serde_json::to_string(&text).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(text, recovered);

    let code = ModelInput::code("fn main() {}", "rust").unwrap();
    let json = serde_json::to_string(&code).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(code, recovered);

    let image = ModelInput::image(vec![1, 2, 3], ImageFormat::Jpeg).unwrap();
    let json = serde_json::to_string(&image).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(image, recovered);

    let audio = ModelInput::audio(vec![1, 2, 3], 16000, 1).unwrap();
    let json = serde_json::to_string(&audio).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(audio, recovered);
}
