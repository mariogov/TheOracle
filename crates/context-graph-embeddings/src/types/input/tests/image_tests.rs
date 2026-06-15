//! Image construction and ImageFormat tests for ModelInput.

use crate::error::EmbeddingError;
use crate::types::input::{ImageFormat, ModelInput};

// ============================================================
// IMAGE CONSTRUCTION TESTS (3 tests)
// ============================================================

#[test]
fn test_image_with_valid_bytes_succeeds() {
    let bytes = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic
    let input = ModelInput::image(bytes.clone(), ImageFormat::Png);
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_image());
    let (img_bytes, format) = input.as_image().unwrap();
    assert_eq!(img_bytes, bytes.as_slice());
    assert_eq!(format, ImageFormat::Png);
}

#[test]
fn test_image_with_empty_bytes_returns_empty_input() {
    let result = ModelInput::image(vec![], ImageFormat::Png);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_image_stores_format_correctly() {
    let bytes = vec![1, 2, 3, 4];

    for format in [
        ImageFormat::Png,
        ImageFormat::Jpeg,
        ImageFormat::WebP,
        ImageFormat::Gif,
    ] {
        let input = ModelInput::image(bytes.clone(), format).unwrap();
        let (_, stored_format) = input.as_image().unwrap();
        assert_eq!(stored_format, format);
    }
}

// ============================================================
// IMAGE FORMAT TESTS (8 tests)
// ============================================================

#[test]
fn test_image_format_mime_type() {
    assert_eq!(ImageFormat::Png.mime_type(), "image/png");
    assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
    assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    assert_eq!(ImageFormat::Gif.mime_type(), "image/gif");
}

#[test]
fn test_image_format_extension() {
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
    assert_eq!(ImageFormat::WebP.extension(), "webp");
    assert_eq!(ImageFormat::Gif.extension(), "gif");
}

#[test]
fn test_image_format_detect_png() {
    let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    assert_eq!(ImageFormat::detect(&png_bytes), Some(ImageFormat::Png));
}

#[test]
fn test_image_format_detect_jpeg() {
    let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
    assert_eq!(ImageFormat::detect(&jpeg_bytes), Some(ImageFormat::Jpeg));
}

#[test]
fn test_image_format_detect_gif() {
    let gif_bytes = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a
    assert_eq!(ImageFormat::detect(&gif_bytes), Some(ImageFormat::Gif));
}

#[test]
fn test_image_format_detect_webp() {
    let webp_bytes = [
        0x52, 0x49, 0x46, 0x46, // RIFF
        0x00, 0x00, 0x00, 0x00, // size (ignored)
        0x57, 0x45, 0x42, 0x50, // WEBP
    ];
    assert_eq!(ImageFormat::detect(&webp_bytes), Some(ImageFormat::WebP));
}

#[test]
fn test_image_format_detect_unknown() {
    let unknown = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
    assert_eq!(ImageFormat::detect(&unknown), None);
}

#[test]
fn test_image_format_detect_too_short() {
    let short = [0x89, 0x50, 0x4E]; // Only 3 bytes, need 4 for PNG
    assert_eq!(ImageFormat::detect(&short), None);
}
