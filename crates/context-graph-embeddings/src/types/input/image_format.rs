//! Image format detection and metadata.
//!
//! Provides `ImageFormat` enum for identifying image types via magic bytes.

use serde::{Deserialize, Serialize};

/// Image format for binary image inputs.
///
/// Supports detection via magic bytes for automatic format identification.
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::types::ImageFormat;
///
/// // Detect PNG from magic bytes
/// let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
/// assert_eq!(ImageFormat::detect(&png_bytes), Some(ImageFormat::Png));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ImageFormat {
    /// PNG format (magic: 0x89 0x50 0x4E 0x47)
    Png = 0,
    /// JPEG format (magic: 0xFF 0xD8 0xFF)
    Jpeg = 1,
    /// WebP format (magic: RIFF....WEBP)
    WebP = 2,
    /// GIF format (magic: GIF8)
    Gif = 3,
}

impl ImageFormat {
    /// Get MIME type for this image format.
    ///
    /// # Returns
    /// Standard MIME type string for HTTP Content-Type headers.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::WebP => "image/webp",
            Self::Gif => "image/gif",
        }
    }

    /// Get file extension for this image format.
    ///
    /// # Returns
    /// Lowercase extension without leading dot.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::WebP => "webp",
            Self::Gif => "gif",
        }
    }

    /// Try to detect format from magic bytes.
    ///
    /// Uses the first bytes of file content to identify format:
    /// - PNG: `\x89PNG` (4 bytes)
    /// - JPEG: `\xFF\xD8\xFF` (3 bytes)
    /// - GIF: `GIF8` (4 bytes)
    /// - WebP: `RIFF....WEBP` (12 bytes)
    ///
    /// # Arguments
    /// * `bytes` - Raw image bytes to analyze
    ///
    /// # Returns
    /// `Some(format)` if detected, `None` if format unknown or bytes too short.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ImageFormat;
    ///
    /// // JPEG magic bytes
    /// let jpeg_start = [0xFF, 0xD8, 0xFF, 0xE0];
    /// assert_eq!(ImageFormat::detect(&jpeg_start), Some(ImageFormat::Jpeg));
    ///
    /// // Unknown format
    /// let unknown = [0x00, 0x01, 0x02, 0x03];
    /// assert_eq!(ImageFormat::detect(&unknown), None);
    /// ```
    #[must_use]
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }

        // PNG magic: \x89PNG
        if bytes[0..4] == [0x89, 0x50, 0x4E, 0x47] {
            return Some(Self::Png);
        }

        // JPEG magic: \xFF\xD8\xFF (check first 3 bytes)
        if bytes[0..3] == [0xFF, 0xD8, 0xFF] {
            return Some(Self::Jpeg);
        }

        // GIF magic: GIF8 (GIF87a or GIF89a)
        if bytes[0..4] == [0x47, 0x49, 0x46, 0x38] {
            return Some(Self::Gif);
        }

        // WebP magic: RIFF....WEBP (requires 12 bytes)
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            return Some(Self::WebP);
        }

        None
    }
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension().to_uppercase())
    }
}
