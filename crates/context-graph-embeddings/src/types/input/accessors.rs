//! Accessor methods, utilities, and Display implementation for ModelInput.
//!
//! This module extends ModelInput with:
//! - Type predicates (is_text, is_code, etc.)
//! - Content accessors (as_text, as_code, etc.)
//! - Utility methods (content_hash, byte_size)
//! - Display formatting

use xxhash_rust::xxh64::xxh64;

use super::{ImageFormat, ModelInput};

impl ModelInput {
    /// Compute xxHash64 of the content for cache keying.
    ///
    /// Hash includes all content bytes:
    /// - Text: UTF-8 bytes of content + instruction if present
    /// - Code: UTF-8 bytes of content + language
    /// - Image: raw bytes + format discriminant
    /// - Audio: raw bytes + sample_rate (little-endian) + channels
    ///
    /// # Returns
    /// 64-bit hash value. Deterministic for identical inputs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let input1 = ModelInput::text("Hello").unwrap();
    /// let input2 = ModelInput::text("Hello").unwrap();
    /// assert_eq!(input1.content_hash(), input2.content_hash());
    ///
    /// let input3 = ModelInput::text("World").unwrap();
    /// assert_ne!(input1.content_hash(), input3.content_hash());
    /// ```
    #[must_use]
    pub fn content_hash(&self) -> u64 {
        match self {
            Self::Text {
                content,
                instruction,
            } => {
                let mut data = content.as_bytes().to_vec();
                match instruction {
                    Some(inst) => {
                        // Discriminator byte 0x01 indicates Some (even for empty string)
                        data.push(0x01);
                        data.extend_from_slice(inst.as_bytes());
                    }
                    None => {
                        // Discriminator byte 0x00 indicates None
                        data.push(0x00);
                    }
                }
                xxh64(&data, 0)
            }
            Self::Code { content, language } => {
                let mut data = content.as_bytes().to_vec();
                data.extend_from_slice(language.as_bytes());
                xxh64(&data, 0)
            }
            Self::Image { bytes, format } => {
                let mut data = bytes.clone();
                data.push(*format as u8);
                xxh64(&data, 0)
            }
            Self::Audio {
                bytes,
                sample_rate,
                channels,
            } => {
                let mut data = bytes.clone();
                data.extend_from_slice(&sample_rate.to_le_bytes());
                data.push(*channels);
                xxh64(&data, 0)
            }
        }
    }

    /// Calculate total memory size in bytes.
    ///
    /// Includes heap allocations for strings and byte vectors.
    /// Used by MemoryTracker (M03-L02) for memory budget management.
    ///
    /// # Returns
    /// Approximate memory usage including:
    /// - String heap allocations (capacity, not just len for accuracy)
    /// - `Vec<u8>` heap allocations
    /// - Struct overhead is NOT included (stack-allocated)
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let input = ModelInput::text("Hello, world!").unwrap();
    /// let size = input.byte_size();
    /// assert!(size >= 13); // At least the string length
    /// ```
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Text {
                content,
                instruction,
            } => content.len() + instruction.as_ref().map_or(0, |s| s.len()),
            Self::Code { content, language } => content.len() + language.len(),
            Self::Image { bytes, .. } => bytes.len(),
            Self::Audio { bytes, .. } => bytes.len(),
        }
    }

    /// Returns true if this is a Text variant.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Returns true if this is a Code variant.
    #[must_use]
    pub const fn is_code(&self) -> bool {
        matches!(self, Self::Code { .. })
    }

    /// Returns true if this is an Image variant.
    #[must_use]
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    /// Returns true if this is an Audio variant.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }

    /// Get text content if this is a Text variant.
    ///
    /// # Returns
    /// `Some((content, instruction))` where instruction is `Some(&str)` if set,
    /// or `None` if this is not a Text variant.
    #[must_use]
    pub fn as_text(&self) -> Option<(&str, Option<&str>)> {
        match self {
            Self::Text {
                content,
                instruction,
            } => Some((content.as_str(), instruction.as_deref())),
            _ => None,
        }
    }

    /// Get code content if this is a Code variant.
    ///
    /// # Returns
    /// `Some((content, language))` or `None` if not a Code variant.
    #[must_use]
    pub fn as_code(&self) -> Option<(&str, &str)> {
        match self {
            Self::Code { content, language } => Some((content.as_str(), language.as_str())),
            _ => None,
        }
    }

    /// Get image bytes if this is an Image variant.
    ///
    /// # Returns
    /// `Some((bytes, format))` or `None` if not an Image variant.
    #[must_use]
    pub fn as_image(&self) -> Option<(&[u8], ImageFormat)> {
        match self {
            Self::Image { bytes, format } => Some((bytes.as_slice(), *format)),
            _ => None,
        }
    }

    /// Get audio bytes if this is an Audio variant.
    ///
    /// # Returns
    /// `Some((bytes, sample_rate, channels))` or `None` if not an Audio variant.
    #[must_use]
    pub fn as_audio(&self) -> Option<(&[u8], u32, u8)> {
        match self {
            Self::Audio {
                bytes,
                sample_rate,
                channels,
            } => Some((bytes.as_slice(), *sample_rate, *channels)),
            _ => None,
        }
    }
}

impl std::fmt::Display for ModelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text {
                content,
                instruction,
            } => {
                let preview: String = content.chars().take(50).collect();
                let suffix = if content.len() > 50 { "..." } else { "" };
                match instruction {
                    Some(inst) => write!(f, "Text[{}: {}{}]", inst, preview, suffix),
                    None => write!(f, "Text[{}{}]", preview, suffix),
                }
            }
            Self::Code { content, language } => {
                let preview: String = content.chars().take(30).collect();
                let suffix = if content.len() > 30 { "..." } else { "" };
                write!(f, "Code[{}: {}{}]", language, preview, suffix)
            }
            Self::Image { bytes, format } => {
                write!(f, "Image[{}: {} bytes]", format, bytes.len())
            }
            Self::Audio {
                bytes,
                sample_rate,
                channels,
            } => {
                let ch_str = if *channels == 1 { "mono" } else { "stereo" };
                write!(
                    f,
                    "Audio[{}Hz {}: {} bytes]",
                    sample_rate,
                    ch_str,
                    bytes.len()
                )
            }
        }
    }
}
