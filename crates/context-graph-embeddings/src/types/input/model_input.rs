//! Multi-modal input types for the embedding pipeline.
//!
//! `ModelInput` provides a unified interface for passing different types of content
//! to the embedding models, allowing each model to handle inputs it supports.

use crate::error::{EmbeddingError, EmbeddingResult};
use serde::{Deserialize, Serialize};

use super::ImageFormat;

/// Multi-modal input for embedding models.
///
/// Each variant carries the data needed for that input type:
/// - Text: content string with optional instruction prefix (for e5-style models)
/// - Code: source code with language identifier
/// - Image: raw bytes with format information
/// - Audio: raw bytes with sample rate and channel count
///
/// # Validation
///
/// All constructors validate inputs and return `EmbeddingError::EmptyInput` for:
/// - Empty content/bytes
/// - Invalid parameters (e.g., sample_rate=0, channels not 1 or 2)
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::types::ModelInput;
///
/// // Create text input
/// let text_input = ModelInput::text("Hello, world!").unwrap();
/// assert!(text_input.is_text());
///
/// // Create code input with language
/// let code_input = ModelInput::code("fn main() {}", "rust").unwrap();
/// assert!(code_input.is_code());
///
/// // Hash for cache key
/// let hash = text_input.content_hash();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelInput {
    /// Text content with optional instruction prefix.
    /// Instruction is prepended for e5-style models (e.g., "query: " or "passage: ").
    Text {
        /// The actual text content to embed.
        content: String,
        /// Optional instruction prefix for e5-style models.
        instruction: Option<String>,
    },
    /// Source code with programming language identifier.
    /// Language should be lowercase (e.g., "rust", "python", "javascript").
    Code {
        /// The source code content.
        content: String,
        /// Programming language identifier (lowercase).
        language: String,
    },
    /// Image bytes with format information.
    /// Bytes must be valid encoded image data (not raw pixels).
    Image {
        /// Raw encoded image bytes (PNG, JPEG, WebP, or GIF).
        bytes: Vec<u8>,
        /// Image format for proper decoding.
        format: ImageFormat,
    },
    /// Audio bytes with sample metadata.
    /// For future audio embedding models.
    Audio {
        /// Raw audio bytes (PCM or encoded).
        bytes: Vec<u8>,
        /// Sample rate in Hz (e.g., 16000, 44100).
        sample_rate: u32,
        /// Number of channels: 1 = mono, 2 = stereo.
        channels: u8,
    },
}

impl ModelInput {
    /// Create a text input.
    ///
    /// # Arguments
    /// * `content` - Text content to embed (must not be empty)
    ///
    /// # Errors
    /// Returns `EmbeddingError::EmptyInput` if content is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let input = ModelInput::text("Hello, world!").unwrap();
    /// assert!(input.is_text());
    /// ```
    pub fn text(content: impl Into<String>) -> EmbeddingResult<Self> {
        let content = content.into();
        if content.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        Ok(Self::Text {
            content,
            instruction: None,
        })
    }

    /// Create a text input with instruction prefix.
    ///
    /// The instruction is prepended for e5-style models (e.g., "query: " or "passage: ").
    /// This helps the model understand the semantic role of the text.
    ///
    /// # Arguments
    /// * `content` - Text content to embed (must not be empty)
    /// * `instruction` - Instruction prefix (e.g., "query:", "passage:", "document:")
    ///
    /// # Errors
    /// Returns `EmbeddingError::EmptyInput` if content is empty.
    /// Note: Empty instruction is allowed (will be stored as Some("")).
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let query = ModelInput::text_with_instruction(
    ///     "What is Rust?",
    ///     "query:"
    /// ).unwrap();
    /// ```
    pub fn text_with_instruction(
        content: impl Into<String>,
        instruction: impl Into<String>,
    ) -> EmbeddingResult<Self> {
        let content = content.into();
        if content.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        Ok(Self::Text {
            content,
            instruction: Some(instruction.into()),
        })
    }

    /// Create a code input.
    ///
    /// # Arguments
    /// * `content` - Source code content (must not be empty)
    /// * `language` - Programming language identifier (must not be empty)
    ///
    /// # Errors
    /// Returns `EmbeddingError::EmptyInput` if content or language is empty.
    ///
    /// # Supported Languages
    ///
    /// Common language identifiers (validation not enforced):
    /// - rust, python, javascript, typescript
    /// - java, kotlin, scala, go, c, cpp, csharp
    /// - ruby, php, swift, sql, html, css
    /// - json, yaml, toml, bash, powershell
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let code = ModelInput::code(
    ///     "fn main() { println!(\"Hello\"); }",
    ///     "rust"
    /// ).unwrap();
    /// ```
    pub fn code(content: impl Into<String>, language: impl Into<String>) -> EmbeddingResult<Self> {
        let content = content.into();
        let language = language.into();

        if content.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        if language.is_empty() {
            return Err(EmbeddingError::ConfigError {
                message: "Code language cannot be empty".to_string(),
            });
        }

        Ok(Self::Code { content, language })
    }

    /// Create an image input.
    ///
    /// # Arguments
    /// * `bytes` - Raw encoded image bytes (must not be empty)
    /// * `format` - Image format (PNG, JPEG, WebP, or GIF)
    ///
    /// # Errors
    /// Returns `EmbeddingError::EmptyInput` if bytes is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::{ModelInput, ImageFormat};
    ///
    /// let png_bytes = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic + data
    /// let image = ModelInput::image(png_bytes, ImageFormat::Png).unwrap();
    /// ```
    pub fn image(bytes: Vec<u8>, format: ImageFormat) -> EmbeddingResult<Self> {
        if bytes.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        Ok(Self::Image { bytes, format })
    }

    /// Create an audio input.
    ///
    /// # Arguments
    /// * `bytes` - Raw audio bytes (must not be empty)
    /// * `sample_rate` - Sample rate in Hz (must be > 0, e.g., 16000, 44100)
    /// * `channels` - Number of channels (must be 1 for mono or 2 for stereo)
    ///
    /// # Errors
    /// Returns `EmbeddingError::EmptyInput` if:
    /// - bytes is empty
    /// - sample_rate is 0
    /// - channels is not 1 or 2
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::ModelInput;
    ///
    /// let audio_bytes = vec![0u8; 1024]; // PCM samples
    /// let audio = ModelInput::audio(audio_bytes, 16000, 1).unwrap(); // 16kHz mono
    /// ```
    pub fn audio(bytes: Vec<u8>, sample_rate: u32, channels: u8) -> EmbeddingResult<Self> {
        if bytes.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        if sample_rate == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "Audio sample_rate cannot be 0".to_string(),
            });
        }
        if channels != 1 && channels != 2 {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "Audio channels must be 1 (mono) or 2 (stereo), got {}",
                    channels
                ),
            });
        }
        Ok(Self::Audio {
            bytes,
            sample_rate,
            channels,
        })
    }
}
