//! Input type discriminator for model compatibility checking.
//!
//! `InputType` is a simple discriminator used to query what input types a model
//! supports, route inputs to compatible models, and reject unsupported inputs early.

use serde::{Deserialize, Serialize};

use super::ModelInput;

/// Input type capability descriptor for model compatibility checking.
///
/// Unlike `ModelInput` which carries actual data, `InputType` is a simple
/// discriminator used to:
/// - Query what input types a model supports
/// - Route inputs to compatible models
/// - Reject unsupported inputs early (fail-fast)
///
/// # Model Compatibility Matrix
///
/// | Model | Text | Code | Image | Audio |
/// |-------|------|------|-------|-------|
/// | Semantic (E1) | Y | Y* | N | N |
/// | TemporalRecent (E2) | Y | Y | N | N |
/// | TemporalPeriodic (E3) | Y | Y | N | N |
/// | TemporalPositional (E4) | Y | Y | N | N |
/// | Causal (E5) | Y | Y | N | N |
/// | Sparse (E6) | Y | Y* | N | N |
/// | Code (E7) | Y* | Y | N | N |
/// | Graph (E8) | Y | Y* | N | N |
/// | HDC (E9) | Y | Y | N | N |
/// | Multimodal (E10) | Y | N | Y | N |
/// | Entity (E11) | Y | Y* | N | N |
/// | LateInteraction (E12) | Y | Y* | N | N |
///
/// *Model can process but is not optimized for this type
///
/// # Example
///
/// ```rust
/// use context_graph_embeddings::types::{InputType, ModelInput};
/// use std::collections::HashSet;
///
/// // Query input type from ModelInput
/// let input = ModelInput::text("Hello").unwrap();
/// let input_type = InputType::from(&input);
/// assert_eq!(input_type, InputType::Text);
///
/// // Use in HashSet for model capability checking
/// let mut supported: HashSet<InputType> = HashSet::new();
/// supported.insert(InputType::Text);
/// supported.insert(InputType::Code);
/// assert!(supported.contains(&InputType::Text));
/// assert!(!supported.contains(&InputType::Image));
///
/// // Check all variants
/// for input_type in InputType::all() {
///     println!("Type: {}", input_type);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum InputType {
    /// Text content (natural language, documents, queries)
    Text = 0,
    /// Source code with language metadata
    Code = 1,
    /// Image data (PNG, JPEG, WebP, GIF)
    Image = 2,
    /// Audio data (PCM, encoded)
    Audio = 3,
}

impl InputType {
    /// Returns a static slice containing all InputType variants.
    ///
    /// Useful for iteration when checking model compatibility across all types.
    ///
    /// # Returns
    /// Static slice with all 4 variants in discriminant order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::InputType;
    ///
    /// let all_types = InputType::all();
    /// assert_eq!(all_types.len(), 4);
    /// assert_eq!(all_types[0], InputType::Text);
    /// ```
    #[must_use]
    pub const fn all() -> &'static [InputType] {
        &[
            InputType::Text,
            InputType::Code,
            InputType::Image,
            InputType::Audio,
        ]
    }

    /// Returns the discriminant value (0-3).
    ///
    /// Matches the `#[repr(u8)]` values for binary serialization.
    #[must_use]
    pub const fn discriminant(&self) -> u8 {
        *self as u8
    }
}

impl std::fmt::Display for InputType {
    /// Displays lowercase type name: "text", "code", "image", "audio".
    ///
    /// This format is used in error messages, logging, and configuration.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Text => "text",
            Self::Code => "code",
            Self::Image => "image",
            Self::Audio => "audio",
        };
        write!(f, "{}", name)
    }
}

impl From<&ModelInput> for InputType {
    /// Converts a ModelInput reference to its corresponding InputType.
    ///
    /// This is the primary bridge between the data-carrying `ModelInput`
    /// and the capability-describing `InputType`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_embeddings::types::{InputType, ModelInput, ImageFormat};
    ///
    /// let text = ModelInput::text("Hello").unwrap();
    /// assert_eq!(InputType::from(&text), InputType::Text);
    ///
    /// let code = ModelInput::code("fn main() {}", "rust").unwrap();
    /// assert_eq!(InputType::from(&code), InputType::Code);
    ///
    /// let image = ModelInput::image(vec![1,2,3], ImageFormat::Png).unwrap();
    /// assert_eq!(InputType::from(&image), InputType::Image);
    ///
    /// let audio = ModelInput::audio(vec![1,2,3], 16000, 1).unwrap();
    /// assert_eq!(InputType::from(&audio), InputType::Audio);
    /// ```
    fn from(input: &ModelInput) -> Self {
        match input {
            ModelInput::Text { .. } => InputType::Text,
            ModelInput::Code { .. } => InputType::Code,
            ModelInput::Image { .. } => InputType::Image,
            ModelInput::Audio { .. } => InputType::Audio,
        }
    }
}
