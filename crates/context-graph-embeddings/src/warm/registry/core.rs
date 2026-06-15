//! Core registry implementation.
//!
//! Contains the [`WarmModelRegistry`] struct and basic operations.

use std::collections::HashMap;

use super::types::WarmModelEntry;
use crate::warm::error::{WarmError, WarmResult};
use crate::warm::handle::ModelHandle;
use crate::warm::state::WarmModelState;

/// Registry for tracking warm model loading state.
///
/// Maintains a HashMap of [`WarmModelEntry`] keyed by model ID.
/// All 12 embedding models should be registered before loading begins.
///
/// # Invariants
///
/// - Model IDs are unique within the registry
/// - State transitions follow the documented state machine
/// - `handle` is `Some` only when state is `Warm`
/// - `handle` is `None` for all other states
#[derive(Debug, Default)]
pub struct WarmModelRegistry {
    /// Model entries keyed by model ID.
    pub(crate) entries: HashMap<String, WarmModelEntry>,
}

impl WarmModelRegistry {
    /// Create a new empty registry.
    ///
    /// The registry starts with no registered models. Use [`register_model`](Self::register_model)
    /// to add entries for each of the 12 embedding models.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = WarmModelRegistry::new();
    /// assert_eq!(registry.model_count(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a new model in Pending state.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier (e.g., "E1_Semantic", "E12_LateInteraction")
    /// * `expected_bytes` - Expected size of model weights in bytes
    /// * `expected_dimension` - Expected output embedding dimension
    ///
    /// # Errors
    ///
    /// Returns [`WarmError::ModelAlreadyRegistered`] if a model with the same ID
    /// already exists in the registry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.register_model("E1_Semantic", 512 * 1024 * 1024, 768)?;
    /// ```
    pub fn register_model(
        &mut self,
        model_id: impl Into<String>,
        expected_bytes: usize,
        expected_dimension: usize,
    ) -> WarmResult<()> {
        let model_id = model_id.into();

        if self.entries.contains_key(&model_id) {
            return Err(WarmError::ModelAlreadyRegistered { model_id });
        }

        let entry = WarmModelEntry::new(model_id.clone(), expected_bytes, expected_dimension);
        self.entries.insert(model_id, entry);
        Ok(())
    }

    /// Get the current state of a model.
    ///
    /// Returns `None` if the model is not registered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(state) = registry.get_state("E1_Semantic") {
    ///     if state.is_warm() {
    ///         println!("Model is ready for inference");
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn get_state(&self, model_id: &str) -> Option<WarmModelState> {
        self.entries.get(model_id).map(|e| e.state.clone())
    }

    /// Get a reference to the VRAM handle for a model.
    ///
    /// Returns `None` if the model is not registered or not in Warm state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(handle) = registry.get_handle("E1_Semantic") {
    ///     println!("VRAM address: {}", handle.vram_address_hex());
    /// }
    /// ```
    #[must_use]
    pub fn get_handle(&self, model_id: &str) -> Option<&ModelHandle> {
        self.entries.get(model_id).and_then(|e| e.handle.as_ref())
    }

    /// Get a reference to an entry by model ID.
    ///
    /// For internal use and testing.
    #[must_use]
    pub fn get_entry(&self, model_id: &str) -> Option<&WarmModelEntry> {
        self.entries.get(model_id)
    }

    /// Get the total number of registered models.
    ///
    /// Expected to be 12 when all components are registered.
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.entries.len()
    }
}
