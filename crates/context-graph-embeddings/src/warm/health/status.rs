//! Health Status Types
//!
//! Defines the overall health status enumeration for the warm loading system.

/// Overall health status of the warm loading system.
///
/// Represents the aggregate state of all models in the system.
/// Used for quick health checks and monitoring integrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmHealthStatus {
    /// All registered models are in `Warm` state and ready for inference.
    ///
    /// This is the only status that indicates the system is fully operational.
    Healthy,

    /// At least one model is still loading or validating.
    ///
    /// No models have failed. The system is progressing toward `Healthy`.
    Loading,

    /// At least one model has failed to load or validate.
    ///
    /// The system cannot serve inference requests. Check `error_messages`
    /// in [`WarmHealthCheck`](super::WarmHealthCheck) for details.
    Unhealthy,

    /// The registry is empty or inaccessible.
    ///
    /// This occurs when:
    /// - No models have been registered yet
    /// - The registry lock is poisoned (thread panic)
    /// - The system has not been initialized
    NotInitialized,
}

impl WarmHealthStatus {
    /// Returns `true` if the system is ready for inference.
    #[inline]
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Returns `true` if models are still loading.
    #[inline]
    #[must_use]
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Returns `true` if at least one model has failed.
    #[inline]
    #[must_use]
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, Self::Unhealthy)
    }

    /// Returns `true` if the system is not initialized.
    #[inline]
    #[must_use]
    pub fn is_not_initialized(&self) -> bool {
        matches!(self, Self::NotInitialized)
    }

    /// Get a human-readable status string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Loading => "loading",
            Self::Unhealthy => "unhealthy",
            Self::NotInitialized => "not_initialized",
        }
    }
}

impl std::fmt::Display for WarmHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
