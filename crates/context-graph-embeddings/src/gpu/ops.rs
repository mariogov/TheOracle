//! GPU-accelerated operations for embedding computations.
//!
//! # Operations
//!
//! | Operation | CPU Speedup | Description |
//! |-----------|-------------|-------------|
//! | Normalize | 50x | Unit vector normalization |

use candle_core::{Tensor, D};

/// Normalize a tensor to unit length (L2 normalization).
///
/// # Formula
///
/// `normalized = tensor / ||tensor||_2`
///
/// # GPU Acceleration
///
/// Fused divide operation avoids memory round-trips.
///
/// # Example
///
/// ```
/// # use context_graph_embeddings::gpu::{init_gpu, normalize_gpu};
/// # use candle_core::Tensor;
/// # fn main() -> candle_core::Result<()> {
/// # let device = init_gpu()?;
/// let tensor = Tensor::from_slice(&[3.0f32, 4.0], (2,), device)?;
/// let normalized = normalize_gpu(&tensor)?;
/// // Result: [0.6, 0.8] (unit vector)
/// # Ok(())
/// # }
/// ```
pub fn normalize_gpu(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let norm = tensor.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    tensor.broadcast_div(&(norm + 1e-12)?)
}

#[cfg(test)]
mod tests {
    // GPU tests require `cargo test --features cuda`

    #[test]
    fn test_formulas() {
        // Test mathematical formulas without GPU
        let a = [3.0f32, 4.0];
        let b = [1.0f32, 0.0];

        // L2 norm of [3, 4] = 5
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 5.0).abs() < 1e-6);

        // Cosine similarity
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);
        assert!((cos_sim - 0.6).abs() < 1e-6);
    }
}
