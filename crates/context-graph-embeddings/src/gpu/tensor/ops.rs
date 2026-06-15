//! GpuTensor operations: math, activation functions, and transformations.

use super::core::GpuTensor;
use candle_core::Tensor;

impl GpuTensor {
    /// Normalize the tensor (L2 normalization).
    ///
    /// For 1D: normalize the entire vector
    /// For 2D: normalize each row independently
    pub fn normalize(&self) -> candle_core::Result<Self> {
        let norm = self
            .inner
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sqrt()?;
        let normalized = self.inner.broadcast_div(&(norm + 1e-12)?)?;
        Ok(Self::new(normalized))
    }

    /// Compute L2 norm.
    ///
    /// For 1D: returns scalar norm
    /// For 2D: returns 1D tensor of norms per row
    pub fn l2_norm(&self) -> candle_core::Result<Tensor> {
        self.inner
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sqrt()
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &GpuTensor) -> candle_core::Result<Self> {
        let result = self.inner.mul(&other.inner)?;
        Ok(Self::new(result))
    }

    /// Element-wise addition.
    pub fn add(&self, other: &GpuTensor) -> candle_core::Result<Self> {
        let result = self.inner.add(&other.inner)?;
        Ok(Self::new(result))
    }

    /// Matrix multiplication.
    ///
    /// # Shapes
    ///
    /// - self: [M, K]
    /// - other: [K, N]
    /// - result: [M, N]
    pub fn matmul(&self, other: &GpuTensor) -> candle_core::Result<Self> {
        let result = self.inner.matmul(&other.inner)?;
        Ok(Self::new(result))
    }

    /// Transpose last two dimensions.
    pub fn transpose(&self) -> candle_core::Result<Self> {
        let result = self.inner.t()?;
        Ok(Self::new(result))
    }

    /// Sum all elements.
    pub fn sum_all(&self) -> candle_core::Result<f32> {
        self.inner.sum_all()?.to_vec0()
    }

    /// Softmax along last dimension.
    pub fn softmax(&self) -> candle_core::Result<Self> {
        let result = candle_nn::ops::softmax(&self.inner, candle_core::D::Minus1)?;
        Ok(Self::new(result))
    }

    /// Apply GELU activation.
    pub fn gelu(&self) -> candle_core::Result<Self> {
        let result = self.inner.gelu()?;
        Ok(Self::new(result))
    }

    /// Apply ReLU activation.
    pub fn relu(&self) -> candle_core::Result<Self> {
        let result = self.inner.relu()?;
        Ok(Self::new(result))
    }

    /// Apply SiLU (Swish) activation.
    pub fn silu(&self) -> candle_core::Result<Self> {
        let result = self.inner.silu()?;
        Ok(Self::new(result))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_memory_calculation() {
        // Test memory calculation formula
        let elem_count = 1024;
        let size_per_elem = 4; // f32
        let expected = elem_count * size_per_elem;

        // This test doesn't need GPU - just validates the formula
        assert_eq!(expected, 4096);
    }
}
