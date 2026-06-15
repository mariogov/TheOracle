//! Codebook training for PQ-8 quantization.
//!
//! This module provides:
//! - K-means clustering for training PQ-8 codebooks
//! - K-means++ initialization for better convergence
//! - Synthetic embedding generation for testing

use super::types::{KMeansConfig, PQ8QuantizationError, SimpleRng, NUM_CENTROIDS, NUM_SUBVECTORS};
use crate::quantization::types::PQ8Codebook;
use tracing::{debug, info, warn};

impl PQ8Codebook {
    /// Train a PQ8 codebook from embedding samples using k-means clustering.
    ///
    /// # Arguments
    ///
    /// * `samples` - Training embedding vectors (minimum 256 samples required)
    /// * `config` - Optional k-means configuration
    ///
    /// # Returns
    ///
    /// Trained codebook ready for quantization.
    ///
    /// # Errors
    ///
    /// - `InsufficientSamples` if fewer than NUM_CENTROIDS samples provided
    /// - `SampleDimensionMismatch` if samples have inconsistent dimensions
    /// - `KMeansDidNotConverge` if clustering fails to converge
    ///
    /// # Algorithm
    ///
    /// For each subvector position:
    /// 1. Extract subvector slices from all training samples
    /// 2. Initialize centroids using k-means++ initialization
    /// 3. Run k-means until convergence or max iterations
    /// 4. Store trained centroids
    pub fn train(
        samples: &[Vec<f32>],
        config: Option<KMeansConfig>,
    ) -> Result<Self, PQ8QuantizationError> {
        let config = config.unwrap_or_default();

        // Validate we have enough samples
        if samples.len() < NUM_CENTROIDS {
            return Err(PQ8QuantizationError::InsufficientSamples {
                required: NUM_CENTROIDS,
                provided: samples.len(),
            });
        }

        // Validate sample dimensions
        if samples.is_empty() {
            return Err(PQ8QuantizationError::EmptyEmbedding);
        }

        let embedding_dim = samples[0].len();
        if !embedding_dim.is_multiple_of(NUM_SUBVECTORS) {
            return Err(PQ8QuantizationError::DimensionNotDivisible { dim: embedding_dim });
        }

        // Validate all samples have same dimension
        for (idx, sample) in samples.iter().enumerate() {
            if sample.len() != embedding_dim {
                return Err(PQ8QuantizationError::SampleDimensionMismatch {
                    sample_idx: idx,
                    expected: embedding_dim,
                    got: sample.len(),
                });
            }
            // Validate no NaN/Inf
            for (i, &val) in sample.iter().enumerate() {
                if val.is_nan() {
                    return Err(PQ8QuantizationError::ContainsNaN { index: i });
                }
                if val.is_infinite() {
                    return Err(PQ8QuantizationError::ContainsInfinity { index: i });
                }
            }
        }

        let subvector_dim = embedding_dim / NUM_SUBVECTORS;
        let mut centroids = Vec::with_capacity(NUM_SUBVECTORS);

        info!(
            target: "quantization::pq8",
            embedding_dim = embedding_dim,
            num_samples = samples.len(),
            subvector_dim = subvector_dim,
            max_iterations = config.max_iterations,
            "Training PQ8 codebook"
        );

        // Train centroids for each subvector position
        for sv_idx in 0..NUM_SUBVECTORS {
            let start = sv_idx * subvector_dim;
            let end = start + subvector_dim;

            // Extract subvectors for this position from all samples
            let subvectors: Vec<Vec<f32>> =
                samples.iter().map(|s| s[start..end].to_vec()).collect();

            // Run k-means clustering for this subvector
            let subvector_centroids =
                Self::kmeans_cluster(&subvectors, NUM_CENTROIDS, &config, sv_idx)?;

            centroids.push(subvector_centroids);
        }

        info!(
            target: "quantization::pq8",
            embedding_dim = embedding_dim,
            "PQ8 codebook training complete"
        );

        Ok(Self {
            embedding_dim,
            num_subvectors: NUM_SUBVECTORS,
            num_centroids: NUM_CENTROIDS,
            centroids,
            codebook_id: Self::generate_codebook_id(samples),
        })
    }

    /// Generate a unique codebook ID based on training data hash.
    ///
    /// Uses a deterministic hash of sample statistics to create reproducible IDs.
    fn generate_codebook_id(samples: &[Vec<f32>]) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        samples.len().hash(&mut hasher);
        if let Some(first) = samples.first() {
            first.len().hash(&mut hasher);
            // Hash values from start, middle, and end for better uniqueness
            let len = first.len();
            for &v in first.iter().take(10) {
                v.to_bits().hash(&mut hasher);
            }
            if len > 20 {
                let mid = len / 2;
                for &v in first.iter().skip(mid).take(10) {
                    v.to_bits().hash(&mut hasher);
                }
            }
            if len > 30 {
                for &v in first.iter().skip(len - 10).take(10) {
                    v.to_bits().hash(&mut hasher);
                }
            }
        }
        // Also hash last sample for additional uniqueness
        if samples.len() > 1 {
            if let Some(last) = samples.last() {
                for &v in last.iter().take(5) {
                    v.to_bits().hash(&mut hasher);
                }
            }
        }
        (hasher.finish() & 0xFFFFFFFF) as u32
    }

    /// K-means clustering implementation for a single subvector position.
    fn kmeans_cluster(
        subvectors: &[Vec<f32>],
        k: usize,
        config: &KMeansConfig,
        subvector_idx: usize,
    ) -> Result<Vec<Vec<f32>>, PQ8QuantizationError> {
        let n = subvectors.len();
        let dim = subvectors[0].len();

        // Initialize centroids using k-means++ for better convergence
        let mut centroids =
            Self::kmeans_plusplus_init(subvectors, k, config.seed + subvector_idx as u64);
        let mut assignments = vec![0usize; n];

        let mut converged = false;
        let mut iteration = 0;

        while iteration < config.max_iterations {
            // E-step: Assign each point to nearest centroid
            for (i, sv) in subvectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_k = 0;
                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = Self::squared_euclidean(sv, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_k = j;
                    }
                }
                assignments[i] = best_k;
            }

            // M-step: Update centroids
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, sv) in subvectors.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (d, &val) in sv.iter().enumerate() {
                    new_centroids[cluster][d] += val;
                }
            }

            // Compute averages and check convergence
            let mut max_movement = 0.0f32;
            for j in 0..k {
                if counts[j] > 0 {
                    let divisor = counts[j] as f32;
                    for val in &mut new_centroids[j] {
                        *val /= divisor;
                    }
                } else {
                    // Handle empty cluster: reinitialize from random point
                    let idx = (config.seed as usize + j + iteration) % n;
                    new_centroids[j] = subvectors[idx].clone();
                }

                let movement = Self::squared_euclidean(&centroids[j], &new_centroids[j]).sqrt();
                if movement > max_movement {
                    max_movement = movement;
                }
            }

            centroids = new_centroids;
            iteration += 1;

            if max_movement < config.convergence_threshold {
                converged = true;
                debug!(
                    target: "quantization::pq8",
                    subvector_idx = subvector_idx,
                    iterations = iteration,
                    "K-means converged"
                );
                break;
            }
        }

        if !converged {
            warn!(
                target: "quantization::pq8",
                subvector_idx = subvector_idx,
                iterations = iteration,
                max_iterations = config.max_iterations,
                "K-means did not fully converge, using best result"
            );
            // Don't error - use best result after max iterations
        }

        Ok(centroids)
    }

    /// K-means++ initialization for better centroid starting points.
    fn kmeans_plusplus_init(data: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
        let n = data.len();
        let mut rng = SimpleRng::new(seed);
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly
        let first_idx = rng.next_usize() % n;
        centroids.push(data[first_idx].clone());

        // Pick remaining centroids with probability proportional to D^2
        let mut distances = vec![f32::MAX; n];

        for _ in 1..k {
            // Update distances to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let dist_to_last = Self::squared_euclidean(point, centroids.last().unwrap());
                distances[i] = distances[i].min(dist_to_last);
            }

            // Compute cumulative probabilities
            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                // All points are at centroids, pick random
                let idx = rng.next_usize() % n;
                centroids.push(data[idx].clone());
                continue;
            }

            let threshold = rng.next_f32() * total;
            let mut cumsum = 0.0f32;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.push(data[chosen].clone());
        }

        centroids
    }

    /// Squared Euclidean distance between two vectors.
    #[inline]
    fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }
}

/// Generate realistic synthetic embeddings for testing codebook training.
/// These embeddings have clustered structure similar to real neural network outputs.
///
/// # Algorithm
/// Generates embeddings in clusters around random centroids, which better represents
/// real embedding distributions that have semantic structure. This enables meaningful
/// PQ codebook training.
///
/// # Arguments
/// * `num_samples` - Number of embeddings to generate
/// * `dim` - Embedding dimension
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of normalized embedding vectors with cluster structure
pub fn generate_realistic_embeddings(num_samples: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SimpleRng::new(seed);
    let mut samples = Vec::with_capacity(num_samples);

    // Create cluster centroids (simulate semantic clusters in real embeddings)
    // Use ~sqrt(num_samples) clusters for good coverage
    let num_clusters = ((num_samples as f32).sqrt() as usize).max(10);
    let mut cluster_centroids: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);

    for _ in 0..num_clusters {
        // Generate cluster centroid with structure (not purely random)
        let mut centroid: Vec<f32> = (0..dim)
            .map(|d| {
                // Create structured centroids with varying activation patterns
                let base = ((d as f32 / dim as f32) * std::f32::consts::TAU).sin();
                let noise = (rng.next_f32() - 0.5) * 0.5;
                base * 0.7 + noise
            })
            .collect();

        // Normalize centroid
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut centroid {
                *v /= norm;
            }
        }
        cluster_centroids.push(centroid);
    }

    // Generate samples around cluster centroids
    for i in 0..num_samples {
        // Select a cluster (with some determinism based on sample index)
        let cluster_idx = (i + rng.next_usize()) % num_clusters;
        let centroid = &cluster_centroids[cluster_idx];

        // Generate embedding near the centroid with small Gaussian noise
        let noise_scale = 0.15; // Small noise to stay near centroid
        let mut embedding: Vec<f32> = centroid
            .iter()
            .map(|&c| {
                // Add small Gaussian-like noise using Box-Muller
                let u1 = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
                let u2 = rng.next_u64() as f64 / u64::MAX as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                c + (z as f32) * noise_scale
            })
            .collect();

        // L2 normalize (real embeddings are typically normalized)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        samples.push(embedding);
    }

    samples
}
