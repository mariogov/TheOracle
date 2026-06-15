//! Codebook persistence (save/load) for PQ-8 quantization.
//!
//! This module provides binary file persistence for trained codebooks:
//! - Save trained codebooks to binary files
//! - Load codebooks from binary files
//! - File format versioning for compatibility

use super::types::{
    PQ8QuantizationError, CODEBOOK_MAGIC, CODEBOOK_VERSION, NUM_CENTROIDS, NUM_SUBVECTORS,
};
use crate::quantization::types::PQ8Codebook;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::info;

impl PQ8Codebook {
    /// Save the trained codebook to a binary file.
    ///
    /// # File Format
    ///
    /// - 4 bytes: Magic "PQ8C"
    /// - 1 byte: Version (currently 1)
    /// - 4 bytes: embedding_dim (u32 little-endian)
    /// - 4 bytes: codebook_id (u32 little-endian)
    /// - For each subvector (8 total):
    ///   - For each centroid (256 total):
    ///     - subvector_dim * 4 bytes: f32 values (little-endian)
    pub fn save(&self, path: &Path) -> Result<(), PQ8QuantizationError> {
        let file = File::create(path).map_err(|e| PQ8QuantizationError::IoError {
            message: format!("Failed to create codebook file: {}", e),
        })?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer
            .write_all(CODEBOOK_MAGIC)
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to write magic: {}", e),
            })?;
        writer
            .write_all(&[CODEBOOK_VERSION])
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to write version: {}", e),
            })?;
        writer
            .write_all(&(self.embedding_dim as u32).to_le_bytes())
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to write embedding_dim: {}", e),
            })?;
        writer
            .write_all(&self.codebook_id.to_le_bytes())
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to write codebook_id: {}", e),
            })?;

        // Write centroids
        for subvector_centroids in &self.centroids {
            for centroid in subvector_centroids {
                for &val in centroid {
                    writer.write_all(&val.to_le_bytes()).map_err(|e| {
                        PQ8QuantizationError::IoError {
                            message: format!("Failed to write centroid value: {}", e),
                        }
                    })?;
                }
            }
        }

        writer.flush().map_err(|e| PQ8QuantizationError::IoError {
            message: format!("Failed to flush codebook file: {}", e),
        })?;

        info!(
            target: "quantization::pq8",
            path = %path.display(),
            embedding_dim = self.embedding_dim,
            codebook_id = self.codebook_id,
            "Saved PQ8 codebook"
        );

        Ok(())
    }

    /// Load a trained codebook from a binary file.
    pub fn load(path: &Path) -> Result<Self, PQ8QuantizationError> {
        let file = File::open(path).map_err(|e| PQ8QuantizationError::IoError {
            message: format!("Failed to open codebook file: {}", e),
        })?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to read magic: {}", e),
            })?;
        if &magic != CODEBOOK_MAGIC {
            return Err(PQ8QuantizationError::InvalidCodebookFormat {
                message: format!(
                    "Invalid magic bytes: expected {:?}, got {:?}",
                    CODEBOOK_MAGIC, magic
                ),
            });
        }

        // Read and verify version
        let mut version = [0u8; 1];
        reader
            .read_exact(&mut version)
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to read version: {}", e),
            })?;
        if version[0] != CODEBOOK_VERSION {
            return Err(PQ8QuantizationError::InvalidCodebookFormat {
                message: format!(
                    "Unsupported codebook version: expected {}, got {}",
                    CODEBOOK_VERSION, version[0]
                ),
            });
        }

        // Read embedding_dim
        let mut dim_bytes = [0u8; 4];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to read embedding_dim: {}", e),
            })?;
        let embedding_dim = u32::from_le_bytes(dim_bytes) as usize;

        // Read codebook_id
        let mut id_bytes = [0u8; 4];
        reader
            .read_exact(&mut id_bytes)
            .map_err(|e| PQ8QuantizationError::IoError {
                message: format!("Failed to read codebook_id: {}", e),
            })?;
        let codebook_id = u32::from_le_bytes(id_bytes);

        let subvector_dim = embedding_dim / NUM_SUBVECTORS;

        // Read centroids
        let mut centroids = Vec::with_capacity(NUM_SUBVECTORS);
        for _ in 0..NUM_SUBVECTORS {
            let mut subvector_centroids = Vec::with_capacity(NUM_CENTROIDS);
            for _ in 0..NUM_CENTROIDS {
                let mut centroid = Vec::with_capacity(subvector_dim);
                for _ in 0..subvector_dim {
                    let mut val_bytes = [0u8; 4];
                    reader.read_exact(&mut val_bytes).map_err(|e| {
                        PQ8QuantizationError::IoError {
                            message: format!("Failed to read centroid value: {}", e),
                        }
                    })?;
                    centroid.push(f32::from_le_bytes(val_bytes));
                }
                subvector_centroids.push(centroid);
            }
            centroids.push(subvector_centroids);
        }

        info!(
            target: "quantization::pq8",
            path = %path.display(),
            embedding_dim = embedding_dim,
            codebook_id = codebook_id,
            "Loaded PQ8 codebook"
        );

        Ok(Self {
            embedding_dim,
            num_subvectors: NUM_SUBVECTORS,
            num_centroids: NUM_CENTROIDS,
            centroids,
            codebook_id,
        })
    }
}
