//! Script to create a test SafeTensors fixture file for TASK-EMB-013 verification.
//! Run with: cargo run --example create_fixture

use std::collections::HashMap;
use std::fs;

fn main() {
    // Create real tensor data - NOT fake
    let tensor_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
    ];

    // Convert to bytes
    let tensor_bytes: Vec<u8> = tensor_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    // Create SafeTensors format
    let shape: Vec<usize> = vec![4, 4];
    let tensor_view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        shape,
        &tensor_bytes,
    ).expect("Failed to create tensor view");

    let mut tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    tensors.insert("test_weights".to_string(), tensor_view);

    let serialized = safetensors::serialize(&tensors, &None::<HashMap<String, String>>)
        .expect("Failed to serialize");

    // Write to fixture file
    let fixture_path = "tests/fixtures/verification_test.safetensors";
    fs::write(fixture_path, &serialized).expect("Failed to write fixture");

    println!("Created fixture: {} ({} bytes)", fixture_path, serialized.len());

    // Compute SHA256
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    let checksum: [u8; 32] = hasher.finalize().into();

    println!("SHA256 checksum (hex): {}", hex::encode(&checksum));
    println!("SHA256 first 8 bytes: {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        checksum[0], checksum[1], checksum[2], checksum[3],
        checksum[4], checksum[5], checksum[6], checksum[7]);
}
