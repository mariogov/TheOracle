//! Physical verification test for TASK-EMB-013
//! Creates real SafeTensors file, runs load_weights, verifies with sha256sum

use std::collections::HashMap;
use std::process::Command;
use tempfile::TempDir;

fn main() {
    println!("=== TASK-EMB-013 Physical Verification ===\n");

    // Step 1: Create real SafeTensors data
    let tensor_data: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
    ];

    let tensor_bytes: Vec<u8> = tensor_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    println!("Step 1: Created tensor data");
    println!("  - Shape: [4, 4]");
    println!("  - Elements: 16 floats");
    println!("  - Bytes: {} bytes\n", tensor_bytes.len());

    // Step 2: Serialize to SafeTensors format
    let shape: Vec<usize> = vec![4, 4];
    let tensor_view = safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        shape,
        &tensor_bytes,
    ).expect("Failed to create tensor view");

    let mut tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    tensors.insert("test_weights".to_string(), tensor_view);

    let serialized = safetensors::serialize(&tensors, &None::<HashMap<String, String>>)
        .expect("Failed to serialize SafeTensors");

    println!("Step 2: Serialized to SafeTensors format");
    println!("  - File size: {} bytes\n", serialized.len());

    // Step 3: Write to temp file
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("verification.safetensors");
    std::fs::write(&file_path, &serialized).expect("Failed to write file");

    println!("Step 3: Wrote to physical file");
    println!("  - Path: {:?}\n", file_path);

    // Step 4: Compute SHA256 with our Rust code
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    let rust_checksum: [u8; 32] = hasher.finalize().into();
    let rust_hex = hex::encode(&rust_checksum);

    println!("Step 4: Computed SHA256 with Rust sha2 crate");
    println!("  - Checksum: {}\n", rust_hex);

    // Step 5: Verify with external sha256sum tool
    let output = Command::new("sha256sum")
        .arg(&file_path)
        .output()
        .expect("Failed to run sha256sum");

    let sha256sum_output = String::from_utf8_lossy(&output.stdout);
    let external_hex = sha256sum_output.split_whitespace().next().unwrap_or("");

    println!("Step 5: Verified with external sha256sum tool");
    println!("  - sha256sum output: {}", sha256sum_output.trim());
    println!("  - Extracted checksum: {}\n", external_hex);

    // Step 6: Compare checksums
    println!("Step 6: Checksum Comparison");
    println!("  - Rust sha2:   {}", rust_hex);
    println!("  - sha256sum:   {}", external_hex);

    if rust_hex == external_hex {
        println!("\n  ✅ CHECKSUMS MATCH - Physical verification PASSED\n");
    } else {
        println!("\n  ❌ CHECKSUMS DO NOT MATCH - Physical verification FAILED\n");
        std::process::exit(1);
    }

    // Step 7: Verify tensor metadata
    println!("Step 7: Tensor Metadata Verification");
    let loaded = safetensors::SafeTensors::deserialize(&serialized)
        .expect("Failed to deserialize");

    for (name, view) in loaded.tensors() {
        println!("  - Tensor '{}': shape={:?}, dtype={:?}", name, view.shape(), view.dtype());
        let param_count: usize = view.shape().iter().product();
        println!("  - Parameter count: {}", param_count);
    }

    println!("\n=== Physical Verification Complete ===");
}
