//! InputType enum tests (M03-F07).

use crate::types::input::{ImageFormat, InputType, ModelInput};
use std::collections::{HashMap, HashSet};

// ============================================================
// INPUT TYPE TESTS (M03-F07)
// ============================================================

#[test]
fn test_input_type_from_model_input_text() {
    println!("BEFORE: Creating ModelInput::Text");
    let input = ModelInput::text("Hello").unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Text);
}

#[test]
fn test_input_type_from_model_input_code() {
    println!("BEFORE: Creating ModelInput::Code");
    let input = ModelInput::code("fn main() {}", "rust").unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Code);
}

#[test]
fn test_input_type_from_model_input_image() {
    println!("BEFORE: Creating ModelInput::Image");
    let input = ModelInput::image(vec![1, 2, 3, 4], ImageFormat::Png).unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Image);
}

#[test]
fn test_input_type_from_model_input_audio() {
    println!("BEFORE: Creating ModelInput::Audio");
    let input = ModelInput::audio(vec![1, 2, 3, 4], 16000, 1).unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Audio);
}

#[test]
fn test_input_type_display_lowercase() {
    assert_eq!(format!("{}", InputType::Text), "text");
    assert_eq!(format!("{}", InputType::Code), "code");
    assert_eq!(format!("{}", InputType::Image), "image");
    assert_eq!(format!("{}", InputType::Audio), "audio");
}

#[test]
fn test_input_type_all_returns_4_variants() {
    let all = InputType::all();
    assert_eq!(all.len(), 4);
    assert_eq!(all[0], InputType::Text);
    assert_eq!(all[1], InputType::Code);
    assert_eq!(all[2], InputType::Image);
    assert_eq!(all[3], InputType::Audio);
}

#[test]
fn test_input_type_can_be_used_as_hashmap_key() {
    let mut map: HashMap<InputType, &str> = HashMap::new();
    map.insert(InputType::Text, "text_value");
    map.insert(InputType::Code, "code_value");
    map.insert(InputType::Image, "image_value");
    map.insert(InputType::Audio, "audio_value");

    assert_eq!(map.get(&InputType::Text), Some(&"text_value"));
    assert_eq!(map.get(&InputType::Code), Some(&"code_value"));
    assert_eq!(map.get(&InputType::Image), Some(&"image_value"));
    assert_eq!(map.get(&InputType::Audio), Some(&"audio_value"));
}

#[test]
fn test_input_type_can_be_used_in_hashset() {
    let mut set: HashSet<InputType> = HashSet::new();
    set.insert(InputType::Text);
    set.insert(InputType::Code);

    assert!(set.contains(&InputType::Text));
    assert!(set.contains(&InputType::Code));
    assert!(!set.contains(&InputType::Image));
    assert!(!set.contains(&InputType::Audio));
}

#[test]
fn test_input_type_copy_semantics() {
    let original = InputType::Text;
    let copied = original; // Copy, not move
    assert_eq!(original, copied); // Both still valid
}

#[test]
fn test_input_type_serde_roundtrip() {
    for input_type in InputType::all() {
        let json = serde_json::to_string(input_type).unwrap();
        let recovered: InputType = serde_json::from_str(&json).unwrap();
        assert_eq!(*input_type, recovered);
        println!(
            "Serialized {:?} as {} and recovered successfully",
            input_type, json
        );
    }
}

#[test]
fn test_input_type_discriminant_values() {
    assert_eq!(InputType::Text.discriminant(), 0);
    assert_eq!(InputType::Code.discriminant(), 1);
    assert_eq!(InputType::Image.discriminant(), 2);
    assert_eq!(InputType::Audio.discriminant(), 3);
}

#[test]
fn test_input_type_debug_formatting() {
    // Debug should show variant name
    assert_eq!(format!("{:?}", InputType::Text), "Text");
    assert_eq!(format!("{:?}", InputType::Code), "Code");
    assert_eq!(format!("{:?}", InputType::Image), "Image");
    assert_eq!(format!("{:?}", InputType::Audio), "Audio");
}

#[test]
fn test_input_type_equality() {
    // Same types should be equal
    assert_eq!(InputType::Text, InputType::Text);
    assert_eq!(InputType::Code, InputType::Code);
    assert_eq!(InputType::Image, InputType::Image);
    assert_eq!(InputType::Audio, InputType::Audio);

    // Different types should not be equal
    assert_ne!(InputType::Text, InputType::Code);
    assert_ne!(InputType::Text, InputType::Image);
    assert_ne!(InputType::Text, InputType::Audio);
    assert_ne!(InputType::Code, InputType::Image);
    assert_ne!(InputType::Code, InputType::Audio);
    assert_ne!(InputType::Image, InputType::Audio);
}
