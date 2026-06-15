//! Audio construction tests for ModelInput.

use crate::error::EmbeddingError;
use crate::types::input::ModelInput;

// ============================================================
// AUDIO CONSTRUCTION TESTS (5 tests)
// ============================================================

#[test]
fn test_audio_with_valid_parameters_succeeds() {
    let bytes = vec![0u8; 1024];
    let input = ModelInput::audio(bytes.clone(), 16000, 1);
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_audio());
    let (audio_bytes, sample_rate, channels) = input.as_audio().unwrap();
    assert_eq!(audio_bytes, bytes.as_slice());
    assert_eq!(sample_rate, 16000);
    assert_eq!(channels, 1);
}

#[test]
fn test_audio_with_empty_bytes_returns_empty_input() {
    let result = ModelInput::audio(vec![], 16000, 1);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_audio_with_sample_rate_zero_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=0, channels=1");
    let result = ModelInput::audio(vec![1, 2, 3], 0, 1);
    println!("AFTER: sample_rate=0 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("sample_rate"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_audio_with_channels_zero_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=16000, channels=0");
    let result = ModelInput::audio(vec![1, 2, 3], 16000, 0);
    println!("AFTER: channels=0 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("channels") && message.contains("1") && message.contains("2"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_audio_with_channels_three_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=16000, channels=3");
    let result = ModelInput::audio(vec![1, 2, 3], 16000, 3);
    println!("AFTER: channels=3 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("3"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

// ============================================================
// STEREO AUDIO TEST
// ============================================================

#[test]
fn test_audio_stereo_accepted() {
    let result = ModelInput::audio(vec![1, 2, 3], 44100, 2);
    assert!(result.is_ok());
    let input = result.unwrap();
    let (_, sample_rate, channels) = input.as_audio().unwrap();
    assert_eq!(sample_rate, 44100);
    assert_eq!(channels, 2);
}
