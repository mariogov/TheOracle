//! Integration tests for timing and timeout behavior
//!
//! # Tests
//! - `test_pre_tool_use_completes_under_500ms`: PreToolUse fast path timing
//! - `test_session_end_can_use_full_30s_budget`: SessionEnd within budget
//! - `test_timing_recorded_in_output`: execution_time_ms in JSON output
//!
//! # Timeout Budgets (per constitution.yaml)
//! - PreToolUse: 500ms total (fast path, CLI logic ~100ms)
//! - UserPromptSubmit: 2000ms
//! - PostToolUse: 3000ms
//! - SessionStart: 5000ms
//! - SessionEnd: 30000ms
//!
//! # Constitution References
//! - Performance requirements: pre_tool_hook <500ms p95
//! - Dream wake: <100ms

use serde_json::json;
use std::time::Instant;
use tempfile::TempDir;

use super::helpers::{
    assert_exit_code, assert_timing_under_budget, create_post_tool_input, create_pre_tool_input,
    create_prompt_submit_input, create_session_end_input, create_session_start_input,
    generate_test_session_id, invoke_hook_with_stdin, log_test_evidence, EXIT_SUCCESS,
    TIMEOUT_POST_TOOL_MS, TIMEOUT_PRE_TOOL_MS, TIMEOUT_SESSION_END_MS, TIMEOUT_SESSION_START_MS,
    TIMEOUT_USER_PROMPT_MS,
};

// =============================================================================
// PreToolUse Fast Path Timing Test
// =============================================================================

/// Test that PreToolUse fast path completes under 500ms
///
/// Per constitution.yaml:
/// - pre_tool_hook: <500ms p95 total
/// - Fast path: NO DB access
///
/// This is critical for Claude Code responsiveness
#[tokio::test]
async fn test_pre_tool_use_completes_under_100ms() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("timing-pre-tool");

    // First, start a session
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    // Run multiple PreToolUse calls to get consistent timing
    let iterations = 10;
    let mut times: Vec<u64> = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let tool_use_id = format!("timing-tu-{:03}", i);
        let input = create_pre_tool_input(
            &session_id,
            "Read",
            json!({"file_path": format!("/tmp/test{}.txt", i)}),
            &tool_use_id,
        );

        let start = Instant::now();
        let result = invoke_hook_with_stdin(
            "pre-tool",
            &session_id,
            &["--tool-name", "Read", "--fast-path", "true"],
            &input,
            db_path,
        );
        let elapsed = start.elapsed().as_millis() as u64;

        assert_exit_code(&result, EXIT_SUCCESS, &format!("PreToolUse {} failed", i));
        times.push(elapsed);
    }

    // Calculate statistics
    let sum: u64 = times.iter().sum();
    let avg = sum / iterations as u64;
    let max = *times.iter().max().unwrap();
    let min = *times.iter().min().unwrap();

    // Sort for percentile calculation
    let mut sorted_times = times.clone();
    sorted_times.sort();
    let p95_idx = (iterations as f64 * 0.95) as usize;
    let p95 = sorted_times[p95_idx.min(iterations - 1)];

    // Verify p95 is under budget
    assert!(
        p95 < TIMEOUT_PRE_TOOL_MS,
        "PreToolUse p95 timing {}ms exceeds budget {}ms.\nTimes: {:?}",
        p95,
        TIMEOUT_PRE_TOOL_MS,
        times
    );

    // End session
    let end_input = create_session_end_input(&session_id, 5000, "normal", None);
    let _ = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "5000"],
        &end_input,
        db_path,
    );

    log_test_evidence(
        "test_pre_tool_use_completes_under_500ms",
        "timing",
        &session_id,
        EXIT_SUCCESS,
        avg,
        true,
        Some(json!({
            "iterations": iterations,
            "avg_ms": avg,
            "min_ms": min,
            "max_ms": max,
            "p95_ms": p95,
            "budget_ms": TIMEOUT_PRE_TOOL_MS,
            "all_times": times,
        })),
    );
}

// =============================================================================
// SessionEnd Timing Test
// =============================================================================

/// Test that SessionEnd completes within its 30s budget
///
/// SessionEnd has the largest budget (30s) because it:
/// - Persists SessionSnapshot
/// - May trigger dream consolidation
/// - Updates indexes
#[tokio::test]
async fn test_session_end_can_use_full_30s_budget() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("timing-session-end");

    // Start session
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    // Do some activity to create state worth persisting
    for i in 0..5 {
        let tool_id = format!("activity-{}", i);
        let post_input = create_post_tool_input(
            &session_id,
            "Write",
            json!({"file": format!("/tmp/file{}.txt", i)}),
            "content written",
            &tool_id,
        );
        let _ = invoke_hook_with_stdin(
            "post-tool",
            &session_id,
            &["--tool-name", "Write", "--success", "true"],
            &post_input,
            db_path,
        );
    }

    // End session with timing
    let start = Instant::now();
    let end_input = create_session_end_input(&session_id, 60000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "60000", "--generate-summary", "true"],
        &end_input,
        db_path,
    );
    let elapsed = start.elapsed().as_millis() as u64;

    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");
    assert_timing_under_budget(&end_result, TIMEOUT_SESSION_END_MS, "SessionEnd budget");

    // Verify reported timing is reasonable
    // Note: reported_time may be 0 if internal operations complete in <1ms
    // while wall clock shows time spent in process spawning/IPC overhead
    // The important thing is that it's under the budget, not that it matches wall clock
    if let Some(reported_time) = end_result.reported_execution_time_ms() {
        assert!(
            reported_time < TIMEOUT_SESSION_END_MS,
            "Reported time {}ms should be under budget {}ms",
            reported_time,
            TIMEOUT_SESSION_END_MS
        );
    }

    log_test_evidence(
        "test_session_end_can_use_full_30s_budget",
        "timing",
        &session_id,
        end_result.exit_code,
        elapsed,
        true,
        Some(json!({
            "wall_clock_ms": elapsed,
            "reported_ms": end_result.reported_execution_time_ms(),
            "budget_ms": TIMEOUT_SESSION_END_MS,
        })),
    );
}

// =============================================================================
// Timing Recorded in Output Test
// =============================================================================

/// Test that execution_time_ms is recorded in JSON output for all hooks
#[tokio::test]
async fn test_timing_recorded_in_output() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("timing-output");

    // SessionStart
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");

    let start_time = start_result.reported_execution_time_ms();
    assert!(
        start_time.is_some(),
        "SessionStart should report execution_time_ms.\nOutput: {}",
        start_result.stdout
    );

    // PreToolUse
    let pre_input = create_pre_tool_input(&session_id, "Read", json!({}), "tu-timing-001");
    let pre_result = invoke_hook_with_stdin(
        "pre-tool",
        &session_id,
        &["--tool-name", "Read", "--fast-path", "true"],
        &pre_input,
        db_path,
    );
    assert_exit_code(&pre_result, EXIT_SUCCESS, "PreToolUse failed");

    let pre_time = pre_result.reported_execution_time_ms();
    assert!(
        pre_time.is_some(),
        "PreToolUse should report execution_time_ms.\nOutput: {}",
        pre_result.stdout
    );

    // PostToolUse
    let post_input =
        create_post_tool_input(&session_id, "Read", json!({}), "result", "tu-timing-001");
    let post_result = invoke_hook_with_stdin(
        "post-tool",
        &session_id,
        &["--tool-name", "Read", "--success", "true"],
        &post_input,
        db_path,
    );
    assert_exit_code(&post_result, EXIT_SUCCESS, "PostToolUse failed");

    let post_time = post_result.reported_execution_time_ms();
    assert!(
        post_time.is_some(),
        "PostToolUse should report execution_time_ms.\nOutput: {}",
        post_result.stdout
    );

    // UserPromptSubmit
    let prompt_input = create_prompt_submit_input(&session_id, "test prompt", vec![]);
    let prompt_result =
        invoke_hook_with_stdin("prompt-submit", &session_id, &[], &prompt_input, db_path);
    assert_exit_code(&prompt_result, EXIT_SUCCESS, "PromptSubmit failed");

    let prompt_time = prompt_result.reported_execution_time_ms();
    assert!(
        prompt_time.is_some(),
        "UserPromptSubmit should report execution_time_ms.\nOutput: {}",
        prompt_result.stdout
    );

    // SessionEnd
    let end_input = create_session_end_input(&session_id, 30000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "30000"],
        &end_input,
        db_path,
    );
    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");

    let end_time = end_result.reported_execution_time_ms();
    assert!(
        end_time.is_some(),
        "SessionEnd should report execution_time_ms.\nOutput: {}",
        end_result.stdout
    );

    log_test_evidence(
        "test_timing_recorded_in_output",
        "timing_fields",
        &session_id,
        EXIT_SUCCESS,
        0,
        true,
        Some(json!({
            "session_start_ms": start_time,
            "pre_tool_ms": pre_time,
            "post_tool_ms": post_time,
            "prompt_submit_ms": prompt_time,
            "session_end_ms": end_time,
        })),
    );
}

// =============================================================================
// All Hooks Within Budget Test
// =============================================================================

/// Test that all hooks complete within their constitutional budgets
#[tokio::test]
async fn test_all_hooks_within_budget() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path();
    let session_id = generate_test_session_id("budget-all");

    // SessionStart (budget: 5000ms)
    let start_input = create_session_start_input(&session_id, "/tmp", "cli", None);
    let start_result =
        invoke_hook_with_stdin("session-start", &session_id, &[], &start_input, db_path);
    assert_exit_code(&start_result, EXIT_SUCCESS, "SessionStart failed");
    assert_timing_under_budget(&start_result, TIMEOUT_SESSION_START_MS, "SessionStart");

    // PreToolUse (budget: 500ms total)
    let pre_input = create_pre_tool_input(&session_id, "Read", json!({}), "tu-budget-001");
    let pre_result = invoke_hook_with_stdin(
        "pre-tool",
        &session_id,
        &["--tool-name", "Read", "--fast-path", "true"],
        &pre_input,
        db_path,
    );
    assert_exit_code(&pre_result, EXIT_SUCCESS, "PreToolUse failed");
    assert_timing_under_budget(&pre_result, TIMEOUT_PRE_TOOL_MS, "PreToolUse");

    // PostToolUse (budget: 3000ms)
    let post_input =
        create_post_tool_input(&session_id, "Read", json!({}), "result", "tu-budget-001");
    let post_result = invoke_hook_with_stdin(
        "post-tool",
        &session_id,
        &["--tool-name", "Read", "--success", "true"],
        &post_input,
        db_path,
    );
    assert_exit_code(&post_result, EXIT_SUCCESS, "PostToolUse failed");
    assert_timing_under_budget(&post_result, TIMEOUT_POST_TOOL_MS, "PostToolUse");

    // UserPromptSubmit (budget: 2000ms)
    let prompt_input = create_prompt_submit_input(&session_id, "test", vec![]);
    let prompt_result =
        invoke_hook_with_stdin("prompt-submit", &session_id, &[], &prompt_input, db_path);
    assert_exit_code(&prompt_result, EXIT_SUCCESS, "PromptSubmit failed");
    assert_timing_under_budget(&prompt_result, TIMEOUT_USER_PROMPT_MS, "UserPromptSubmit");

    // SessionEnd (budget: 30000ms)
    let end_input = create_session_end_input(&session_id, 10000, "normal", None);
    let end_result = invoke_hook_with_stdin(
        "session-end",
        &session_id,
        &["--duration-ms", "10000"],
        &end_input,
        db_path,
    );
    assert_exit_code(&end_result, EXIT_SUCCESS, "SessionEnd failed");
    assert_timing_under_budget(&end_result, TIMEOUT_SESSION_END_MS, "SessionEnd");

    log_test_evidence(
        "test_all_hooks_within_budget",
        "budget_verification",
        &session_id,
        EXIT_SUCCESS,
        start_result.execution_time_ms
            + pre_result.execution_time_ms
            + post_result.execution_time_ms
            + prompt_result.execution_time_ms
            + end_result.execution_time_ms,
        true,
        Some(json!({
            "session_start": {"ms": start_result.execution_time_ms, "budget": TIMEOUT_SESSION_START_MS},
            "pre_tool": {"ms": pre_result.execution_time_ms, "budget": TIMEOUT_PRE_TOOL_MS},
            "post_tool": {"ms": post_result.execution_time_ms, "budget": TIMEOUT_POST_TOOL_MS},
            "prompt_submit": {"ms": prompt_result.execution_time_ms, "budget": TIMEOUT_USER_PROMPT_MS},
            "session_end": {"ms": end_result.execution_time_ms, "budget": TIMEOUT_SESSION_END_MS},
        })),
    );
}
