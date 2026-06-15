//! Memory management commands
//!
//! Commands for memory capture and context injection per PRD Section 9.3.
//!
//! # Commands
//!
//! - `memory inject-context`: Inject relevant context for UserPromptSubmit hook
//! - `memory inject-brief`: Inject brief context for PreToolUse hook (<200 tokens)
//! - `memory capture-memory`: Capture hook descriptions from PostToolUse/SessionEnd
//! - `memory capture-response`: Capture Claude responses from Stop hook
//!
//! # Constitution Compliance
//!
//! - ARCH-01: TeleologicalArray is atomic (all 13 embeddings)
//! - ARCH-06: All memory ops through service layer
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - ARCH-10: Divergence detection uses SEMANTIC embedders only
//! - ARCH-11: Memory sources: HookDescription, ClaudeResponse, MDFileChunk
//! - AP-14: No .unwrap() in library code
//! - AP-26: Exit code 1 on error, 2 on corruption

pub mod capture;
pub mod inject;

use clap::Subcommand;

/// Memory management subcommands.
///
/// These commands handle memory storage, retrieval, and context injection
/// for Claude Code hook integration.
#[derive(Subcommand)]
pub enum MemoryCommands {
    /// Inject relevant context from memory store
    ///
    /// Called by UserPromptSubmit and SessionStart hooks to inject relevant
    /// memories into Claude Code's context. Embeds query using all 13 embedders,
    /// retrieves similar memories, detects divergence (semantic spaces only),
    /// and outputs formatted markdown to stdout.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # With query argument
    /// context-graph-cli memory inject-context "How do I implement HDBSCAN?"
    ///
    /// # With environment variable
    /// USER_PROMPT="test query" context-graph-cli memory inject-context
    ///
    /// # With custom budget
    /// context-graph-cli memory inject-context --budget 800 "test"
    /// ```
    InjectContext(inject::InjectContextArgs),

    /// Inject brief context for PreToolUse hook
    ///
    /// Generates compact context (<200 tokens) for tool execution.
    /// Uses TOOL_DESCRIPTION or TOOL_NAME environment variable as query.
    /// Does NOT include divergence alerts (too verbose for brief context).
    ///
    /// # Examples
    ///
    /// ```bash
    /// # With environment variable (typical hook usage)
    /// TOOL_DESCRIPTION="Writing file" context-graph-cli memory inject-brief
    ///
    /// # With explicit query
    /// context-graph-cli memory inject-brief "Editing code"
    /// ```
    InjectBrief(inject::InjectBriefArgs),

    /// Capture hook description as memory
    ///
    /// Stores hook event descriptions as HookDescription memories.
    /// Called by PostToolUse and SessionEnd hooks.
    /// Silent on success (no stdout output).
    ///
    /// # Environment Variables
    ///
    /// - `TOOL_DESCRIPTION`: Primary content source
    /// - `SESSION_SUMMARY`: Fallback content source
    /// - `CLAUDE_SESSION_ID`: Session identifier
    /// - `CONTEXT_GRAPH_DATA_DIR`: Database path
    ///
    /// # Examples
    ///
    /// ```bash
    /// # With content flag
    /// context-graph-cli memory capture-memory \
    ///   --content "Edited src/main.rs to add new function" \
    ///   --hook-type post_tool_use \
    ///   --tool-name Edit
    ///
    /// # With environment variable (typical hook usage)
    /// TOOL_DESCRIPTION="Refactored module" \
    /// context-graph-cli memory capture-memory --hook-type post_tool_use
    /// ```
    CaptureMemory(capture::CaptureMemoryArgs),

    /// Capture Claude response as memory
    ///
    /// Stores Claude responses as ClaudeResponse memories.
    /// Called by Stop hook.
    /// Silent on success (no stdout output).
    ///
    /// # Environment Variables
    ///
    /// - `RESPONSE_SUMMARY`: Content source
    /// - `CLAUDE_SESSION_ID`: Session identifier
    /// - `CONTEXT_GRAPH_DATA_DIR`: Database path
    ///
    /// # Examples
    ///
    /// ```bash
    /// # With content flag
    /// context-graph-cli memory capture-response \
    ///   --content "Session completed with 5 tasks"
    ///
    /// # With environment variable (typical hook usage)
    /// RESPONSE_SUMMARY="All tests passing" \
    /// context-graph-cli memory capture-response
    /// ```
    CaptureResponse(capture::CaptureResponseArgs),
}

/// Handle memory subcommands.
///
/// Routes to appropriate handler based on subcommand.
/// Returns exit code per AP-26: 0=success, 1=error, 2=corruption.
pub async fn handle_memory_command(cmd: MemoryCommands) -> i32 {
    match cmd {
        MemoryCommands::InjectContext(args) => inject::handle_inject_context(args).await,
        MemoryCommands::InjectBrief(args) => inject::handle_inject_brief(args).await,
        MemoryCommands::CaptureMemory(args) => capture::handle_capture_memory(args).await,
        MemoryCommands::CaptureResponse(args) => capture::handle_capture_response(args).await,
    }
}
