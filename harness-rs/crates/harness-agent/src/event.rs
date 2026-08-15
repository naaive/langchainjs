use harness_core::{Message, ModelError, ToolOutput, Usage};
use thiserror::Error;

/// Everything observable about a run, as one typed stream. UI code matches on
/// this enum; there are no string event names to subscribe to.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Incremental assistant text.
    TextDelta(String),
    /// Incremental extended-thinking text.
    ThinkingDelta(String),
    /// A tool call is about to execute (input fully assembled).
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool call finished; `output.is_error` distinguishes failure.
    ToolCallFinished {
        id: String,
        name: String,
        output: ToolOutput,
    },
    /// The run is paused waiting for a human decision on a tool call.
    /// Respond via [`AgentRun::decide`](crate::AgentRun::decide).
    AwaitingApproval(ApprovalRequest),
    /// Terminal event: the run finished successfully.
    Done(RunResult),
}

/// A tool call held for human approval.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Human verdict on an [`ApprovalRequest`].
#[derive(Debug, Clone)]
pub enum Decision {
    Approve,
    /// Deny with an optional reason the model will see as the tool result.
    Deny(Option<String>),
}

/// Final state of a successful run. `messages` is the complete history
/// (input included) — pass it back into the next `run` call for multi-turn
/// sessions, or serialize it to persist the conversation.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub messages: Vec<Message>,
    /// Text of the final assistant message.
    pub final_text: String,
    /// Token usage summed over every model call in the run.
    pub usage: Usage,
    /// Number of model calls the run took.
    pub turns: u32,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Model(#[from] ModelError),

    #[error("run exceeded max_turns ({0})")]
    MaxTurns(u32),

    #[error("run was abandoned: {0}")]
    Abandoned(String),
}
