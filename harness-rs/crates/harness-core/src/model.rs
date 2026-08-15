use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::{Message, ModelError};

/// A tool as advertised to the model: name, human description, JSON Schema
/// for its input. Providers translate this to their wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// How the model is allowed to use the advertised tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model decides freely (the default when unset).
    Auto,
    /// Model must call some tool.
    Any,
    /// Model must call this specific tool.
    Tool(String),
}

/// One model invocation. There is no session state hidden in the provider —
/// the full conversation travels in `messages` every call, which is what
/// makes runs resumable and providers stateless.
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
    /// Upper bound on generated tokens. Providers apply their own default if 0.
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// A complete model response: the assistant message, why generation stopped,
/// and token accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Typed streaming events. This is the whole event vocabulary — no string
/// event names, no untyped payload dictionaries.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// The model began emitting a tool call. Its input arrives incrementally
    /// as `InputJsonDelta` and is complete in the final [`Response`].
    ToolUseStart {
        id: String,
        name: String,
    },
    InputJsonDelta(String),
    /// Always the final event of a successful stream: the fully assembled
    /// response, identical to what a non-streaming call would return.
    Completed(Response),
}

pub type ModelStream<'a> = Pin<Box<dyn Stream<Item = Result<StreamEvent, ModelError>> + Send + 'a>>;

/// An LLM provider. This is the entire surface a provider must implement:
/// one streaming method. `generate` has a default implementation that drains
/// the stream; providers with a cheaper non-streaming path can override it.
///
/// There is deliberately no `batch`: batching is `futures::stream` composition
/// in the caller's code, not a framework API.
#[async_trait::async_trait]
pub trait ChatModel: Send + Sync {
    /// Stream a response. The stream must yield `Completed` as its final
    /// event on success.
    fn stream(&self, req: Request) -> ModelStream<'_>;

    /// The provider's model identifier (for telemetry). Empty if unknown.
    fn model_id(&self) -> &str {
        ""
    }

    async fn generate(&self, req: Request) -> Result<Response, ModelError> {
        let mut stream = self.stream(req);
        while let Some(event) = stream.next().await {
            if let StreamEvent::Completed(resp) = event? {
                return Ok(resp);
            }
        }
        Err(ModelError::Stream(
            "stream ended without a Completed event".into(),
        ))
    }
}
