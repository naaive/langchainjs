use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ToolError;

/// Ambient context handed to every tool invocation. Kept deliberately small;
/// tools needing richer state should close over it at construction time.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Working directory tools should resolve relative paths against.
    pub cwd: PathBuf,
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// What a tool hands back to the model. Errors are data, not exceptions:
/// a failed tool call becomes an `is_error` result the model can react to,
/// instead of aborting the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Conversion used by `#[tool]`-generated code so tool functions can return
/// plain `String`s, JSON values, or a full [`ToolOutput`].
pub trait IntoToolOutput {
    fn into_tool_output(self) -> ToolOutput;
}

impl IntoToolOutput for ToolOutput {
    fn into_tool_output(self) -> ToolOutput {
        self
    }
}

impl IntoToolOutput for String {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for &str {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self)
    }
}

impl IntoToolOutput for serde_json::Value {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text(self.to_string())
    }
}

impl IntoToolOutput for () {
    fn into_tool_output(self) -> ToolOutput {
        ToolOutput::text("ok")
    }
}

/// Something the model can call. Usually implemented via the `#[tool]` macro,
/// which derives the JSON Schema at compile time — but it is a plain
/// object-safe trait, so tools can also be hand-written or proxied.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema describing the tool's input object.
    fn input_schema(&self) -> serde_json::Value;
    /// Tools that mutate the outside world can demand human approval; the
    /// agent runtime surfaces this as a first-class `AwaitingApproval` state.
    fn needs_approval(&self) -> bool {
        false
    }
    async fn call(
        &self,
        input: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError>;
}

impl dyn Tool {
    /// The [`ToolSpec`](crate::ToolSpec) advertised to the model.
    pub fn spec(&self) -> crate::ToolSpec {
        crate::ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}
