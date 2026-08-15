use std::sync::Arc;

use harness_core::{Request, Tool, ToolContext, ToolError, ToolOutput};

/// A tool invocation as seen by middleware.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Cross-cutting hooks around the agent's model and tool calls. Unlike
/// observer-style callbacks, middleware can rewrite requests and short-circuit
/// or wrap tool execution (caching, rate limits, PII filtering, ...).
#[harness_core::async_trait]
pub trait Middleware: Send + Sync {
    /// Runs before every model call; may rewrite the request (e.g. trim or
    /// summarize history, inject context).
    async fn before_model_call(&self, _req: &mut Request) {}

    /// Onion-style wrapper around tool execution. Call `next.run(call)` to
    /// proceed; skip it to short-circuit with your own output.
    async fn on_tool_call(
        &self,
        call: ToolCall,
        next: ToolNext<'_>,
    ) -> Result<ToolOutput, ToolError> {
        next.run(call).await
    }
}

/// The remainder of the middleware chain, ending at the tool itself.
pub struct ToolNext<'a> {
    pub(crate) chain: &'a [Arc<dyn Middleware>],
    pub(crate) tool: &'a dyn Tool,
    pub(crate) ctx: &'a ToolContext,
}

impl ToolNext<'_> {
    pub async fn run(self, call: ToolCall) -> Result<ToolOutput, ToolError> {
        match self.chain.split_first() {
            Some((head, rest)) => {
                let next = ToolNext {
                    chain: rest,
                    tool: self.tool,
                    ctx: self.ctx,
                };
                head.on_tool_call(call, next).await
            }
            None => self.tool.call(call.input, self.ctx).await,
        }
    }
}
