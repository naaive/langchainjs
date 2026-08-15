//! Agents as tools: the multi-agent primitive.
//!
//! There is no subgraph machinery — a sub-agent is just a [`Tool`] whose
//! implementation happens to run another agent to completion. The parent
//! model delegates by calling the tool; orchestration richer than delegation
//! (parallel fan-out, voting, pipelines) belongs in ordinary Rust code.

use harness_core::{Tool, ToolContext, ToolError, ToolOutput};
use serde_json::json;

use crate::agent::Agent;

/// A sub-agent exposed as a tool. Created via [`Agent::as_tool`].
///
/// Each invocation starts a fresh conversation with the sub-agent and runs it
/// to completion, returning its final text. Approval-gated tools inside the
/// sub-agent are auto-denied (there is no human attached to a delegated run);
/// give sub-agents approval-free toolsets.
pub struct AgentTool {
    agent: Agent,
    name: String,
    description: String,
}

impl Agent {
    /// Expose this agent as a tool for another agent. `description` tells the
    /// calling model when to delegate — write it like any good tool
    /// description.
    pub fn as_tool(&self, name: impl Into<String>, description: impl Into<String>) -> AgentTool {
        AgentTool {
            agent: self.clone(),
            name: name.into(),
            description: description.into(),
        }
    }
}

#[harness_core::async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained task description for the sub-agent. \
                                    It sees nothing but this text."
                }
            },
            "required": ["task"]
        })
    }

    async fn call(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let task = input["task"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("missing required field: task".into()))?;
        let result = self
            .agent
            .run(Vec::new(), task)
            .wait()
            .await
            .map_err(|e| ToolError::Execution(format!("sub-agent failed: {e}")))?;
        Ok(ToolOutput::text(result.final_text))
    }
}
