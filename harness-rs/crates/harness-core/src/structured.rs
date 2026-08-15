//! Typed structured output: `model.generate_as::<T>(req)`.
//!
//! Implemented as an extension trait over any [`ChatModel`]: the target type's
//! JSON Schema (via `schemars`) is exposed as a forced tool call, and the tool
//! input is deserialized into `T`. If the model's output fails validation, the
//! serde error is fed back and the model gets a bounded number of repair
//! attempts before the call fails.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{ChatModel, Content, Message, ModelError, Request, Role, ToolChoice, ToolSpec};

/// Name of the synthetic tool used to carry structured output.
pub const STRUCTURED_OUTPUT_TOOL: &str = "emit_structured_output";

/// How many times a schema-invalid response is sent back for repair.
const REPAIR_ATTEMPTS: u32 = 2;

#[async_trait::async_trait]
pub trait GenerateStructured: ChatModel {
    /// Generate a response deserialized into `T`. Tools already present in
    /// the request remain available to the model; the final answer must come
    /// through the synthetic output tool, which this method forces via
    /// `tool_choice` when the request carries no other tools.
    async fn generate_as<T>(&self, mut req: Request) -> Result<T, ModelError>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        let schema = serde_json::to_value(schemars::schema_for!(T))
            .map_err(|e| ModelError::Deserialize(format!("schema for output type: {e}")))?;
        // Only force the tool when it's the sole tool; otherwise the model
        // may legitimately need to call real tools first.
        if req.tools.is_empty() {
            req.tool_choice = Some(ToolChoice::Tool(STRUCTURED_OUTPUT_TOOL.into()));
        }
        req.tools.push(ToolSpec {
            name: STRUCTURED_OUTPUT_TOOL.into(),
            description: "Emit the final answer in the required structured format. \
                          Call this exactly once with the complete answer."
                .into(),
            input_schema: schema,
        });

        for attempt in 0..=REPAIR_ATTEMPTS {
            let resp = self.generate(req.clone()).await?;
            let output = resp
                .message
                .tool_uses()
                .find(|(_, name, _)| *name == STRUCTURED_OUTPUT_TOOL)
                .map(|(id, _, input)| (id.to_string(), input.clone()));

            match output {
                Some((id, input)) => match serde_json::from_value::<T>(input) {
                    Ok(value) => return Ok(value),
                    Err(e) if attempt < REPAIR_ATTEMPTS => {
                        req.messages.push(resp.message.clone());
                        req.messages.push(Message {
                            role: Role::User,
                            content: vec![Content::ToolResult {
                                tool_use_id: id,
                                content: format!(
                                    "Output failed schema validation: {e}. \
                                     Call {STRUCTURED_OUTPUT_TOOL} again with corrected arguments."
                                ),
                                is_error: true,
                            }],
                        });
                    }
                    Err(e) => {
                        return Err(ModelError::Deserialize(format!(
                        "structured output failed validation after {REPAIR_ATTEMPTS} repairs: {e}"
                    )))
                    }
                },
                None if attempt < REPAIR_ATTEMPTS => {
                    req.messages.push(resp.message.clone());
                    req.messages.push(Message::user(format!(
                        "You must provide the final answer by calling the \
                         {STRUCTURED_OUTPUT_TOOL} tool."
                    )));
                }
                None => {
                    return Err(ModelError::Deserialize(
                        "model never produced structured output".into(),
                    ))
                }
            }
        }
        unreachable!("loop returns on every branch of the final attempt")
    }
}

impl<M: ChatModel + ?Sized> GenerateStructured for M {}
