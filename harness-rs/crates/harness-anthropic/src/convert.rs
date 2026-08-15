//! Mapping between harness types and the Messages API wire format.

use harness_core::{Content, Message, ModelError, Request, Response, Role, StopReason, Usage};
use serde_json::{json, Value};

pub fn to_body(req: &Request, model: &str, default_max_tokens: u32, stream: bool) -> Value {
    let messages: Vec<Value> = req.messages.iter().map(message_to_wire).collect();
    let mut body = json!({
        "model": model,
        "max_tokens": if req.max_tokens == 0 { default_max_tokens } else { req.max_tokens },
        "messages": messages,
        "stream": stream,
    });
    if let Some(system) = &req.system {
        body["system"] = json!(system);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = json!(temp);
    }
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
    }
    body
}

fn message_to_wire(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            Content::Text { text } => Some(json!({"type": "text", "text": text})),
            Content::ToolUse { id, name, input } => {
                Some(json!({"type": "tool_use", "id": id, "name": name, "input": input}))
            }
            Content::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Some(json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error,
            })),
            // Thinking blocks require signatures to round-trip; until extended
            // thinking is supported end-to-end we don't send them back.
            Content::Thinking { .. } => None,
        })
        .collect();
    json!({"role": role, "content": content})
}

pub fn parse_stop_reason(value: Option<&str>) -> StopReason {
    match value {
        Some("end_turn") | None => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some(other) => StopReason::Other(other.to_string()),
    }
}

pub fn parse_content_block(block: &Value) -> Option<Content> {
    match block["type"].as_str()? {
        "text" => Some(Content::Text {
            text: block["text"].as_str().unwrap_or_default().to_string(),
        }),
        "thinking" => Some(Content::Thinking {
            thinking: block["thinking"].as_str().unwrap_or_default().to_string(),
        }),
        "tool_use" => Some(Content::ToolUse {
            id: block["id"].as_str().unwrap_or_default().to_string(),
            name: block["name"].as_str().unwrap_or_default().to_string(),
            input: block["input"].clone(),
        }),
        // Unknown block types (e.g. future additions) are skipped rather than
        // failing the whole response.
        _ => None,
    }
}

pub fn parse_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
    }
}

/// Parse a complete (non-streaming) Messages API response.
pub fn parse_response(value: &Value) -> Result<Response, ModelError> {
    let blocks = value["content"]
        .as_array()
        .ok_or_else(|| ModelError::Deserialize("response missing content array".into()))?;
    let content: Vec<Content> = blocks.iter().filter_map(parse_content_block).collect();
    Ok(Response {
        message: Message {
            role: Role::Assistant,
            content,
        },
        stop_reason: parse_stop_reason(value["stop_reason"].as_str()),
        usage: parse_usage(&value["usage"]),
    })
}
