//! Mapping between harness types and the Chat Completions wire format,
//! plus the streaming chunk assembler.

use harness_core::{
    Content, Message, ModelError, Request, Response, Role, StopReason, StreamEvent, ToolChoice,
    Usage,
};
use serde_json::{json, Value};

pub fn to_body(req: &Request, model: &str, stream: bool) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = &req.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    for msg in &req.messages {
        messages.extend(message_to_wire(msg));
    }

    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });
    if stream {
        // Ask for a final usage chunk; compatible servers that don't support
        // this option generally ignore it.
        body["stream_options"] = json!({"include_usage": true});
    }
    if req.max_tokens > 0 {
        body["max_tokens"] = json!(req.max_tokens);
    }
    if let Some(temp) = req.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(top_p) = req.top_p {
        body["top_p"] = json!(top_p);
    }
    if !req.stop_sequences.is_empty() {
        body["stop"] = json!(req.stop_sequences);
    }
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Any => json!("required"),
            ToolChoice::Tool(name) => {
                json!({"type": "function", "function": {"name": name}})
            }
        };
    }
    body
}

/// One harness message can become several wire messages: tool results are
/// separate `role: "tool"` messages in this dialect.
fn message_to_wire(msg: &Message) -> Vec<Value> {
    match msg.role {
        Role::Assistant => {
            let text = msg.text();
            let tool_calls: Vec<Value> = msg
                .tool_uses()
                .map(|(id, name, input)| {
                    json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": input.to_string()},
                    })
                })
                .collect();
            let mut m = json!({"role": "assistant"});
            m["content"] = if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            };
            if !tool_calls.is_empty() {
                m["tool_calls"] = json!(tool_calls);
            }
            vec![m]
        }
        Role::User => {
            let mut out = Vec::new();
            let mut text = String::new();
            for block in &msg.content {
                match block {
                    Content::Text { text: t } => text.push_str(t),
                    Content::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let content = if *is_error {
                            format!("ERROR: {content}")
                        } else {
                            content.clone()
                        };
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content,
                        }));
                    }
                    Content::ToolUse { .. } | Content::Thinking { .. } => {}
                }
            }
            if !text.is_empty() {
                out.push(json!({"role": "user", "content": text}));
            }
            out
        }
    }
}

pub fn parse_stop_reason(value: Option<&str>) -> StopReason {
    match value {
        Some("stop") | None => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_string()),
    }
}

pub fn parse_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
    }
}

fn parse_arguments(arguments: &str) -> Result<Value, ModelError> {
    if arguments.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    serde_json::from_str(arguments)
        .map_err(|e| ModelError::Deserialize(format!("unparseable tool arguments: {e}")))
}

/// Parse a complete (non-streaming) Chat Completions response.
pub fn parse_response(value: &Value) -> Result<Response, ModelError> {
    let choice = value["choices"]
        .get(0)
        .ok_or_else(|| ModelError::Deserialize("response has no choices".into()))?;
    let wire = &choice["message"];
    let mut content = Vec::new();
    if let Some(text) = wire["content"].as_str() {
        if !text.is_empty() {
            content.push(Content::Text {
                text: text.to_string(),
            });
        }
    }
    if let Some(calls) = wire["tool_calls"].as_array() {
        for (i, call) in calls.iter().enumerate() {
            content.push(Content::ToolUse {
                id: call["id"]
                    .as_str()
                    .unwrap_or(&format!("call_{i}"))
                    .to_string(),
                name: call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                input: parse_arguments(call["function"]["arguments"].as_str().unwrap_or_default())?,
            });
        }
    }
    Ok(Response {
        message: Message {
            role: Role::Assistant,
            content,
        },
        stop_reason: parse_stop_reason(choice["finish_reason"].as_str()),
        usage: parse_usage(&value["usage"]),
    })
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
    started_emitted: bool,
}

/// Folds streaming chunks into typed [`StreamEvent`]s and the final
/// [`Response`]. Chat Completions streams have no explicit terminator event
/// carrying the message, so [`ChunkAssembler::finish`] builds it when the
/// `[DONE]` sentinel (or end of stream) is reached.
#[derive(Default)]
pub struct ChunkAssembler {
    text: String,
    tool_calls: Vec<PartialToolCall>, // indexed by the wire `index`
    finish_reason: Option<String>,
    usage: Usage,
    finished: bool,
}

impl ChunkAssembler {
    pub fn handle(&mut self, chunk: &Value) -> Result<Vec<StreamEvent>, ModelError> {
        let mut out = Vec::new();
        // The final usage-only chunk has an empty `choices` array.
        let usage = &chunk["usage"];
        if usage.is_object() {
            self.usage = parse_usage(usage);
        }
        let Some(choice) = chunk["choices"].get(0) else {
            return Ok(out);
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                self.text.push_str(text);
                out.push(StreamEvent::TextDelta(text.to_string()));
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                let index = call["index"].as_u64().unwrap_or(0) as usize;
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(PartialToolCall::default());
                }
                let partial = &mut self.tool_calls[index];
                if let Some(id) = call["id"].as_str() {
                    partial.id = id.to_string();
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    partial.name.push_str(name);
                }
                if !partial.started_emitted && !partial.name.is_empty() {
                    partial.started_emitted = true;
                    if partial.id.is_empty() {
                        partial.id = format!("call_{index}");
                    }
                    out.push(StreamEvent::ToolUseStart {
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                    });
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    if !args.is_empty() {
                        partial.arguments.push_str(args);
                        out.push(StreamEvent::InputJsonDelta(args.to_string()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Build the final `Completed` event. Returns `None` if already finished.
    pub fn finish(&mut self) -> Option<StreamEvent> {
        if self.finished {
            return None;
        }
        self.finished = true;
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(Content::Text {
                text: std::mem::take(&mut self.text),
            });
        }
        for (i, call) in self.tool_calls.drain(..).enumerate() {
            let input =
                parse_arguments(&call.arguments).unwrap_or(Value::Object(Default::default()));
            content.push(Content::ToolUse {
                id: if call.id.is_empty() {
                    format!("call_{i}")
                } else {
                    call.id
                },
                name: call.name,
                input,
            });
        }
        Some(StreamEvent::Completed(Response {
            message: Message {
                role: Role::Assistant,
                content,
            },
            stop_reason: parse_stop_reason(self.finish_reason.as_deref()),
            usage: self.usage,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness_core::ToolSpec;

    #[test]
    fn builds_wire_messages_with_tool_results_split_out() {
        let req = Request {
            system: Some("be brief".into()),
            messages: vec![
                Message::user("hi"),
                Message {
                    role: Role::Assistant,
                    content: vec![
                        Content::Text {
                            text: "checking".into(),
                        },
                        Content::ToolUse {
                            id: "t1".into(),
                            name: "f".into(),
                            input: json!({"a": 1}),
                        },
                    ],
                },
                Message {
                    role: Role::User,
                    content: vec![Content::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "42".into(),
                        is_error: true,
                    }],
                },
            ],
            tools: vec![ToolSpec {
                name: "f".into(),
                description: "d".into(),
                input_schema: json!({"type": "object"}),
            }],
            tool_choice: Some(ToolChoice::Any),
            max_tokens: 100,
            temperature: None,
            top_p: None,
            stop_sequences: vec![],
        };
        let body = to_body(&req, "test-model", false);
        let messages = body["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "checking");
        assert_eq!(messages[2]["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"a\":1}"
        );
        // Tool results become separate role:"tool" messages.
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "t1");
        assert!(messages[3]["content"]
            .as_str()
            .unwrap()
            .starts_with("ERROR:"));
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"][0]["function"]["name"], "f");
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn assembles_streamed_chunks_into_response() {
        let mut asm = ChunkAssembler::default();
        let chunks = vec![
            json!({"choices": [{"delta": {"content": "Let me "}}]}),
            json!({"choices": [{"delta": {"content": "look."}}]}),
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_9", "function": {"name": "f", "arguments": ""}}
            ]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"a\""}}
            ]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ":1}"}}
            ]}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}),
        ];

        let mut text = String::new();
        let mut saw_start = false;
        for chunk in chunks {
            for ev in asm.handle(&chunk).unwrap() {
                match ev {
                    StreamEvent::TextDelta(t) => text.push_str(&t),
                    StreamEvent::ToolUseStart { id, name } => {
                        saw_start = true;
                        assert_eq!(id, "call_9");
                        assert_eq!(name, "f");
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(text, "Let me look.");
        assert!(saw_start);

        let StreamEvent::Completed(resp) = asm.finish().unwrap() else {
            panic!("finish must yield Completed");
        };
        assert!(asm.finish().is_none(), "finish is idempotent");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 11);
        assert_eq!(resp.usage.output_tokens, 7);
        assert_eq!(resp.message.text(), "Let me look.");
        match &resp.message.content[1] {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "call_9");
                assert_eq!(name, "f");
                assert_eq!(input["a"], 1);
            }
            other => panic!("expected tool use, got {other:?}"),
        }
    }

    #[test]
    fn parses_non_streaming_response() {
        let value = json!({
            "choices": [{
                "message": {
                    "content": "hello",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}
        });
        let resp = parse_response(&value).unwrap();
        assert_eq!(resp.message.text(), "hello");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.message.tool_uses().count(), 1);
    }
}
