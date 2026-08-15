//! SSE event parsing and incremental response assembly for the streaming API.

use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use harness_core::{Content, Message, ModelError, Response, Role, StreamEvent, Usage};
use serde_json::Value;

use crate::convert;

/// Decode an HTTP byte stream into parsed SSE JSON payloads.
pub fn events(
    bytes: impl Stream<Item = reqwest::Result<bytes::Bytes>> + Send,
) -> impl Stream<Item = Result<Value, ModelError>> + Send {
    bytes.eventsource().filter_map(|event| async {
        match event {
            Ok(ev) => {
                if ev.data.is_empty() {
                    None
                } else {
                    Some(
                        serde_json::from_str::<Value>(&ev.data)
                            .map_err(|e| ModelError::Stream(format!("bad SSE payload: {e}"))),
                    )
                }
            }
            Err(e) => Some(Err(ModelError::Stream(e.to_string()))),
        }
    })
}

/// Folds the Messages API streaming event sequence into typed [`StreamEvent`]s
/// and, at `message_stop`, the fully assembled [`Response`].
#[derive(Default)]
pub struct Assembler {
    blocks: Vec<Content>,
    /// Accumulated partial JSON per tool_use block index.
    tool_json: Vec<(usize, String)>,
    stop_reason: Option<String>,
    usage: Usage,
}

impl Assembler {
    pub fn handle(&mut self, event: Value) -> Result<Vec<StreamEvent>, ModelError> {
        let kind = event["type"].as_str().unwrap_or_default();
        match kind {
            "message_start" => {
                self.usage
                    .add(convert::parse_usage(&event["message"]["usage"]));
                Ok(vec![])
            }
            "content_block_start" => {
                let block = &event["content_block"];
                let index = event["index"].as_u64().unwrap_or(0) as usize;
                let mut out = Vec::new();
                if let Some(content) = convert::parse_content_block(block) {
                    if let Content::ToolUse { id, name, .. } = &content {
                        out.push(StreamEvent::ToolUseStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                        self.tool_json.push((index, String::new()));
                    }
                    while self.blocks.len() <= index {
                        self.blocks.push(Content::Text {
                            text: String::new(),
                        });
                    }
                    self.blocks[index] = content;
                }
                Ok(out)
            }
            "content_block_delta" => {
                let index = event["index"].as_u64().unwrap_or(0) as usize;
                let delta = &event["delta"];
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        if let Some(Content::Text { text: t }) = self.blocks.get_mut(index) {
                            t.push_str(text);
                        }
                        Ok(vec![StreamEvent::TextDelta(text.to_string())])
                    }
                    "thinking_delta" => {
                        let text = delta["thinking"].as_str().unwrap_or_default();
                        if let Some(Content::Thinking { thinking }) = self.blocks.get_mut(index) {
                            thinking.push_str(text);
                        }
                        Ok(vec![StreamEvent::ThinkingDelta(text.to_string())])
                    }
                    "input_json_delta" => {
                        let partial = delta["partial_json"].as_str().unwrap_or_default();
                        if let Some((_, buf)) = self.tool_json.iter_mut().find(|(i, _)| *i == index)
                        {
                            buf.push_str(partial);
                        }
                        Ok(vec![StreamEvent::InputJsonDelta(partial.to_string())])
                    }
                    _ => Ok(vec![]),
                }
            }
            "content_block_stop" => {
                let index = event["index"].as_u64().unwrap_or(0) as usize;
                if let Some(pos) = self.tool_json.iter().position(|(i, _)| *i == index) {
                    let (_, buf) = self.tool_json.remove(pos);
                    let input: Value = if buf.trim().is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(&buf).map_err(|e| {
                            ModelError::Stream(format!("unparseable tool input JSON: {e}"))
                        })?
                    };
                    if let Some(Content::ToolUse { input: i, .. }) = self.blocks.get_mut(index) {
                        *i = input;
                    }
                }
                Ok(vec![])
            }
            "message_delta" => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(reason.to_string());
                }
                self.usage.add(convert::parse_usage(&event["usage"]));
                Ok(vec![])
            }
            "message_stop" => {
                let response = Response {
                    message: Message {
                        role: Role::Assistant,
                        content: std::mem::take(&mut self.blocks),
                    },
                    stop_reason: convert::parse_stop_reason(self.stop_reason.as_deref()),
                    usage: self.usage,
                };
                Ok(vec![StreamEvent::Completed(response)])
            }
            "error" => Err(ModelError::Api {
                status: 0,
                message: event["error"].to_string(),
            }),
            // "ping" and unknown event types.
            _ => Ok(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn assembles_text_and_tool_use_from_event_sequence() {
        let mut asm = Assembler::default();
        let events = vec![
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 25}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I'll check"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " the weather."}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Tokyo\"}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
            json!({"type": "message_stop"}),
        ];

        let mut text = String::new();
        let mut completed = None;
        for ev in events {
            for out in asm.handle(ev).unwrap() {
                match out {
                    StreamEvent::TextDelta(t) => text.push_str(&t),
                    StreamEvent::Completed(r) => completed = Some(r),
                    _ => {}
                }
            }
        }

        assert_eq!(text, "I'll check the weather.");
        let resp = completed.expect("message_stop must yield Completed");
        assert_eq!(resp.stop_reason, harness_core::StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 25);
        assert_eq!(resp.usage.output_tokens, 12);
        assert_eq!(resp.message.content.len(), 2);
        match &resp.message.content[1] {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "Tokyo");
            }
            other => panic!("expected tool use, got {other:?}"),
        }
    }

    #[test]
    fn error_event_fails_the_stream() {
        let mut asm = Assembler::default();
        let err = asm
            .handle(json!({"type": "error", "error": {"type": "overloaded_error"}}))
            .unwrap_err();
        assert!(matches!(err, ModelError::Api { .. }));
    }
}
