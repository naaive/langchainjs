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
