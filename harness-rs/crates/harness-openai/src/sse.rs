//! SSE decoding for Chat Completions streams (`data: {...}` / `data: [DONE]`).

use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use harness_core::ModelError;
use serde_json::Value;

pub fn events(
    bytes: impl Stream<Item = reqwest::Result<bytes::Bytes>> + Send,
) -> impl Stream<Item = Result<Value, ModelError>> + Send {
    bytes.eventsource().filter_map(|event| async {
        match event {
            Ok(ev) => {
                let data = ev.data.trim();
                if data.is_empty() || data == "[DONE]" {
                    None
                } else {
                    Some(
                        serde_json::from_str::<Value>(data)
                            .map_err(|e| ModelError::Stream(format!("bad SSE payload: {e}"))),
                    )
                }
            }
            Err(e) => Some(Err(ModelError::Stream(e.to_string()))),
        }
    })
}
