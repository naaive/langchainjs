//! Context compaction: summarize older conversation history when it grows
//! past a budget, as a [`Middleware`](crate::Middleware).
//!
//! The canonical history the agent owns is never mutated — compaction rewrites
//! the *request* on its way to the model, replacing an old prefix of messages
//! with a model-written summary. The summary is cached and reused until the
//! (still growing) suffix exceeds the budget again.

use std::sync::Arc;

use harness_core::{ChatModel, Content, Message, Request, Role};
use tokio::sync::Mutex;

use crate::middleware::Middleware;

fn summary_message(text: &str) -> Message {
    Message::user(format!("[Summary of earlier conversation]\n{text}"))
}

fn apply(req: &mut Request, summary: Message, covered: usize) {
    let mut compacted = vec![summary];
    compacted.extend_from_slice(&req.messages[covered..]);
    tracing::debug!(
        original = req.messages.len(),
        compacted = compacted.len(),
        "compacted conversation history"
    );
    req.messages = compacted;
}

/// Rough token estimate: ~4 chars per token, on serialized content length.
fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0) / 4)
        .sum()
}

/// Summarizes the oldest part of the conversation once the estimated token
/// count exceeds `budget_tokens`, keeping at least `keep_recent` messages
/// verbatim. Construct with the (possibly cheaper) model that writes the
/// summaries.
pub struct Compaction {
    model: Arc<dyn ChatModel>,
    budget_tokens: usize,
    keep_recent: usize,
    /// Cached summary: (number of original messages it covers, summary text).
    cache: Mutex<Option<(usize, String)>>,
}

impl Compaction {
    pub fn new(model: impl ChatModel + 'static, budget_tokens: usize) -> Self {
        Self {
            model: Arc::new(model),
            budget_tokens,
            keep_recent: 8,
            cache: Mutex::new(None),
        }
    }

    /// Minimum number of most-recent messages kept verbatim (default 8).
    pub fn keep_recent(mut self, n: usize) -> Self {
        self.keep_recent = n;
        self
    }

    /// Pick how many leading messages to fold into the summary. The cut must
    /// land on a `user` message that isn't a tool result, so no
    /// tool_use/tool_result pair is ever split across the boundary.
    fn split_point(&self, messages: &[Message]) -> Option<usize> {
        let latest_allowed = messages.len().saturating_sub(self.keep_recent.max(1));
        (1..=latest_allowed)
            .rev()
            .find(|&i| {
                messages[i].role == Role::User
                    && !matches!(
                        messages[i].content.first(),
                        Some(Content::ToolResult { .. })
                    )
            })
            .filter(|&i| i > 0)
    }

    async fn summarize(&self, messages: &[Message]) -> Result<String, harness_core::ModelError> {
        let transcript = serde_json::to_string_pretty(messages).unwrap_or_default();
        let req = Request {
            system: Some(
                "You compress agent conversation history. Summarize the transcript into a \
                 dense brief a coding agent can resume from: user goals, decisions made, \
                 files/entities touched with their current state, tool call outcomes, and \
                 unresolved threads. No preamble."
                    .into(),
            ),
            messages: vec![Message::user(transcript)],
            ..Default::default()
        };
        Ok(self.model.generate(req).await?.message.text())
    }
}

#[harness_core::async_trait]
impl Middleware for Compaction {
    async fn before_model_call(&self, req: &mut Request) {
        if estimate_tokens(&req.messages) <= self.budget_tokens {
            return;
        }
        let Some(split) = self.split_point(&req.messages) else {
            return;
        };

        let mut cache = self.cache.lock().await;
        // History is append-only, so a previously summarized prefix is still
        // a valid summary. Reuse it when the compacted result would fit the
        // budget; only re-summarize (at a later split) when it wouldn't.
        if let Some((covered, text)) = cache.as_ref() {
            if *covered > 0 && *covered <= split {
                let summary_msg = summary_message(text);
                let projected = estimate_tokens(std::slice::from_ref(&summary_msg))
                    + estimate_tokens(&req.messages[*covered..]);
                if projected <= self.budget_tokens {
                    apply(req, summary_msg, *covered);
                    return;
                }
            }
        }

        match self.summarize(&req.messages[..split]).await {
            Ok(text) => {
                *cache = Some((split, text.clone()));
                apply(req, summary_message(&text), split);
            }
            Err(e) => {
                // A failed summary must not take the run down; send the
                // request through uncompacted.
                tracing::warn!(error = %e, "context compaction failed; sending full history");
            }
        }
    }
}
