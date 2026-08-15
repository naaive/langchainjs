use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use harness_core::{
    ChatModel, Content, Message, ModelError, Request, Response, Role, StopReason, StreamEvent,
    Tool, ToolContext, ToolOutput, ToolSpec, Usage,
};
use tokio::sync::mpsc;

use crate::event::{AgentError, AgentEvent, ApprovalRequest, Decision, RunResult};
use crate::middleware::{Middleware, ToolCall, ToolNext};
use crate::run::AgentRun;

/// Retry behavior for transient model failures (rate limits, 5xx, transport).
/// A call is only retried while nothing of it has been streamed to the
/// consumer yet, so the UI never sees duplicated output.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
        }
    }
}

/// A configured agent: model + system prompt + tools + middleware. Cheap to
/// clone; holds no per-conversation state. Conversation history lives with
/// the caller and travels through [`Agent::run`] / [`RunResult::messages`].
#[derive(Clone)]
pub struct Agent {
    inner: Arc<Inner>,
}

struct Inner {
    model: Arc<dyn ChatModel>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    middlewares: Vec<Arc<dyn Middleware>>,
    max_turns: u32,
    max_tokens: u32,
    temperature: Option<f32>,
    retry: RetryPolicy,
    ctx: ToolContext,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Start a run: `history` is the prior conversation (empty for a fresh
    /// one), `input` the new user message. Returns immediately with an
    /// [`AgentRun`] — a stream of [`AgentEvent`]s ending in `Done`.
    pub fn run(&self, history: Vec<Message>, input: impl Into<String>) -> AgentRun {
        let mut messages = history;
        messages.push(Message::user(input));
        self.resume(messages)
    }

    /// Continue from a raw message history (e.g. one restored from a
    /// checkpoint) without appending a new user message.
    pub fn resume(&self, messages: Vec<Message>) -> AgentRun {
        let (event_tx, event_rx) = mpsc::channel(256);
        let (decision_tx, decision_rx) = mpsc::channel(8);
        let inner = self.inner.clone();
        tokio::spawn(async move {
            if let Err(e) = inner.drive(messages, &event_tx, decision_rx).await {
                let _ = event_tx.send(Err(e)).await;
            }
        });
        AgentRun::new(event_rx, decision_tx)
    }
}

type EventSender = mpsc::Sender<Result<AgentEvent, AgentError>>;

impl Inner {
    async fn drive(
        &self,
        mut messages: Vec<Message>,
        events: &EventSender,
        mut decisions: mpsc::Receiver<Decision>,
    ) -> Result<(), AgentError> {
        let specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();
        let mut usage = Usage::default();

        for turn in 1..=self.max_turns {
            let mut req = Request {
                system: self.system.clone(),
                messages: messages.clone(),
                tools: specs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
            };
            for mw in &self.middlewares {
                mw.before_model_call(&mut req).await;
            }

            let resp = self.call_model(req, events).await?;
            tracing::debug!(turn, stop_reason = ?resp.stop_reason, "model call completed");
            usage.add(resp.usage);
            messages.push(resp.message.clone());

            if resp.stop_reason != StopReason::ToolUse {
                let final_text = resp.message.text();
                emit(
                    events,
                    AgentEvent::Done(RunResult {
                        messages,
                        final_text,
                        usage,
                        turns: turn,
                    }),
                )
                .await?;
                return Ok(());
            }

            let calls: Vec<ToolCall> = resp
                .message
                .tool_uses()
                .map(|(id, name, input)| ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    input: input.clone(),
                })
                .collect();

            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                emit(
                    events,
                    AgentEvent::ToolCallStarted {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        input: call.input.clone(),
                    },
                )
                .await?;

                let output = self.execute_tool(&call, events, &mut decisions).await?;

                emit(
                    events,
                    AgentEvent::ToolCallFinished {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        output: output.clone(),
                    },
                )
                .await?;

                results.push(Content::ToolResult {
                    tool_use_id: call.id,
                    content: output.content,
                    is_error: output.is_error,
                });
            }
            messages.push(Message {
                role: Role::User,
                content: results,
            });
        }

        Err(AgentError::MaxTurns(self.max_turns))
    }

    /// One model call: stream deltas out as events, retry with backoff while
    /// the failure is transient and nothing has been emitted yet.
    async fn call_model(&self, req: Request, events: &EventSender) -> Result<Response, AgentError> {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let mut emitted = false;
            let mut stream = self.model.stream(req.clone());
            let mut failure: Option<ModelError> = None;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(StreamEvent::TextDelta(t)) => {
                        emitted = true;
                        emit(events, AgentEvent::TextDelta(t)).await?;
                    }
                    Ok(StreamEvent::ThinkingDelta(t)) => {
                        emitted = true;
                        emit(events, AgentEvent::ThinkingDelta(t)).await?;
                    }
                    // Tool input deltas aren't forwarded; the complete call is
                    // surfaced as ToolCallStarted once assembled.
                    Ok(StreamEvent::ToolUseStart { .. }) | Ok(StreamEvent::InputJsonDelta(_)) => {}
                    Ok(StreamEvent::Completed(resp)) => return Ok(resp),
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }

            let error = failure
                .unwrap_or_else(|| ModelError::Stream("stream ended without completion".into()));
            let retryable = !emitted && error.is_retryable() && attempt < self.retry.max_attempts;
            if !retryable {
                return Err(error.into());
            }
            let delay = self.retry.base_delay * 2u32.saturating_pow(attempt - 1);
            tracing::debug!(attempt, ?delay, error = %error, "retrying model call");
            tokio::time::sleep(delay).await;
        }
    }

    async fn execute_tool(
        &self,
        call: &ToolCall,
        events: &EventSender,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Result<ToolOutput, AgentError> {
        let Some(tool) = self.tools.iter().find(|t| t.name() == call.name) else {
            return Ok(ToolOutput::error(format!("unknown tool: {}", call.name)));
        };

        if tool.needs_approval() {
            emit(
                events,
                AgentEvent::AwaitingApproval(ApprovalRequest {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.input.clone(),
                }),
            )
            .await?;
            match decisions.recv().await {
                Some(Decision::Approve) => {}
                Some(Decision::Deny(reason)) => {
                    return Ok(ToolOutput::error(
                        reason.unwrap_or_else(|| "denied by user".into()),
                    ));
                }
                None => {
                    return Err(AgentError::Abandoned(
                        "approval requested but the run was dropped".into(),
                    ))
                }
            }
        }

        let next = ToolNext {
            chain: &self.middlewares,
            tool: tool.as_ref(),
            ctx: &self.ctx,
        };
        // Tool failures are data for the model, not run-fatal errors.
        Ok(next
            .run(call.clone())
            .await
            .unwrap_or_else(|e| ToolOutput::error(e.to_string())))
    }
}

async fn emit(events: &EventSender, event: AgentEvent) -> Result<(), AgentError> {
    events
        .send(Ok(event))
        .await
        .map_err(|_| AgentError::Abandoned("event consumer dropped".into()))
}

pub struct AgentBuilder {
    model: Option<Arc<dyn ChatModel>>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    middlewares: Vec<Arc<dyn Middleware>>,
    max_turns: u32,
    max_tokens: u32,
    temperature: Option<f32>,
    retry: RetryPolicy,
    ctx: ToolContext,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            model: None,
            system: None,
            tools: Vec::new(),
            middlewares: Vec::new(),
            max_turns: 50,
            max_tokens: 0,
            temperature: None,
            retry: RetryPolicy::default(),
            ctx: ToolContext::default(),
        }
    }
}

impl AgentBuilder {
    pub fn model(mut self, model: impl ChatModel + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn middleware(mut self, mw: impl Middleware + 'static) -> Self {
        self.middlewares.push(Arc::new(mw));
        self
    }

    /// Safety cap on model calls per run (default 50).
    pub fn max_turns(mut self, n: u32) -> Self {
        self.max_turns = n;
        self
    }

    /// Max generated tokens per model call (0 = provider default).
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    pub fn tool_context(mut self, ctx: ToolContext) -> Self {
        self.ctx = ctx;
        self
    }

    /// # Panics
    /// If no model was configured.
    pub fn build(self) -> Agent {
        Agent {
            inner: Arc::new(Inner {
                model: self.model.expect("Agent requires a model"),
                system: self.system,
                tools: self.tools,
                middlewares: self.middlewares,
                max_turns: self.max_turns,
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                retry: self.retry,
                ctx: self.ctx,
            }),
        }
    }
}
