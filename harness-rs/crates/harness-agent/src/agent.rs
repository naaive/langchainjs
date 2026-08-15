use std::sync::Arc;
use std::time::Duration;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use harness_core::{
    ChatModel, Content, Message, ModelError, Request, Response, Role, StopReason, StreamEvent,
    Tool, ToolContext, ToolOutput, ToolSpec, Usage,
};
use tokio::sync::mpsc;
use tracing::Instrument;

use crate::checkpoint::Checkpointer;
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
/// the caller ([`Agent::run`] / [`RunResult::messages`]) or, for durable
/// sessions, with the configured [`Checkpointer`] ([`Agent::run_session`]).
#[derive(Clone)]
pub struct Agent {
    inner: Arc<Inner>,
}

struct Inner {
    model: Arc<dyn ChatModel>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    middlewares: Vec<Arc<dyn Middleware>>,
    checkpointer: Option<Arc<dyn Checkpointer>>,
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
        self.spawn(SessionSource::Messages(messages))
    }

    /// Continue from a raw message history (e.g. one restored from a
    /// checkpoint) without appending a new user message.
    pub fn resume(&self, messages: Vec<Message>) -> AgentRun {
        self.spawn(SessionSource::Resume(messages))
    }

    /// Run against a durable session: history is loaded from the configured
    /// [`Checkpointer`] under `session`, and every append (assistant message,
    /// tool results) is persisted before the run proceeds — a crash loses at
    /// most the model call in flight.
    ///
    /// Requires a checkpointer; the run fails with [`AgentError::Config`]
    /// otherwise.
    pub fn run_session(&self, session: impl Into<String>, input: impl Into<String>) -> AgentRun {
        self.spawn(SessionSource::Checkpoint {
            session: session.into(),
            input: input.into(),
        })
    }

    fn spawn(&self, source: SessionSource) -> AgentRun {
        let (event_tx, event_rx) = mpsc::channel(256);
        let (decision_tx, decision_rx) = mpsc::channel(8);
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let outcome = async {
                let (messages, session) = match source {
                    SessionSource::Messages(m) | SessionSource::Resume(m) => (m, None),
                    SessionSource::Checkpoint { session, input } => {
                        let Some(cp) = &inner.checkpointer else {
                            return Err(AgentError::Config(
                                "run_session requires a checkpointer".into(),
                            ));
                        };
                        let mut messages = cp.load(&session).await?.unwrap_or_default();
                        messages.push(Message::user(input));
                        (messages, Some(session))
                    }
                };
                inner
                    .drive(messages, &event_tx, decision_rx, session.as_deref())
                    .await
            }
            .await;
            if let Err(e) = outcome {
                let _ = event_tx.send(Err(e)).await;
            }
        });
        AgentRun::new(event_rx, decision_tx)
    }
}

enum SessionSource {
    Messages(Vec<Message>),
    Resume(Vec<Message>),
    Checkpoint { session: String, input: String },
}

type EventSender = mpsc::Sender<Result<AgentEvent, AgentError>>;

impl Inner {
    async fn drive(
        &self,
        mut messages: Vec<Message>,
        events: &EventSender,
        mut decisions: mpsc::Receiver<Decision>,
        session: Option<&str>,
    ) -> Result<(), AgentError> {
        let specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();
        let mut usage = Usage::default();
        self.checkpoint(session, &messages).await?;

        for turn in 1..=self.max_turns {
            let mut req = Request {
                system: self.system.clone(),
                messages: messages.clone(),
                tools: specs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                ..Default::default()
            };
            for mw in &self.middlewares {
                mw.before_model_call(&mut req).await;
            }

            let span = tracing::info_span!(
                "model_call",
                turn,
                gen_ai.operation.name = "chat",
                gen_ai.request.model = self.model.model_id(),
            );
            let resp = self.call_model(req, events).instrument(span).await?;
            tracing::debug!(
                turn,
                stop_reason = ?resp.stop_reason,
                input_tokens = resp.usage.input_tokens,
                output_tokens = resp.usage.output_tokens,
                "model call completed"
            );
            usage.add(resp.usage);
            messages.push(resp.message.clone());
            self.checkpoint(session, &messages).await?;

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

            let results = self.execute_tools(calls, events, &mut decisions).await?;
            messages.push(Message {
                role: Role::User,
                content: results,
            });
            self.checkpoint(session, &messages).await?;
        }

        Err(AgentError::MaxTurns(self.max_turns))
    }

    async fn checkpoint(
        &self,
        session: Option<&str>,
        messages: &[Message],
    ) -> Result<(), AgentError> {
        if let (Some(cp), Some(session)) = (&self.checkpointer, session) {
            cp.save(session, messages).await?;
        }
        Ok(())
    }

    /// Execute one batch of tool calls. Calls without an approval gate run
    /// concurrently (`ToolCallFinished` is emitted as each completes);
    /// approval-gated calls then run sequentially, since each blocks on a
    /// human decision. Results are returned in the model's original order.
    async fn execute_tools(
        &self,
        calls: Vec<ToolCall>,
        events: &EventSender,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Result<Vec<Content>, AgentError> {
        for call in &calls {
            emit(
                events,
                AgentEvent::ToolCallStarted {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.input.clone(),
                },
            )
            .await?;
        }

        let needs_approval: Vec<bool> = calls
            .iter()
            .map(|c| {
                self.find_tool(&c.name)
                    .map(|t| t.needs_approval())
                    .unwrap_or(false)
            })
            .collect();
        let mut outputs: Vec<Option<ToolOutput>> = calls.iter().map(|_| None).collect();

        let mut concurrent = FuturesUnordered::new();
        for (i, call) in calls.iter().enumerate() {
            if !needs_approval[i] {
                concurrent.push(async move { (i, self.run_tool(call).await) });
            }
        }
        while let Some((i, output)) = concurrent.next().await {
            emit(
                events,
                AgentEvent::ToolCallFinished {
                    id: calls[i].id.clone(),
                    name: calls[i].name.clone(),
                    output: output.clone(),
                },
            )
            .await?;
            outputs[i] = Some(output);
        }
        drop(concurrent);

        for (i, call) in calls.iter().enumerate() {
            if !needs_approval[i] {
                continue;
            }
            let output = self.execute_with_approval(call, events, decisions).await?;
            emit(
                events,
                AgentEvent::ToolCallFinished {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    output: output.clone(),
                },
            )
            .await?;
            outputs[i] = Some(output);
        }

        Ok(calls
            .into_iter()
            .zip(outputs)
            .map(|(call, output)| {
                let output = output.expect("every tool call produces an output");
                Content::ToolResult {
                    tool_use_id: call.id,
                    content: output.content,
                    is_error: output.is_error,
                }
            })
            .collect())
    }

    async fn execute_with_approval(
        &self,
        call: &ToolCall,
        events: &EventSender,
        decisions: &mut mpsc::Receiver<Decision>,
    ) -> Result<ToolOutput, AgentError> {
        emit(
            events,
            AgentEvent::AwaitingApproval(ApprovalRequest {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.input.clone(),
            }),
        )
        .await?;
        let mut call = call.clone();
        match decisions.recv().await {
            Some(Decision::Approve) => {}
            Some(Decision::Edit(input)) => call.input = input,
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
        Ok(self.run_tool(&call).await)
    }

    /// Run a single tool through the middleware chain. Tool failures are data
    /// for the model, not run-fatal errors.
    async fn run_tool(&self, call: &ToolCall) -> ToolOutput {
        let Some(tool) = self.find_tool(&call.name) else {
            return ToolOutput::error(format!("unknown tool: {}", call.name));
        };
        let span = tracing::info_span!(
            "tool_call",
            gen_ai.operation.name = "execute_tool",
            gen_ai.tool.name = call.name.as_str(),
        );
        async {
            let next = ToolNext {
                chain: &self.middlewares,
                tool: tool.as_ref(),
                ctx: &self.ctx,
            };
            next.run(call.clone())
                .await
                .unwrap_or_else(|e| ToolOutput::error(e.to_string()))
        }
        .instrument(span)
        .await
    }

    fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name)
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
    checkpointer: Option<Arc<dyn Checkpointer>>,
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
            checkpointer: None,
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

    /// Durable session storage, enabling [`Agent::run_session`].
    pub fn checkpointer(mut self, cp: impl Checkpointer + 'static) -> Self {
        self.checkpointer = Some(Arc::new(cp));
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
                checkpointer: self.checkpointer,
                max_turns: self.max_turns,
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                retry: self.retry,
                ctx: self.ctx,
            }),
        }
    }
}
