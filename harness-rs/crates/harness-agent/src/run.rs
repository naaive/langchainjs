use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc;

use crate::event::{AgentError, AgentEvent, Decision, RunResult};

/// A live agent run: a `Stream` of [`AgentEvent`]s that is also the handle
/// for answering approval requests. Dropping it cancels the run.
pub struct AgentRun {
    events: mpsc::Receiver<Result<AgentEvent, AgentError>>,
    decisions: mpsc::Sender<Decision>,
}

impl AgentRun {
    pub(crate) fn new(
        events: mpsc::Receiver<Result<AgentEvent, AgentError>>,
        decisions: mpsc::Sender<Decision>,
    ) -> Self {
        Self { events, decisions }
    }

    /// Next event, or `None` once the run has terminated.
    pub async fn next_event(&mut self) -> Option<Result<AgentEvent, AgentError>> {
        self.events.recv().await
    }

    /// Answer the pending [`AgentEvent::AwaitingApproval`] request.
    pub async fn decide(&self, decision: Decision) {
        let _ = self.decisions.send(decision).await;
    }

    /// Drive the run to completion, discarding intermediate events. Approval
    /// requests are auto-denied — attach an interactive consumer (iterate the
    /// stream) if the run may need approvals.
    pub async fn wait(mut self) -> Result<RunResult, AgentError> {
        while let Some(event) = self.events.recv().await {
            match event? {
                AgentEvent::AwaitingApproval(_) => {
                    self.decide(Decision::Deny(Some(
                        "no interactive approver attached; call denied".into(),
                    )))
                    .await;
                }
                AgentEvent::Done(result) => return Ok(result),
                _ => {}
            }
        }
        Err(AgentError::Abandoned(
            "event stream ended without a Done event".into(),
        ))
    }
}

impl Stream for AgentRun {
    type Item = Result<AgentEvent, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}
