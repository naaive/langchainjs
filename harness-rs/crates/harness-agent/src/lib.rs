//! The harness agent runtime.
//!
//! An agent here is not a graph: it is a productized version of the loop
//! everyone writes by hand — call the model, execute requested tools, feed
//! results back, repeat until the model stops asking. The runtime adds the
//! parts that are genuinely hard to hand-roll well:
//!
//! - a **typed event stream** ([`AgentEvent`]) covering the whole run, so one
//!   API serves both "just give me the answer" and "render every token";
//! - **human-in-the-loop approval** as a first-class state, not a callback;
//! - **middleware** for cross-cutting concerns on model and tool calls;
//! - **retry** with exponential backoff for transient model failures;
//! - a **serializable message history**, so persistence is `serde`, not an
//!   execution engine.
//!
//! Orchestration across agents (branching, parallelism, hand-offs) is left to
//! ordinary Rust control flow in the caller's code — `match` is the condition
//! edge, `join!` is the parallel node.

mod agent;
mod agent_tool;
mod checkpoint;
mod compaction;
mod event;
mod middleware;
mod run;

pub use agent::{Agent, AgentBuilder, RetryPolicy};
pub use agent_tool::AgentTool;
pub use checkpoint::{CheckpointError, Checkpointer, FileCheckpointer, MemoryCheckpointer};
pub use compaction::Compaction;
pub use event::{AgentError, AgentEvent, ApprovalRequest, Decision, RunResult};
pub use middleware::{Middleware, ToolCall, ToolNext};
pub use run::AgentRun;
