# harness-rs

A Rust-native agent framework. Inspired by what LangChain gets right — a
uniform model interface, a rich tool ecosystem, streaming everywhere — with
none of its historical baggage, and no graph engine.

**Design thesis: LangChain uses framework code to compensate for what Python
can't express; harness uses Rust's own language features to shrink the
framework.** Control flow is `loop`/`match`/`join!`, not LCEL combinators or
graph DSLs. Type safety comes from traits and `serde`, not runtime validation.
What remains in the framework is only what a language can't give you:

1. a **uniform, minimal model interface** (`ChatModel`: one streaming method);
2. a **tool system** with compile-time JSON Schema derivation (`#[tool]`);
3. an **agent runtime**: the model⇄tools loop with a typed event stream,
   first-class human-in-the-loop approval, retries, and middleware.

## Layout

```
crates/
  harness-core        Pure traits & types: Message, Content, ChatModel, Tool,
                      structured output. Zero heavy deps.
  harness-macros      #[tool] proc macro: schema + Tool impl derived from an async fn.
  harness-anthropic   Anthropic Messages API provider (SSE streaming, tool use).
  harness-openai      OpenAI-compatible Chat Completions provider
                      (OpenAI, Ollama, vLLM, llama.cpp, gateways).
  harness-agent       Agent runtime: loop, AgentEvent stream, approvals, middleware,
                      retry, checkpointing, context compaction.
examples/
  code-agent          A Claude Code-style terminal coding agent (the flagship example).
```

Dependency direction is strictly downward: `harness-core` knows nothing about
providers or the runtime, so implementing a new provider means depending on one
tiny crate and writing one method.

## Quick tour

Define tools with plain async functions — the JSON Schema is derived from the
signature at compile time, the doc comment becomes the description:

```rust
use harness_core::{ToolContext, ToolError};
use harness_macros::tool;

/// Replace old_string with new_string in a file. old_string must be unique.
#[tool]
async fn edit_file(
    path: String,
    old_string: String,
    new_string: String,
    ctx: &ToolContext,          // injected by the runtime, not in the schema
) -> Result<String, ToolError> { /* ... */ }

/// Run a shell command.
#[tool(approval)]               // pauses the run for human approval
async fn bash(command: String, ctx: &ToolContext) -> Result<String, ToolError> { /* ... */ }
```

Build an agent and drive it as a typed event stream:

```rust
let agent = Agent::builder()
    .model(Anthropic::new("claude-sonnet-5"))
    .system("You are a coding agent.")
    .tools(tools![read_file, edit_file, bash])
    .build();

let mut run = agent.run(history, "fix the failing test");
while let Some(event) = run.next_event().await {
    match event? {
        AgentEvent::TextDelta(t) => print!("{t}"),
        AgentEvent::ToolCallStarted { name, .. } => println!("⚙ {name}"),
        AgentEvent::AwaitingApproval(req) => run.decide(ask_user(&req)).await,
        AgentEvent::Done(result) => history = result.messages,
        _ => {}
    }
}
```

One API serves both "just give me the answer" (`run.wait().await`) and "render
every token" (iterate the stream). Events are an exhaustive enum — no string
event names, no untyped payload dicts. Approval decisions are `Approve`,
`Deny(reason)` — which the model sees as an error result and reacts to — or
`Edit(new_input)` to run the tool with corrected arguments.

Durable sessions need one extra line:

```rust
let agent = Agent::builder()
    .model(model)
    .checkpointer(FileCheckpointer::new(".sessions"))   // or MemoryCheckpointer, or your own
    .build();

agent.run_session("ticket-4711", "continue the migration");
```

`run_session` loads history from the store, and every append (assistant
message, tool results) is persisted before the run proceeds — a crash loses at
most the model call in flight, never completed tool work. Because a checkpoint
is just the serialized message history, any storage that can hold bytes can be
a `Checkpointer` (three methods).

Typed structured output works on any model, with schema-repair retries built
in:

```rust
#[derive(Deserialize, JsonSchema)]
struct Invoice { customer: String, total_cents: u64 }

let invoice: Invoice = model.generate_as(req).await?;
```

When history outgrows the context window, the `Compaction` middleware
summarizes the old prefix with a (possibly cheaper) model and caches the
summary; the agent's canonical history is never mutated:

```rust
.middleware(Compaction::new(Anthropic::new("claude-haiku-4-5"), 150_000))
```

## The flagship example: a Claude Code-style agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -p code-agent
```

An interactive terminal coding agent with `read_file` / `write_file` /
`edit_file` / `list_dir` / `bash`. It streams its reasoning, shows each tool
call as it executes, and pauses for `y/N` approval before running any shell
command — the runtime's `AwaitingApproval` state driving a real UI.
Conversations are checkpointed to `.harness/sessions/`, so quitting and
restarting the binary resumes where you left off (`HARNESS_SESSION` selects
the session, `/clear` wipes it). The whole thing is ~400 lines, most of which
is terminal rendering.

## Architectural positions (vs. LangChain / LangGraph)

| Concern | LangChain / LangGraph | harness |
|---|---|---|
| Orchestration | LCEL combinators / graph DSL | native Rust control flow; the framework ships only the resumable loop |
| Events | string-keyed callbacks | exhaustive `AgentEvent` / `StreamEvent` enums |
| Tool schemas | runtime reflection | proc macro, checked at compile time |
| Human-in-the-loop | graph interrupt | first-class `AwaitingApproval` run state |
| Persistence | graph checkpointer | `Message` is `Serialize`; history round-trips through any store |
| Sync/async split | dual API surface | async only; batching is `futures` composition |
| Multi-agent | subgraphs | agents are values: call them, `join!` them, or wrap one as a `Tool` |

Why no graph engine: a graph DSL re-implements branching, looping and
parallelism as library data structures because Python chains couldn't express
them. In Rust, `match` is the conditional edge and `tokio::join!` is the
parallel node — with the borrow checker and exhaustiveness checking thrown in.
What people actually need from LangGraph — durable, interruptible execution —
is provided by the runtime's serializable history and approval states, not by
a scheduler.

Tool errors are data, not exceptions: a failed tool call becomes an
`is_error: true` result the model can react to, so agents self-correct instead
of crashing.

## Status & roadmap

Working today:

- core traits; `#[tool]` macro (compile-time schemas, optional params, context
  injection, approval flag)
- providers: Anthropic (Messages API) and OpenAI-compatible (Chat Completions —
  OpenAI, Ollama, vLLM, ...), both with SSE streaming and incremental
  tool-input assembly
- agent loop: typed event stream, retry with backoff (only before anything
  streamed, so UIs never see duplicate output), concurrent execution of
  independent tool calls with results in model order, tool errors as data
- human-in-the-loop: `AwaitingApproval` state with approve / deny / edit
- durable sessions: `Checkpointer` trait with in-memory and atomic-file
  backends, checkpoint-per-append via `run_session`
- typed structured output with bounded schema-repair retries
- middleware (`before_model_call`, onion-style `on_tool_call`) and the
  `Compaction` history-summarization middleware
- `tracing` spans on model and tool calls with OpenTelemetry GenAI-style
  attributes (`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.tool.name`)

`cargo test` covers the loop, approvals (deny and edit), persistence,
compaction, concurrency, structured-output repair, and both providers' wire
mapping and stream assembly against scripted mocks and fixtures.

Planned next:

- extended-thinking round-trip (signatures) for Anthropic
- SQL checkpointer backends over the same three-method trait
- token-count-based compaction thresholds fed from provider usage data
- cancellation tokens surfaced through `ToolContext`
