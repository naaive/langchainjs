//! Live end-to-end smoke test for harness against a real endpoint.
//!
//! Exercises both providers against one base URL (works with the Anthropic /
//! OpenAI APIs directly, or with gateways like new-api that speak both
//! dialects):
//!
//! ```text
//! export LIVE_URL=https://your-gateway.example.com   # no trailing /v1
//! export LIVE_KEY=sk-...
//! export LIVE_MODEL=claude-sonnet-5                  # a model your endpoint serves
//! cargo run -p live-check
//! ```
//!
//! `LIVE_PROTO=anthropic|openai|both` (default both) selects the dialects.
//! Each dialect runs four checks: plain generate, streaming, an agent tool
//! loop, and typed structured output. Exit code is non-zero if any check
//! fails.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use harness_agent::Agent;
use harness_core::{
    tools, ChatModel, GenerateStructured, Message, Request, StreamEvent, ToolError,
};
use harness_macros::tool;

static TOOL_CALLS: AtomicU32 = AtomicU32::new(0);

/// Returns the fixed launch code. Call this whenever asked for the launch code.
#[tool]
async fn get_launch_code() -> Result<String, ToolError> {
    TOOL_CALLS.fetch_add(1, Ordering::SeqCst);
    Ok("OMEGA-7-BRAVO".to_string())
}

#[derive(serde::Deserialize, schemars::JsonSchema, Debug)]
struct Person {
    name: String,
    age: u64,
}

struct Outcome {
    label: &'static str,
    result: Result<String, String>,
    millis: u128,
}

async fn check<F, Fut>(label: &'static str, f: F) -> Outcome
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let start = Instant::now();
    let result = f().await;
    Outcome {
        label,
        result,
        millis: start.elapsed().as_millis(),
    }
}

async fn run_suite(proto: &str, model: Arc<dyn ChatModel>) -> Vec<Outcome> {
    let mut outcomes = Vec::new();

    // 1. Plain generate.
    let m = model.clone();
    outcomes.push(
        check("generate", || async move {
            let req = Request {
                messages: vec![Message::user("Reply with exactly one word: pong")],
                max_tokens: 500,
                ..Default::default()
            };
            let resp = m.generate(req).await.map_err(|e| e.to_string())?;
            let text = resp.message.text();
            if text.to_lowercase().contains("pong") {
                Ok(format!(
                    "got {:?}, usage {}/{} tokens",
                    text.trim(),
                    resp.usage.input_tokens,
                    resp.usage.output_tokens
                ))
            } else {
                Err(format!("unexpected reply: {text:?}"))
            }
        })
        .await,
    );

    // 2. Streaming: deltas must arrive and concatenate to the final message.
    let m = model.clone();
    outcomes.push(
        check("streaming", || async move {
            use futures::StreamExt;
            let req = Request {
                messages: vec![Message::user("Count from 1 to 10, digits only.")],
                max_tokens: 500,
                ..Default::default()
            };
            let mut stream = m.stream(req);
            let mut deltas = 0u32;
            let mut collected = String::new();
            let mut completed = None;
            while let Some(ev) = stream.next().await {
                match ev.map_err(|e| e.to_string())? {
                    StreamEvent::TextDelta(t) => {
                        deltas += 1;
                        collected.push_str(&t);
                    }
                    StreamEvent::Completed(r) => completed = Some(r),
                    _ => {}
                }
            }
            let resp = completed.ok_or("stream ended without Completed")?;
            if collected != resp.message.text() {
                return Err("deltas do not concatenate to final text".into());
            }
            if deltas == 0 {
                return Err("no text deltas received".into());
            }
            Ok(format!("{deltas} deltas, final text matches"))
        })
        .await,
    );

    // 3. Agent tool loop: the model must actually call our tool.
    let m = model.clone();
    outcomes.push(
        check("tool loop", || async move {
            TOOL_CALLS.store(0, Ordering::SeqCst);
            let agent = Agent::builder()
                .model(ArcModel(m))
                .system("Use the available tools to answer. Be terse.")
                .tools(tools![get_launch_code])
                .max_turns(4)
                .build();
            let result = agent
                .run(vec![], "What is the launch code? Use the tool, then state it.")
                .wait()
                .await
                .map_err(|e| e.to_string())?;
            if TOOL_CALLS.load(Ordering::SeqCst) == 0 {
                return Err(format!(
                    "model never called the tool; said: {:?}",
                    result.final_text
                ));
            }
            if !result.final_text.contains("OMEGA-7-BRAVO") {
                return Err(format!(
                    "tool result not used in answer: {:?}",
                    result.final_text
                ));
            }
            Ok(format!(
                "{} turns, answer contains the code",
                result.turns
            ))
        })
        .await,
    );

    // 4. Structured output.
    let m = model.clone();
    outcomes.push(
        check("structured output", || async move {
            let req = Request {
                messages: vec![Message::user(
                    "Ada Lovelace was born in 1815 and died at age 36.",
                )],
                max_tokens: 500,
                ..Default::default()
            };
            let person: Person = m.generate_as(req).await.map_err(|e| e.to_string())?;
            if person.age == 36 && person.name.to_lowercase().contains("lovelace") {
                Ok(format!("{person:?}"))
            } else {
                Err(format!("wrong extraction: {person:?}"))
            }
        })
        .await,
    );

    println!("\n━━ {proto} ━━");
    for o in &outcomes {
        match &o.result {
            Ok(detail) => println!("  ✓ {:<18} {:>6}ms  {detail}", o.label, o.millis),
            Err(err) => println!("  ✗ {:<18} {:>6}ms  {err}", o.label, o.millis),
        }
    }
    outcomes
}

/// Adapter: the builder takes ownership of a model, but the suite holds an Arc.
struct ArcModel(Arc<dyn ChatModel>);

impl ChatModel for ArcModel {
    fn stream(&self, req: Request) -> harness_core::ModelStream<'_> {
        self.0.stream(req)
    }
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
}

#[tokio::main]
async fn main() {
    let url = std::env::var("LIVE_URL").unwrap_or_default();
    let key = std::env::var("LIVE_KEY").unwrap_or_default();
    let model = std::env::var("LIVE_MODEL").unwrap_or_else(|_| "claude-sonnet-5".into());
    let proto = std::env::var("LIVE_PROTO").unwrap_or_else(|_| "both".into());
    if url.is_empty() || key.is_empty() {
        eprintln!("set LIVE_URL, LIVE_KEY (and optionally LIVE_MODEL, LIVE_PROTO)");
        std::process::exit(2);
    }
    let url = url.trim_end_matches('/');
    println!("endpoint: {url}\nmodel:    {model}");

    let mut all = Vec::new();
    if proto == "both" || proto == "anthropic" {
        let m: Arc<dyn ChatModel> = Arc::new(
            harness_anthropic::Anthropic::new(&model)
                .with_base_url(url)
                .with_api_key(&key),
        );
        all.extend(run_suite("anthropic (/v1/messages)", m).await);
    }
    if proto == "both" || proto == "openai" {
        let m: Arc<dyn ChatModel> = Arc::new(
            harness_openai::OpenAi::new(&model)
                .with_base_url(format!("{url}/v1"))
                .with_api_key(&key),
        );
        all.extend(run_suite("openai (/v1/chat/completions)", m).await);
    }

    let failed = all.iter().filter(|o| o.result.is_err()).count();
    println!("\n{} checks, {} failed", all.len(), failed);
    if failed > 0 {
        std::process::exit(1);
    }
}
