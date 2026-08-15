//! Tests for durable sessions, structured output, compaction middleware,
//! concurrent tool execution, and approval editing — all against mock models.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use harness_agent::{
    Agent, AgentEvent, Checkpointer, Compaction, Decision, FileCheckpointer, MemoryCheckpointer,
};
use harness_core::{
    ChatModel, Content, GenerateStructured, Message, ModelStream, Request, Response, Role,
    StopReason, StreamEvent, ToolError, Usage, STRUCTURED_OUTPUT_TOOL,
};
use harness_macros::tool;

struct MockModel {
    script: Mutex<VecDeque<Response>>,
    seen_requests: Mutex<Vec<Request>>,
}

impl MockModel {
    fn new(script: Vec<Response>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            seen_requests: Mutex::new(Vec::new()),
        }
    }
}

impl ChatModel for MockModel {
    fn stream(&self, req: Request) -> ModelStream<'_> {
        self.seen_requests.lock().unwrap().push(req);
        let resp = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock script exhausted");
        Box::pin(futures::stream::iter(vec![Ok(StreamEvent::Completed(
            resp,
        ))]))
    }
}

fn text_response(text: &str) -> Response {
    Response {
        message: Message::assistant(text),
        stop_reason: StopReason::EndTurn,
        usage: Usage::default(),
    }
}

fn tool_use_response(calls: &[(&str, &str, serde_json::Value)]) -> Response {
    Response {
        message: Message {
            role: Role::Assistant,
            content: calls
                .iter()
                .map(|(id, name, input)| Content::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input: input.clone(),
                })
                .collect(),
        },
        stop_reason: StopReason::ToolUse,
        usage: Usage::default(),
    }
}

// ---------------------------------------------------------------- sessions

#[tokio::test]
async fn run_session_persists_and_reloads_history() {
    let cp = Arc::new(MemoryCheckpointer::new());
    let agent = |script| {
        Agent::builder()
            .model(MockModel::new(script))
            .checkpointer(cp.clone())
            .build()
    };

    let r1 = agent(vec![text_response("first answer")])
        .run_session("s1", "first question")
        .wait()
        .await
        .unwrap();
    assert_eq!(r1.messages.len(), 2);

    // A brand-new agent (fresh process, same store) continues the session.
    let r2 = agent(vec![text_response("second answer")])
        .run_session("s1", "second question")
        .wait()
        .await
        .unwrap();
    assert_eq!(r2.messages.len(), 4, "history reloaded from checkpoint");
    assert_eq!(r2.messages[0].text(), "first question");
    assert_eq!(r2.messages[3].text(), "second answer");

    let stored = cp.load("s1").await.unwrap().unwrap();
    assert_eq!(stored.len(), 4);
}

#[tokio::test]
async fn file_checkpointer_round_trips_and_survives_hostile_ids() {
    let dir = std::env::temp_dir().join(format!("harness-cp-test-{}", std::process::id()));
    let cp = FileCheckpointer::new(&dir);
    let messages = vec![
        Message::user("hello"),
        Message {
            role: Role::Assistant,
            content: vec![Content::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
            }],
        },
    ];
    cp.save("weird/../id with spaces", &messages).await.unwrap();
    let loaded = cp.load("weird/../id with spaces").await.unwrap().unwrap();
    assert_eq!(loaded, messages);
    assert!(cp.load("missing").await.unwrap().is_none());
    cp.delete("weird/../id with spaces").await.unwrap();
    assert!(cp.load("weird/../id with spaces").await.unwrap().is_none());
    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn run_session_without_checkpointer_fails_with_config_error() {
    let agent = Agent::builder().model(MockModel::new(vec![])).build();
    let err = agent.run_session("s", "hi").wait().await.unwrap_err();
    assert!(matches!(err, harness_agent::AgentError::Config(_)));
}

// -------------------------------------------------------- structured output

#[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq)]
struct Invoice {
    customer: String,
    total_cents: u64,
}

#[tokio::test]
async fn generate_as_deserializes_and_repairs() {
    // First response: schema-invalid (total_cents is a string). Second: valid.
    let bad = tool_use_response(&[(
        "s1",
        STRUCTURED_OUTPUT_TOOL,
        serde_json::json!({"customer": "ACME", "total_cents": "12"}),
    )]);
    let good = tool_use_response(&[(
        "s2",
        STRUCTURED_OUTPUT_TOOL,
        serde_json::json!({"customer": "ACME", "total_cents": 1200}),
    )]);
    let model = MockModel::new(vec![bad, good]);

    let req = Request {
        messages: vec![Message::user("bill ACME $12")],
        ..Default::default()
    };
    let invoice: Invoice = model.generate_as(req).await.unwrap();
    assert_eq!(
        invoice,
        Invoice {
            customer: "ACME".into(),
            total_cents: 1200
        }
    );

    // The repair round-trip carried the validation error back to the model.
    let seen = model.seen_requests.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let repair_req = &seen[1];
    let last = repair_req.messages.last().unwrap();
    match &last.content[0] {
        Content::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("schema validation"));
        }
        other => panic!("expected repair tool result, got {other:?}"),
    }
    // The synthetic tool was forced since the request had no other tools.
    assert!(seen[0].tool_choice.is_some());
}

// -------------------------------------------------------------- compaction

#[tokio::test]
async fn compaction_summarizes_old_history_and_caches() {
    let summarizer_calls = Arc::new(AtomicU32::new(0));
    struct Summarizer(Arc<AtomicU32>);
    impl ChatModel for Summarizer {
        fn stream(&self, _req: Request) -> ModelStream<'_> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Box::pin(futures::stream::iter(vec![Ok(StreamEvent::Completed(
                text_response("SUMMARY OF OLD WORK"),
            ))]))
        }
    }

    // Enough long history to blow a tiny budget.
    let mut history: Vec<Message> = Vec::new();
    for i in 0..20 {
        history.push(Message::user(format!("question {i}: {}", "x".repeat(400))));
        history.push(Message::assistant(format!(
            "answer {i}: {}",
            "y".repeat(400)
        )));
    }

    let main_model = MockModel::new(vec![text_response("done 1"), text_response("done 2")]);
    let agent = Agent::builder()
        .model(main_model)
        .middleware(
            // Budget low enough that the long history triggers compaction,
            // high enough that the compacted suffix keeps fitting on the next
            // run, so the cached summary is reused.
            Compaction::new(Summarizer(summarizer_calls.clone()), 2000).keep_recent(4),
        )
        .build();

    let r1 = agent.run(history, "next task").wait().await.unwrap();
    assert_eq!(summarizer_calls.load(Ordering::SeqCst), 1);

    // Second run over the grown history: same split point → cached summary.
    let _r2 = agent.run(r1.messages, "another task").wait().await.unwrap();
    assert_eq!(
        summarizer_calls.load(Ordering::SeqCst),
        1,
        "summary should be cached for an unchanged split point"
    );
}

#[tokio::test]
async fn compaction_rewrites_request_but_not_canonical_history() {
    struct CapturingModel {
        seen: Mutex<Vec<usize>>, // message counts per request
    }
    impl ChatModel for CapturingModel {
        fn stream(&self, req: Request) -> ModelStream<'_> {
            self.seen.lock().unwrap().push(req.messages.len());
            let first = req.messages.first().unwrap().text();
            assert!(
                first.starts_with("[Summary of earlier conversation]"),
                "compacted request must start with the summary, got: {first:.60}"
            );
            Box::pin(futures::stream::iter(vec![Ok(StreamEvent::Completed(
                text_response("ok"),
            ))]))
        }
    }
    struct StaticSummarizer;
    impl ChatModel for StaticSummarizer {
        fn stream(&self, _req: Request) -> ModelStream<'_> {
            Box::pin(futures::stream::iter(vec![Ok(StreamEvent::Completed(
                text_response("condensed"),
            ))]))
        }
    }

    let mut history = Vec::new();
    for i in 0..30 {
        history.push(Message::user(format!("q{i} {}", "x".repeat(300))));
        history.push(Message::assistant(format!("a{i} {}", "y".repeat(300))));
    }
    let original_len = history.len();

    let agent = Agent::builder()
        .model(CapturingModel {
            seen: Mutex::new(vec![]),
        })
        .middleware(Compaction::new(StaticSummarizer, 300).keep_recent(4))
        .build();

    let result = agent.run(history, "go").wait().await.unwrap();
    // Canonical history is untouched: all originals + new user + assistant.
    assert_eq!(result.messages.len(), original_len + 2);
}

// ------------------------------------------------- concurrent tools & edit

/// Sleeps for the given milliseconds, then reports when it finished.
#[tool]
async fn slow_op(millis: u64) -> Result<String, ToolError> {
    tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
    Ok(format!("slept {millis}"))
}

#[tokio::test]
async fn independent_tool_calls_run_concurrently() {
    // Two 150ms tools in one batch: sequential would take 300ms+.
    let model = MockModel::new(vec![
        tool_use_response(&[
            ("t1", "slow_op", serde_json::json!({"millis": 150})),
            ("t2", "slow_op", serde_json::json!({"millis": 150})),
        ]),
        text_response("done"),
    ]);
    let agent = Agent::builder().model(model).tool(slow_op).build();

    let start = std::time::Instant::now();
    let result = agent.run(vec![], "go").wait().await.unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result.final_text, "done");
    assert!(
        elapsed < std::time::Duration::from_millis(280),
        "tools did not run concurrently: {elapsed:?}"
    );
    // Results arrive in the model's original order regardless of completion.
    match &result.messages[2].content[..] {
        [Content::ToolResult { tool_use_id: a, .. }, Content::ToolResult { tool_use_id: b, .. }] => {
            assert_eq!((a.as_str(), b.as_str()), ("t1", "t2"));
        }
        other => panic!("expected two tool results, got {other:?}"),
    }
}

/// Echo with approval, used to test Decision::Edit.
#[tool(approval)]
async fn gated_echo(text: String) -> Result<String, ToolError> {
    Ok(format!("ran: {text}"))
}

#[tokio::test]
async fn approval_edit_replaces_tool_input() {
    let model = MockModel::new(vec![
        tool_use_response(&[("t1", "gated_echo", serde_json::json!({"text": "rm -rf /"}))]),
        text_response("finished"),
    ]);
    let agent = Agent::builder().model(model).tool(gated_echo).build();

    let mut run = agent.run(vec![], "go");
    let mut output = None;
    while let Some(event) = run.next_event().await {
        match event.unwrap() {
            AgentEvent::AwaitingApproval(req) => {
                assert_eq!(req.input["text"], "rm -rf /");
                run.decide(Decision::Edit(serde_json::json!({"text": "ls"})))
                    .await;
            }
            AgentEvent::ToolCallFinished { output: o, .. } => output = Some(o),
            AgentEvent::Done(_) => break,
            _ => {}
        }
    }
    let output = output.unwrap();
    assert!(!output.is_error);
    assert_eq!(output.content, "ran: ls", "edited input should be used");
}

// ------------------------------------------------------------- multi-agent

#[tokio::test]
async fn sub_agent_runs_as_a_tool() {
    // Researcher: answers directly.
    let researcher = Agent::builder()
        .model(MockModel::new(vec![text_response("Rust 1.94 is current")]))
        .system("you research things")
        .build();

    // Lead delegates to the researcher, then answers.
    let lead_model = MockModel::new(vec![
        tool_use_response(&[(
            "d1",
            "researcher",
            serde_json::json!({"task": "what is the current Rust version?"}),
        )]),
        text_response("According to research: Rust 1.94"),
    ]);
    let lead = Agent::builder()
        .model(lead_model)
        .tool(researcher.as_tool("researcher", "Delegate research questions to this agent."))
        .build();

    let result = lead
        .run(vec![], "find the rust version")
        .wait()
        .await
        .unwrap();
    assert_eq!(result.final_text, "According to research: Rust 1.94");
    match &result.messages[2].content[0] {
        Content::ToolResult {
            content, is_error, ..
        } => {
            assert!(!is_error);
            assert_eq!(content, "Rust 1.94 is current");
        }
        other => panic!("expected sub-agent result, got {other:?}"),
    }
}
