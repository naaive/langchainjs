//! End-to-end tests of the agent loop against a scripted mock model,
//! exercising the `#[tool]` macro, tool execution, and the approval flow.

use std::collections::VecDeque;
use std::sync::Mutex;

use harness_agent::{Agent, AgentEvent, Decision};
use harness_core::{
    ChatModel, Content, Message, ModelStream, Request, Response, Role, StopReason, StreamEvent,
    Tool, ToolError, Usage,
};
use harness_macros::tool;

/// Replays a scripted sequence of responses, one per model call.
struct MockModel {
    script: Mutex<VecDeque<Response>>,
}

impl MockModel {
    fn new(script: Vec<Response>) -> Self {
        Self {
            script: Mutex::new(script.into()),
        }
    }
}

impl ChatModel for MockModel {
    fn stream(&self, _req: Request) -> ModelStream<'_> {
        let resp = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock script exhausted");
        let mut events: Vec<Result<StreamEvent, _>> = Vec::new();
        let text = resp.message.text();
        if !text.is_empty() {
            events.push(Ok(StreamEvent::TextDelta(text)));
        }
        events.push(Ok(StreamEvent::Completed(resp)));
        Box::pin(futures::stream::iter(events))
    }
}

fn tool_use_response(id: &str, name: &str, input: serde_json::Value) -> Response {
    Response {
        message: Message {
            role: Role::Assistant,
            content: vec![Content::ToolUse {
                id: id.into(),
                name: name.into(),
                input,
            }],
        },
        stop_reason: StopReason::ToolUse,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
        },
    }
}

fn text_response(text: &str) -> Response {
    Response {
        message: Message::assistant(text),
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            input_tokens: 20,
            output_tokens: 7,
        },
    }
}

/// Echo the given text back.
#[tool]
async fn echo(text: String) -> Result<String, ToolError> {
    Ok(format!("echo: {text}"))
}

/// A destructive operation that must be approved by a human.
#[tool(approval)]
async fn destroy(target: Option<String>) -> Result<String, ToolError> {
    Ok(format!(
        "destroyed {}",
        target.unwrap_or_else(|| "everything".into())
    ))
}

#[test]
fn tool_macro_generates_schema_and_metadata() {
    let schema = Tool::input_schema(&echo);
    assert_eq!(
        schema["properties"]["text"]["type"], "string",
        "schema: {schema}"
    );
    assert!(schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|v| v == "text"));
    assert_eq!(Tool::name(&echo), "echo");
    assert_eq!(Tool::description(&echo), "Echo the given text back.");
    assert!(!Tool::needs_approval(&echo));

    // Option<T> parameters are not required.
    let schema = Tool::input_schema(&destroy);
    let required = schema["required"].as_array().cloned().unwrap_or_default();
    assert!(!required.iter().any(|v| v == "target"), "schema: {schema}");
    assert!(Tool::needs_approval(&destroy));
}

#[tokio::test]
async fn runs_tool_loop_to_completion() {
    let model = MockModel::new(vec![
        tool_use_response("t1", "echo", serde_json::json!({"text": "hi"})),
        text_response("all done"),
    ]);
    let agent = Agent::builder()
        .model(model)
        .system("test agent")
        .tool(echo)
        .build();

    let result = agent.run(vec![], "please echo hi").wait().await.unwrap();

    assert_eq!(result.final_text, "all done");
    assert_eq!(result.turns, 2);
    assert_eq!(result.usage.input_tokens, 30);
    // user, assistant(tool_use), user(tool_result), assistant(text)
    assert_eq!(result.messages.len(), 4);
    match &result.messages[2].content[0] {
        Content::ToolResult {
            content, is_error, ..
        } => {
            assert_eq!(content, "echo: hi");
            assert!(!is_error);
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[tokio::test]
async fn approval_denial_is_reported_to_the_model() {
    let model = MockModel::new(vec![
        tool_use_response("t1", "destroy", serde_json::json!({})),
        text_response("understood"),
    ]);
    let agent = Agent::builder().model(model).tool(destroy).build();

    let mut run = agent.run(vec![], "destroy everything");
    let mut saw_approval = false;
    let mut denied_output = None;
    while let Some(event) = run.next_event().await {
        match event.unwrap() {
            AgentEvent::AwaitingApproval(req) => {
                assert_eq!(req.name, "destroy");
                saw_approval = true;
                run.decide(Decision::Deny(Some("too dangerous".into())))
                    .await;
            }
            AgentEvent::ToolCallFinished { output, .. } => {
                denied_output = Some(output);
            }
            AgentEvent::Done(result) => {
                assert_eq!(result.final_text, "understood");
                break;
            }
            _ => {}
        }
    }
    assert!(saw_approval, "approval request never surfaced");
    let output = denied_output.expect("tool call never finished");
    assert!(output.is_error);
    assert_eq!(output.content, "too dangerous");
}

#[tokio::test]
async fn unknown_tool_becomes_error_result() {
    let model = MockModel::new(vec![
        tool_use_response("t1", "nonexistent", serde_json::json!({})),
        text_response("recovered"),
    ]);
    let agent = Agent::builder().model(model).tool(echo).build();

    let result = agent.run(vec![], "go").wait().await.unwrap();
    match &result.messages[2].content[0] {
        Content::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("unknown tool"));
        }
        other => panic!("expected tool result, got {other:?}"),
    }
    assert_eq!(result.final_text, "recovered");
}
