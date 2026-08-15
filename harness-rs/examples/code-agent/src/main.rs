//! A Claude Code-style terminal coding agent built on harness.
//!
//! ```text
//! export ANTHROPIC_API_KEY=sk-ant-...
//! cargo run -p code-agent
//! ```
//!
//! The agent explores and edits the current directory with file tools and a
//! shell. Shell commands pause the run for interactive approval (y/N) before
//! executing — the runtime's `AwaitingApproval` state driving a real UI.

mod tools;

use std::io::Write as _;
use std::sync::Arc;

use harness_agent::{Agent, AgentEvent, Checkpointer, Decision, FileCheckpointer};
use harness_anthropic::Anthropic;
use harness_core::{tools, ToolContext};

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

fn system_prompt(cwd: &str) -> String {
    format!(
        "You are a coding agent operating in the user's terminal.\n\
         Working directory: {cwd}\n\n\
         Guidelines:\n\
         - Explore before you edit: use list_dir, read_file and bash (grep/find) to \
           understand the code before changing it.\n\
         - Prefer edit_file for surgical changes; use write_file only for new files or \
           full rewrites.\n\
         - Use bash for searching, git, building and running tests. Verify your changes \
           by running the project's own checks when they exist.\n\
         - Keep replies concise; report what you changed and how you verified it.\n\
         - If a task is ambiguous, state your assumption and proceed."
    )
}

#[tokio::main]
async fn main() {
    if std::env::var("ANTHROPIC_API_KEY")
        .unwrap_or_default()
        .is_empty()
    {
        eprintln!("error: set ANTHROPIC_API_KEY to run the code agent");
        std::process::exit(1);
    }
    let model_id = std::env::var("HARNESS_MODEL").unwrap_or_else(|_| "claude-sonnet-5".to_string());
    let session = std::env::var("HARNESS_SESSION").unwrap_or_else(|_| "default".to_string());
    let ctx = ToolContext::default();
    let cwd = ctx.cwd.display().to_string();

    // Conversations are durable: every turn is checkpointed to disk, so
    // restarting the binary resumes the session where it left off.
    let store = Arc::new(FileCheckpointer::new(ctx.cwd.join(".harness/sessions")));

    let agent = Agent::builder()
        .model(Anthropic::new(&model_id))
        .system(system_prompt(&cwd))
        .tools(tools![
            tools::read_file,
            tools::write_file,
            tools::edit_file,
            tools::list_dir,
            tools::bash,
        ])
        .checkpointer(store.clone())
        .tool_context(ctx)
        .max_turns(100)
        .build();

    println!("{BOLD}code-agent{RESET} {DIM}· {model_id} · session {session} · {cwd}{RESET}");
    println!("{DIM}Describe a task. \"exit\" quits, \"/clear\" resets the conversation.{RESET}\n");

    loop {
        let Some(line) = prompt(&format!("{CYAN}❯{RESET} ")).await else {
            break;
        };
        let input = line.trim().to_string();
        match input.as_str() {
            "" => continue,
            "exit" | "quit" => break,
            "/clear" => {
                match store.delete(&session).await {
                    Ok(()) => println!("{DIM}conversation cleared{RESET}"),
                    Err(e) => eprintln!("{YELLOW}error:{RESET} {e}"),
                }
                continue;
            }
            _ => {}
        }

        if let Err(e) = converse(&agent, &session, &input).await {
            eprintln!("\n{YELLOW}error:{RESET} {e}");
        }
        println!();
    }
}

/// Run one user turn to completion, rendering events as they stream. History
/// travels through the checkpointer, keyed by the session id.
async fn converse(
    agent: &Agent,
    session: &str,
    input: &str,
) -> Result<(), harness_agent::AgentError> {
    let mut run = agent.run_session(session, input);
    let mut result = None;
    while let Some(event) = run.next_event().await {
        match event? {
            AgentEvent::TextDelta(t) => {
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
            AgentEvent::ThinkingDelta(_) => {}
            AgentEvent::ToolCallStarted { name, input, .. } => {
                println!("\n{DIM}⚙ {name} {}{RESET}", compact(&name, &input));
            }
            AgentEvent::ToolCallFinished { output, .. } => {
                let marker = if output.is_error { "✗" } else { "✓" };
                println!("{DIM}  {marker} {}{RESET}", summarize(&output.content));
            }
            AgentEvent::AwaitingApproval(req) => {
                let detail = compact(&req.name, &req.input);
                let answer = prompt(&format!(
                    "{YELLOW}  {} wants to run:{RESET} {detail}\n{YELLOW}  approve? [y/N]{RESET} ",
                    req.name
                ))
                .await
                .unwrap_or_default();
                if answer.trim().eq_ignore_ascii_case("y") {
                    run.decide(Decision::Approve).await;
                } else {
                    run.decide(Decision::Deny(Some("denied by user".into())))
                        .await;
                }
            }
            AgentEvent::Done(r) => {
                println!(
                    "\n{DIM}· {} turns · {} in / {} out tokens{RESET}",
                    r.turns, r.usage.input_tokens, r.usage.output_tokens
                );
                result = Some(());
            }
        }
    }
    result
        .ok_or_else(|| harness_agent::AgentError::Abandoned("run ended without completing".into()))
}

/// One-line rendering of a tool input for the console.
fn compact(name: &str, input: &serde_json::Value) -> String {
    let s = match name {
        "bash" => input["command"].as_str().unwrap_or_default().to_string(),
        _ => input.to_string(),
    };
    let one_line = s.replace('\n', " ⏎ ");
    if one_line.chars().count() > 120 {
        format!("{}…", one_line.chars().take(120).collect::<String>())
    } else {
        one_line
    }
}

fn summarize(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let first = lines.first().copied().unwrap_or("");
    let first = if first.chars().count() > 80 {
        format!("{}…", first.chars().take(80).collect::<String>())
    } else {
        first.to_string()
    };
    if lines.len() > 1 {
        format!("{first} {DIM}(+{} lines){RESET}", lines.len() - 1)
    } else {
        first
    }
}

/// Read one line from stdin without blocking the async runtime.
async fn prompt(text: &str) -> Option<String> {
    print!("{text}");
    let _ = std::io::stdout().flush();
    tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => Some(line),
            Err(_) => None,
        }
    })
    .await
    .ok()
    .flatten()
}
