//! The coding agent's toolbox: file inspection and editing plus a shell,
//! all defined with `#[tool]`. Shell commands require human approval.

use std::path::{Path, PathBuf};

use harness_core::{ToolContext, ToolError};
use harness_macros::tool;

const MAX_OUTPUT_BYTES: usize = 48_000;
const BASH_TIMEOUT_SECS: u64 = 120;

fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_OUTPUT_BYTES {
        let mut cut = MAX_OUTPUT_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n... [output truncated]");
    }
    s
}

/// Read a UTF-8 text file. Returns the content with 1-based line numbers.
/// Use `offset` and `limit` (line counts) to page through large files.
#[tool]
pub async fn read_file(
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let full = resolve(&ctx.cwd, &path);
    let content = tokio::fs::read_to_string(&full)
        .await
        .map_err(|e| ToolError::Execution(format!("cannot read {}: {e}", full.display())))?;
    let start = offset.unwrap_or(0);
    let limit = limit.unwrap_or(2000);
    let numbered: Vec<String> = content
        .lines()
        .enumerate()
        .skip(start)
        .take(limit)
        .map(|(i, line)| format!("{:>5}\t{line}", i + 1))
        .collect();
    if numbered.is_empty() {
        return Ok(format!(
            "{} is empty (or offset is past EOF)",
            full.display()
        ));
    }
    Ok(truncate(numbered.join("\n")))
}

/// Create or overwrite a file with the given content. Parent directories are
/// created as needed.
#[tool]
pub async fn write_file(
    path: String,
    content: String,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let full = resolve(&ctx.cwd, &path);
    if let Some(parent) = full.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let bytes = content.len();
    tokio::fs::write(&full, content)
        .await
        .map_err(|e| ToolError::Execution(format!("cannot write {}: {e}", full.display())))?;
    Ok(format!("wrote {bytes} bytes to {}", full.display()))
}

/// Replace `old_string` with `new_string` in a file. `old_string` must occur
/// exactly once — include enough surrounding context to make it unique.
#[tool]
pub async fn edit_file(
    path: String,
    old_string: String,
    new_string: String,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let full = resolve(&ctx.cwd, &path);
    let content = tokio::fs::read_to_string(&full)
        .await
        .map_err(|e| ToolError::Execution(format!("cannot read {}: {e}", full.display())))?;
    let matches = content.matches(&old_string).count();
    match matches {
        0 => Err(ToolError::Execution(
            "old_string not found in file — re-read the file and try again".into(),
        )),
        1 => {
            let updated = content.replacen(&old_string, &new_string, 1);
            tokio::fs::write(&full, updated).await?;
            Ok(format!("edited {}", full.display()))
        }
        n => Err(ToolError::Execution(format!(
            "old_string matches {n} locations — add surrounding context to make it unique"
        ))),
    }
}

/// List a directory (non-recursive). Directories are suffixed with `/`.
#[tool]
pub async fn list_dir(path: Option<String>, ctx: &ToolContext) -> Result<String, ToolError> {
    let full = resolve(&ctx.cwd, path.as_deref().unwrap_or("."));
    let mut reader = tokio::fs::read_dir(&full)
        .await
        .map_err(|e| ToolError::Execution(format!("cannot list {}: {e}", full.display())))?;
    let mut entries = Vec::new();
    while let Some(entry) = reader.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
        entries.push(if is_dir { format!("{name}/") } else { name });
    }
    entries.sort();
    if entries.is_empty() {
        return Ok(format!("{} is empty", full.display()));
    }
    Ok(truncate(entries.join("\n")))
}

/// Run a shell command with `sh -c` in the working directory and return its
/// combined output. Use this for searching (grep/find), git, builds and tests.
#[tool(approval)]
pub async fn bash(command: String, ctx: &ToolContext) -> Result<String, ToolError> {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(BASH_TIMEOUT_SECS),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&ctx.cwd)
            .output(),
    )
    .await
    .map_err(|_| ToolError::Execution(format!("command timed out after {BASH_TIMEOUT_SECS}s")))?
    .map_err(|e| ToolError::Execution(format!("failed to spawn: {e}")))?;

    let mut out = String::new();
    out.push_str(&String::from_utf8_lossy(&result.stdout));
    let stderr = String::from_utf8_lossy(&result.stderr);
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("[stderr]\n");
        out.push_str(&stderr);
    }
    if !result.status.success() {
        return Ok(truncate(format!(
            "exit status: {}\n{out}",
            result
                .status
                .code()
                .map_or("signal".into(), |c| c.to_string())
        )));
    }
    if out.trim().is_empty() {
        out = "(no output)".into();
    }
    Ok(truncate(out))
}
