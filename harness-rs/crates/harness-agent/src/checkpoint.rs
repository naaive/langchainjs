//! Durable conversation state.
//!
//! Because [`Message`] is plain serde data, a checkpoint is just the message
//! history — there is no execution-engine state to snapshot. The runtime
//! saves after every append (assistant message, tool results), so a process
//! crash mid-run loses at most the model call in flight; resuming a session
//! picks up with all completed tool work intact.

use std::collections::HashMap;
use std::path::PathBuf;

use harness_core::Message;
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("checkpoint io: {0}")]
    Io(#[from] std::io::Error),

    #[error("checkpoint serialization: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Storage backend for session histories. Implementations must make `save`
/// atomic per session (a torn write must not corrupt the previous state).
#[harness_core::async_trait]
pub trait Checkpointer: Send + Sync {
    async fn save(&self, session: &str, messages: &[Message]) -> Result<(), CheckpointError>;
    async fn load(&self, session: &str) -> Result<Option<Vec<Message>>, CheckpointError>;
    async fn delete(&self, session: &str) -> Result<(), CheckpointError>;
}

/// Share one store between the agent and application code
/// (`.checkpointer(store.clone())` with `store: Arc<FileCheckpointer>`).
#[harness_core::async_trait]
impl<T: Checkpointer + ?Sized> Checkpointer for std::sync::Arc<T> {
    async fn save(&self, session: &str, messages: &[Message]) -> Result<(), CheckpointError> {
        (**self).save(session, messages).await
    }
    async fn load(&self, session: &str) -> Result<Option<Vec<Message>>, CheckpointError> {
        (**self).load(session).await
    }
    async fn delete(&self, session: &str) -> Result<(), CheckpointError> {
        (**self).delete(session).await
    }
}

/// In-memory checkpointer: survives across runs within a process. Useful for
/// tests and for multi-turn sessions in a long-lived service.
#[derive(Default)]
pub struct MemoryCheckpointer {
    sessions: Mutex<HashMap<String, Vec<Message>>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        Self::default()
    }
}

#[harness_core::async_trait]
impl Checkpointer for MemoryCheckpointer {
    async fn save(&self, session: &str, messages: &[Message]) -> Result<(), CheckpointError> {
        self.sessions
            .lock()
            .await
            .insert(session.to_string(), messages.to_vec());
        Ok(())
    }

    async fn load(&self, session: &str) -> Result<Option<Vec<Message>>, CheckpointError> {
        Ok(self.sessions.lock().await.get(session).cloned())
    }

    async fn delete(&self, session: &str) -> Result<(), CheckpointError> {
        self.sessions.lock().await.remove(session);
        Ok(())
    }
}

/// File-based checkpointer: one JSON file per session in a directory.
/// Writes go to a temp file first and are renamed into place, so a crash
/// mid-write never corrupts the previous checkpoint.
pub struct FileCheckpointer {
    dir: PathBuf,
}

impl FileCheckpointer {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, session: &str) -> PathBuf {
        // Session ids become file names; anything path-hostile is mapped away.
        let safe: String = session
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.dir.join(format!("{safe}.json"))
    }
}

#[harness_core::async_trait]
impl Checkpointer for FileCheckpointer {
    async fn save(&self, session: &str, messages: &[Message]) -> Result<(), CheckpointError> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let path = self.path(session);
        let tmp = path.with_extension("json.tmp");
        let data = serde_json::to_vec_pretty(messages)?;
        tokio::fs::write(&tmp, data).await?;
        tokio::fs::rename(&tmp, &path).await?;
        Ok(())
    }

    async fn load(&self, session: &str) -> Result<Option<Vec<Message>>, CheckpointError> {
        match tokio::fs::read(self.path(session)).await {
            Ok(data) => Ok(Some(serde_json::from_slice(&data)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn delete(&self, session: &str) -> Result<(), CheckpointError> {
        match tokio::fs::remove_file(self.path(session)).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
