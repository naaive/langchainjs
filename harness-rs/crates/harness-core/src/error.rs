use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    /// Transport-level failure (connect, TLS, timeout).
    #[error("http error: {0}")]
    Http(String),

    /// The API answered with a non-success status.
    #[error("api error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// The stream broke or produced something unparseable mid-flight.
    #[error("stream error: {0}")]
    Stream(String),

    #[error("failed to deserialize model output: {0}")]
    Deserialize(String),
}

impl ModelError {
    /// Whether retrying the same request may succeed. Rate limits, server
    /// errors and transport failures are retryable; malformed requests and
    /// deserialization bugs are not.
    pub fn is_retryable(&self) -> bool {
        match self {
            ModelError::Http(_) | ModelError::Stream(_) => true,
            ModelError::Api { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            ModelError::Deserialize(_) => false,
        }
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    /// The model supplied input that doesn't match the tool's schema.
    #[error("invalid tool input: {0}")]
    InvalidInput(String),

    /// The tool ran but failed.
    #[error("{0}")]
    Execution(String),
}

impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        ToolError::Execution(e.to_string())
    }
}
