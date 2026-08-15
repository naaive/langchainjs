//! OpenAI-compatible Chat Completions provider for harness.
//!
//! Speaks the `/chat/completions` dialect, which makes it the adapter for
//! OpenAI itself and for the many servers that implement the same API —
//! Ollama, vLLM, llama.cpp, LM Studio, most gateways:
//!
//! ```ignore
//! let openai = OpenAi::new("gpt-4o");                       // OPENAI_API_KEY
//! let local  = OpenAi::new("llama3.2")
//!     .with_base_url("http://localhost:11434/v1")           // Ollama
//!     .with_api_key("ollama");
//! ```

pub mod convert;
mod sse;

use async_stream::try_stream;
use futures::StreamExt;
use harness_core::{ChatModel, ModelError, ModelStream, Request, Response};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Chat Completions API client (OpenAI or any compatible server).
#[derive(Clone)]
pub struct OpenAi {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAi {
    /// Create a client for `model`, reading the key from `OPENAI_API_KEY`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: model.into(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Point at a compatible server, e.g. `http://localhost:11434/v1` for
    /// Ollama. The path must include the API prefix (usually `/v1`).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    fn request(&self, req: &Request, stream: bool) -> reqwest::RequestBuilder {
        let body = convert::to_body(req, &self.model, stream);
        self.client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
    }
}

async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response, ModelError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let message = resp.text().await.unwrap_or_default();
    Err(ModelError::Api {
        status: status.as_u16(),
        message,
    })
}

#[harness_core::async_trait]
impl ChatModel for OpenAi {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn stream(&self, req: Request) -> ModelStream<'_> {
        let request = self.request(&req, true);
        Box::pin(try_stream! {
            let resp = request
                .send()
                .await
                .map_err(|e| ModelError::Http(e.to_string()))?;
            let resp = check_status(resp).await?;

            let mut events = std::pin::pin!(sse::events(resp.bytes_stream()));
            let mut assembler = convert::ChunkAssembler::default();
            while let Some(chunk) = events.next().await {
                for out in assembler.handle(&chunk?)? {
                    yield out;
                }
            }
            // Compatible servers end with `data: [DONE]`; the assembled
            // response is emitted when the sentinel (or the stream end)
            // is reached.
            if let Some(out) = assembler.finish() {
                yield out;
            }
        })
    }

    async fn generate(&self, req: Request) -> Result<Response, ModelError> {
        let resp = self
            .request(&req, false)
            .send()
            .await
            .map_err(|e| ModelError::Http(e.to_string()))?;
        let resp = check_status(resp).await?;
        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ModelError::Deserialize(e.to_string()))?;
        convert::parse_response(&value)
    }
}
