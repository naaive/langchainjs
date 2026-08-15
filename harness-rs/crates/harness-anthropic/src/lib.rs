//! Anthropic Messages API provider for harness.
//!
//! Implements [`ChatModel`] over the Messages API, including SSE streaming
//! with incremental tool-input assembly. The provider is stateless: every
//! request carries the full conversation.

mod convert;
mod sse;

use async_stream::try_stream;
use futures::StreamExt;
use harness_core::{ChatModel, ModelError, ModelStream, Request, Response};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Anthropic Messages API client.
///
/// ```ignore
/// let model = Anthropic::new("claude-sonnet-5"); // reads ANTHROPIC_API_KEY
/// ```
#[derive(Clone)]
pub struct Anthropic {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl Anthropic {
    /// Create a client for `model`, reading the key from `ANTHROPIC_API_KEY`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            model: model.into(),
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn request(&self, req: &Request, stream: bool) -> reqwest::RequestBuilder {
        let body = convert::to_body(req, &self.model, DEFAULT_MAX_TOKENS, stream);
        self.client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
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
impl ChatModel for Anthropic {
    fn stream(&self, req: Request) -> ModelStream<'_> {
        let request = self.request(&req, true);
        Box::pin(try_stream! {
            let resp = request
                .send()
                .await
                .map_err(|e| ModelError::Http(e.to_string()))?;
            let resp = check_status(resp).await?;

            let mut events = std::pin::pin!(sse::events(resp.bytes_stream()));
            let mut assembler = sse::Assembler::default();
            while let Some(event) = events.next().await {
                for out in assembler.handle(event?)? {
                    yield out;
                }
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
