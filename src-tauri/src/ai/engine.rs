//! Core abstractions for the AI Execution Engine (specs §3.3).
//!
//! Model access is hidden behind two traits so the inference router can swap a
//! local GPU backend (`llama.cpp` / Vulkan) for a cloud fallback transparently:
//!
//! * [`LlmEngine`] — text generation and image analysis, both **streaming**
//!   (token-incremental) for UI responsiveness.
//! * [`Embedder`] — CPU-side embedding (`all-MiniLM-L6-v2`, 384-dim). Kept
//!   separate because it is backend-independent and never goes to the cloud.
//!
//! Callers depend only on these traits; new backends implement them.

use std::fmt;

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Who authored a turn in a chat node's conversation (§11.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Model,
}

/// One turn of a chat-node conversation. The ordered history is what gives the
/// model memory of the exchange so far — a follow-up sees every prior turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub text: String,
}

/// Dimensionality of the embedding vectors produced by [`Embedder`].
///
/// Fixed by `all-MiniLM-L6-v2`. The LanceDB tables declare `VECTOR(384)`
/// columns to match (specs §5).
pub const EMBEDDING_DIM: usize = 384;

/// Errors surfaced by the inference and embedding backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    /// The backend ran but failed to produce a valid result.
    InferenceFailed(String),
    /// A network/transport failure talking to a remote (cloud) backend.
    NetworkError(String),
    /// No usable compute backend is available (e.g. GPU library missing).
    HardwareUnavailable,
    /// The request was refused by the Privacy Guard before reaching a backend
    /// (blocked URL, or a tab without AI consent). Cloud fallback in particular
    /// must never fire for these — see [`crate::ai::privacy`].
    ConsentDenied(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::InferenceFailed(msg) => write!(f, "inference failed: {msg}"),
            LlmError::NetworkError(msg) => write!(f, "network error: {msg}"),
            LlmError::HardwareUnavailable => write!(f, "no compute backend available"),
            LlmError::ConsentDenied(msg) => write!(f, "consent denied: {msg}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// A stream of incremental text tokens, or an error mid-stream.
///
/// `'static` because the stream outlives the borrow of `&self` that produced
/// it (it is handed off to the Tauri event loop / a oneshot channel).
pub type TokenStream = BoxStream<'static, Result<String, LlmError>>;

/// A text/vision generation backend (local GPU or cloud).
#[async_trait]
pub trait LlmEngine: Send + Sync {
    /// Streaming text generation. `context` carries any RAG-retrieved material
    /// or system framing; `prompt` is the user turn.
    async fn generate_text(&self, prompt: &str, context: &str) -> Result<TokenStream, LlmError>;

    /// Streaming **multi-turn** generation: `history` is the ordered conversation
    /// (ending with the latest user turn) so the model remembers the exchange
    /// (§11.1 follow-ups stay in the node). `context` is the ancestor grounding.
    ///
    /// Default: fall back to a single-turn call on the last user message, so
    /// backends that don't model a conversation still work.
    async fn generate_chat(
        &self,
        history: &[ChatTurn],
        context: &str,
    ) -> Result<TokenStream, LlmError> {
        let last = history
            .iter()
            .rev()
            .find(|t| t.role == ChatRole::User)
            .map(|t| t.text.as_str())
            .unwrap_or("");
        self.generate_text(last, context).await
    }

    /// Streaming multimodal image analysis (transcribe / explain / translate).
    /// `image_bytes` is the raw encoded image (PNG/JPEG/…).
    async fn analyze_image(&self, image_bytes: &[u8], prompt: &str)
        -> Result<TokenStream, LlmError>;

    /// Stable identifier of this backend, for local logging/telemetry and
    /// routing decisions (e.g. `"llama-cpp-vulkan"`, `"gemini-flash"`).
    fn backend_id(&self) -> &str;
}

/// CPU-side text embedder. Independent of the GPU/cloud generation backend.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed `text` into a vector of length [`EMBEDDING_DIM`].
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError>;

    /// Identifier of the embedding model, stored alongside every vector as
    /// `embedding_model_version` so rows can be re-indexed when it changes
    /// (specs §5).
    fn model_version(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_is_human_readable() {
        assert_eq!(
            LlmError::HardwareUnavailable.to_string(),
            "no compute backend available"
        );
        assert_eq!(
            LlmError::InferenceFailed("oom".into()).to_string(),
            "inference failed: oom"
        );
    }

    #[test]
    fn llm_error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&LlmError::HardwareUnavailable);
    }
}
