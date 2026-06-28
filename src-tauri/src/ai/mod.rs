//! AI Execution Engine (specs §3, Phase 1).
//!
//! Headless core, built and tested before the browser shell. Sub-modules:
//!
//! * [`engine`] — the `LlmEngine` / `Embedder` trait seam and `LlmError`.
//! * [`privacy`] — the Privacy Guard (URL classification + per-tab consent).
//! * [`router`] — the hybrid local/cloud inference router.
//!
//! Concrete backends (llama.cpp/Vulkan, MiniLM embedder, cloud) and the LanceDB
//! RAG pipeline are added on top of these as Phase 1 progresses.

pub mod embedder;
pub mod engine;
pub mod llama;
pub mod privacy;
pub mod rag;
pub mod router;

pub use embedder::{MiniLmEmbedder, MiniLmVariant, MINILM_MODEL_VERSION};
pub use engine::{Embedder, LlmEngine, LlmError, TokenStream, EMBEDDING_DIM};
pub use llama::{LlamaConfig, LlamaEngine};
pub use privacy::{BlockReason, GuardDecision, PrivacyGuard, TabId, UrlCategory, UrlClassifier};
pub use router::{HardwareProfile, InferenceRouter, Route};
