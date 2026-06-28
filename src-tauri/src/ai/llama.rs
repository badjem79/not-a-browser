//! Local generation backend: Gemma via `llama.cpp` with the **Vulkan** backend
//! (specs §3.1). Implements [`LlmEngine::generate_text`] with token streaming.
//!
//! `llama.cpp` inference is synchronous and its context is single-threaded, so
//! each request runs the decode loop on a dedicated OS thread and streams tokens
//! back over a tokio MPSC channel — mirroring the AI Execution Engine design in
//! specs §3.2 (serialized inference, incremental tokens to the UI). The returned
//! [`TokenStream`] is just the receiving end of that channel.
//!
//! `analyze_image` (vision via mmproj/mtmd) is the next step and currently errors.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::mpsc::Sender;
use tokio_stream::wrappers::ReceiverStream;

use crate::ai::engine::{LlmEngine, LlmError, TokenStream};

/// Process-global llama backend. `LlamaBackend::init()` must run exactly once
/// per process; the model/context APIs only take `&LlamaBackend` as a witness
/// that it has been initialized, so a shared `&'static` is all they need.
fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| LlamaBackend::init().expect("failed to initialize llama backend"))
}

/// Runtime configuration for [`LlamaEngine`].
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Layers to offload to the GPU. `999` = the whole model on Vulkan.
    pub n_gpu_layers: u32,
    /// Context window in tokens (KV cache size). 4096 is a comfortable default;
    /// Gemma supports far more, at higher VRAM cost (specs §4.1).
    pub n_ctx: u32,
    /// Maximum number of tokens to generate per request.
    pub n_predict: i32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 999,
            n_ctx: 4096,
            n_predict: 512,
        }
    }
}

/// Gemma generation backend over llama.cpp/Vulkan.
pub struct LlamaEngine {
    model: Arc<LlamaModel>,
    config: LlamaConfig,
    backend_id: String,
}

impl LlamaEngine {
    /// Load a GGUF model from disk and offload it to the GPU. This is heavy
    /// (mmaps ~7 GB and uploads weights to VRAM); call it off the UI thread.
    pub fn load(model_path: impl AsRef<Path>, config: LlamaConfig) -> Result<Self, LlmError> {
        let backend = backend();
        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| LlmError::InferenceFailed(format!("load model: {e}")))?;
        Ok(Self {
            model: Arc::new(model),
            config,
            backend_id: "llama-cpp-vulkan".to_string(),
        })
    }
}

#[async_trait]
impl LlmEngine for LlamaEngine {
    async fn generate_text(&self, prompt: &str, context: &str) -> Result<TokenStream, LlmError> {
        let full_prompt = format_gemma_prompt(prompt, context);
        let model = self.model.clone();
        let n_ctx = self.config.n_ctx;
        let n_predict = self.config.n_predict;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, LlmError>>(64);
        std::thread::spawn(move || {
            if let Err(e) = run_generation(&model, &full_prompt, n_ctx, n_predict, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn analyze_image(
        &self,
        _image_bytes: &[u8],
        _prompt: &str,
    ) -> Result<TokenStream, LlmError> {
        Err(LlmError::InferenceFailed(
            "analyze_image not yet implemented (vision via mmproj/mtmd is the next step)".into(),
        ))
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }
}

/// Wrap a prompt in Gemma's chat template. Any RAG `context` is prepended to the
/// user turn (Gemma has no dedicated system role).
fn format_gemma_prompt(prompt: &str, context: &str) -> String {
    let user = if context.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{context}\n\n{prompt}")
    };
    format!("<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n")
}

/// Run the synchronous decode loop, streaming each detokenized piece into `tx`.
/// Returns early (Ok) if the receiver is dropped — the caller stopped reading.
fn run_generation(
    model: &LlamaModel,
    prompt: &str,
    n_ctx: u32,
    n_predict: i32,
    tx: &Sender<Result<String, LlmError>>,
) -> Result<(), LlmError> {
    let backend = backend();
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| LlmError::InferenceFailed(format!("create context: {e}")))?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|e| LlmError::InferenceFailed(format!("tokenize: {e}")))?;

    // Feed the prompt: only the last token needs its logits computed.
    let mut batch = LlamaBatch::new(512, 1);
    let last = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch
            .add(token, i, &[0], i == last)
            .map_err(|e| LlmError::InferenceFailed(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| LlmError::InferenceFailed(format!("decode prompt: {e}")))?;

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let eos = model.token_eos();
    let stop_at = n_cur + n_predict;

    while n_cur <= stop_at {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if token == eos || model.is_eog_token(token) {
            break;
        }

        // `special = false`: don't render control tokens as visible text.
        let piece = model
            .token_to_piece(token, &mut decoder, false, None)
            .map_err(|e| LlmError::InferenceFailed(format!("detokenize: {e}")))?;
        if !piece.is_empty() && tx.blocking_send(Ok(piece)).is_err() {
            return Ok(()); // receiver dropped
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| LlmError::InferenceFailed(format!("batch add: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::InferenceFailed(format!("decode: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("gemma")
            .join("gemma-4-12b-it-Q4_0.gguf")
    }

    #[test]
    fn gemma_prompt_uses_chat_template() {
        let p = format_gemma_prompt("Hi", "");
        assert!(p.starts_with("<start_of_turn>user\nHi<end_of_turn>"));
        assert!(p.ends_with("<start_of_turn>model\n"));
        let with_ctx = format_gemma_prompt("Q", "DOC");
        assert!(with_ctx.contains("DOC\n\nQ"));
    }

    #[tokio::test]
    async fn generates_text_on_gpu() {
        let path = model_path();
        if !path.exists() {
            eprintln!("skipping: {} not downloaded", path.display());
            return;
        }
        // Small generation budget for a fast smoke test.
        let cfg = LlamaConfig {
            n_predict: 24,
            ..LlamaConfig::default()
        };
        let engine = LlamaEngine::load(&path, cfg).expect("load Gemma");
        assert_eq!(engine.backend_id(), "llama-cpp-vulkan");

        let stream = engine
            .generate_text("Reply with exactly the word: hello", "")
            .await
            .expect("start generation");
        let output: String = stream
            .filter_map(|r| async move { r.ok() })
            .collect::<Vec<_>>()
            .await
            .concat();

        assert!(!output.trim().is_empty(), "model produced no text");
        eprintln!("Gemma said: {output:?}");
    }
}
