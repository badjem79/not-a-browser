//! Local generation backend: Gemma via `llama.cpp` with the **Vulkan** backend
//! (specs §3.1). Implements [`LlmEngine::generate_text`] with token streaming.
//!
//! `llama.cpp` inference is synchronous and its context is single-threaded, so
//! each request runs the decode loop on a dedicated OS thread and streams tokens
//! back over a tokio MPSC channel — mirroring the AI Execution Engine design in
//! specs §3.2 (serialized inference, incremental tokens to the UI). The returned
//! [`TokenStream`] is just the receiving end of that channel.
//!
//! Native multimodality (Gemma 4 is encoder-free, omni): [`LlamaEngine::analyze_image`]
//! and [`LlamaEngine::analyze_audio`] feed images / audio (≤30 s) through the
//! `mmproj` projector via llama.cpp's `mtmd` interface, sharing the same decode
//! loop as text generation. Long audio is handled by chunking (see
//! [`LlamaEngine::analyze_audio`]).

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::mpsc::Sender;
use tokio_stream::wrappers::ReceiverStream;

use crate::ai::engine::{ChatRole, ChatTurn, LlmEngine, LlmError, TokenStream};

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
    /// Context window in tokens (KV cache size). 8192 gives multi-turn chats
    /// room to grow; Gemma supports far more, at higher VRAM cost (specs §4.1).
    pub n_ctx: u32,
    /// Logical batch size for prompt/media prefill.
    pub n_batch: i32,
    /// Maximum number of tokens to generate per request.
    pub n_predict: i32,
    /// Path to the multimodal projector (`mmproj`) GGUF. When set, enables
    /// [`LlamaEngine::analyze_image`] / [`LlamaEngine::analyze_audio`].
    pub mmproj_path: Option<PathBuf>,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 999,
            n_ctx: 8192,
            n_batch: 512,
            n_predict: 512,
            mmproj_path: None,
        }
    }
}

/// Gemma generation backend over llama.cpp/Vulkan.
pub struct LlamaEngine {
    // `mtmd` is declared before `model` so it is dropped first: its C context
    // references the model's weights and must not outlive them.
    mtmd: Option<Arc<MtmdContext>>,
    model: Arc<LlamaModel>,
    config: LlamaConfig,
    backend_id: String,
}

impl LlamaEngine {
    /// Load a GGUF model from disk and offload it to the GPU. This is heavy
    /// (reads ~7 GB and uploads weights to VRAM); call it off the UI thread.
    /// When `config.mmproj_path` is set, the multimodal projector is loaded too.
    pub fn load(model_path: impl AsRef<Path>, config: LlamaConfig) -> Result<Self, LlmError> {
        let backend = backend();
        // `use_mmap(false)`: with full GPU offload the weights live in VRAM, so we
        // don't want the ~7 GB GGUF also kept resident as an mmap'd page cache in
        // RAM. Disabling mmap frees the host-side copy after upload to VRAM —
        // lower steady-state RAM (the project's minimal-footprint goal) at the
        // cost of a slightly slower cold load (the file is read instead of lazily
        // mapped). Weights stay in VRAM for the whole session regardless.
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers)
            .with_use_mmap(false);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| LlmError::InferenceFailed(format!("load model: {e}")))?;

        let mtmd = match &config.mmproj_path {
            Some(path) => {
                let path = path
                    .to_str()
                    .ok_or_else(|| LlmError::InferenceFailed("non-UTF8 mmproj path".into()))?;
                let marker = CString::new(mtmd_default_marker())
                    .map_err(|e| LlmError::InferenceFailed(format!("marker: {e}")))?;
                let params = MtmdContextParams {
                    use_gpu: true,
                    print_timings: false,
                    n_threads: 4,
                    media_marker: marker,
                    image_min_tokens: -1,
                    image_max_tokens: -1,
                };
                let ctx = MtmdContext::init_from_file(path, &model, &params)
                    .map_err(|e| LlmError::InferenceFailed(format!("load mmproj: {e:?}")))?;
                Some(Arc::new(ctx))
            }
            None => None,
        };

        Ok(Self {
            mtmd,
            model: Arc::new(model),
            config,
            backend_id: "llama-cpp-vulkan".to_string(),
        })
    }

    /// Whether multimodal (image/audio) input is available (an mmproj was loaded).
    pub fn supports_multimodal(&self) -> bool {
        self.mtmd.is_some()
    }

    /// Analyze a short audio clip (≤30 s). Decodes wav/mp3/flac from `audio_bytes`
    /// and streams the model's text response (e.g. a transcription or an answer to
    /// a spoken command). For audio longer than the model's native window, split it
    /// into ≤30 s PCM windows and call this per window, concatenating the results
    /// (see `chunk_audio_pcm`).
    pub async fn analyze_audio(
        &self,
        audio_bytes: &[u8],
        prompt: &str,
    ) -> Result<TokenStream, LlmError> {
        let bytes = audio_bytes.to_vec();
        let instruction = if prompt.trim().is_empty() {
            "Transcribe this audio verbatim."
        } else {
            prompt
        };
        let full_prompt = format_gemma4_media_prompt(instruction);
        self.spawn_mtmd(full_prompt, move |ctx| {
            MtmdBitmap::from_buffer(ctx, &bytes, false)
                .map_err(|e| LlmError::InferenceFailed(format!("decode audio: {e:?}")))
        })
    }

    /// Shared multimodal entry point: build a bitmap (inside the worker thread,
    /// where the `MtmdContext` lives), then run the mtmd prefill + decode loop,
    /// streaming tokens back.
    /// Run a fully-formatted text prompt on a background thread, streaming tokens
    /// back over an MPSC channel wrapped as a [`TokenStream`]. Shared by the
    /// single-turn (`generate_text`) and multi-turn (`generate_chat`) paths.
    fn spawn_generation(&self, full_prompt: String) -> TokenStream {
        let model = self.model.clone();
        let n_ctx = self.config.n_ctx;
        let n_batch = self.config.n_batch;
        let n_predict = self.config.n_predict;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, LlmError>>(64);
        std::thread::spawn(move || {
            if let Err(e) = run_generation(&model, &full_prompt, n_ctx, n_batch, n_predict, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });
        Box::pin(ReceiverStream::new(rx))
    }

    fn spawn_mtmd<F>(&self, full_prompt: String, build_bitmap: F) -> Result<TokenStream, LlmError>
    where
        F: FnOnce(&MtmdContext) -> Result<MtmdBitmap, LlmError> + Send + 'static,
    {
        let mtmd = self
            .mtmd
            .clone()
            .ok_or_else(|| LlmError::InferenceFailed("multimodal disabled: no mmproj loaded".into()))?;
        let model = self.model.clone();
        let n_ctx = self.config.n_ctx;
        let n_batch = self.config.n_batch;
        let n_predict = self.config.n_predict;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, LlmError>>(64);
        std::thread::spawn(move || {
            let result = build_bitmap(&mtmd).and_then(|bitmap| {
                run_mtmd_generation(&model, &mtmd, &full_prompt, bitmap, n_ctx, n_batch, n_predict, &tx)
            });
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}

/// Split interleaved/mono PCM `f32` samples into ≤`window_secs`-second windows at
/// `sample_rate` Hz. Each window can be turned into an audio bitmap
/// (`MtmdBitmap::from_audio_data`) and transcribed independently, then the texts
/// concatenated — the basis for long-audio transcription within Gemma 4's 30 s
/// native limit.
pub fn chunk_audio_pcm(samples: &[f32], sample_rate: u32, window_secs: u32) -> Vec<&[f32]> {
    let window = (sample_rate as usize) * (window_secs.max(1) as usize);
    if window == 0 || samples.is_empty() {
        return Vec::new();
    }
    samples.chunks(window).collect()
}

#[async_trait]
impl LlmEngine for LlamaEngine {
    async fn generate_text(&self, prompt: &str, context: &str) -> Result<TokenStream, LlmError> {
        Ok(self.spawn_generation(format_gemma4_prompt(prompt, context)))
    }

    async fn generate_chat(
        &self,
        history: &[ChatTurn],
        context: &str,
    ) -> Result<TokenStream, LlmError> {
        Ok(self.spawn_generation(format_gemma4_chat(history, context)))
    }

    async fn analyze_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<TokenStream, LlmError> {
        let bytes = image_bytes.to_vec();
        let instruction = if prompt.trim().is_empty() {
            "Describe this image."
        } else {
            prompt
        };
        let full_prompt = format_gemma4_media_prompt(instruction);
        self.spawn_mtmd(full_prompt, move |ctx| {
            MtmdBitmap::from_buffer(ctx, &bytes, false)
                .map_err(|e| LlmError::InferenceFailed(format!("decode image: {e:?}")))
        })
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }
}

/// Wrap a prompt in Gemma 4's chat format, with **thinking disabled**.
///
/// Gemma 4's template (the `enable_thinking=false` branch) pre-fills an *empty*
/// `thought` channel — `<|channel>thought\n<channel|>` — at the end of the
/// prompt, which makes the model skip reasoning and emit only the final answer.
/// Without this priming the model generates the channel markup itself and it
/// leaks into the output. Any RAG `context` is prepended to the user turn
/// (Gemma has no dedicated system role). Tokenize with `parse_special=true` so
/// the `<|turn>` / `<|channel>` control strings become their special tokens.
fn format_gemma4_prompt(prompt: &str, context: &str) -> String {
    format_gemma4_chat(
        &[ChatTurn {
            role: ChatRole::User,
            text: prompt.to_string(),
        }],
        context,
    )
}

/// Wrap a full multi-turn conversation in Gemma 4's chat format, thinking
/// disabled. `history` is the ordered exchange ending with the latest user
/// turn; each turn becomes its own `<|turn>{role}…<turn|>` block so the model
/// truly sees the conversation (not just the last question). `context` (ancestor
/// grounding) is prepended to the **first** user turn — Gemma has no system role.
/// A fresh assistant turn is opened at the end with the primed empty thought
/// channel so generation starts on the answer.
fn format_gemma4_chat(history: &[ChatTurn], context: &str) -> String {
    let mut s = String::new();
    let mut first_user = true;
    for turn in history {
        match turn.role {
            ChatRole::User => {
                let body = if first_user && !context.trim().is_empty() {
                    format!("{context}\n\n{}", turn.text)
                } else {
                    turn.text.clone()
                };
                first_user = false;
                s.push_str(&format!("<|turn>user\n{body}<turn|>\n"));
            }
            ChatRole::Model => {
                s.push_str(&format!("<|turn>model\n{}<turn|>\n", turn.text));
            }
        }
    }
    s.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    s
}

/// Gemma 4 prompt for a single media input. The `<__media__>` marker is replaced
/// by the image/audio tokens during `mtmd` tokenization. Thinking is disabled
/// (primed empty thought) so the model answers directly.
fn format_gemma4_media_prompt(instruction: &str) -> String {
    let marker = mtmd_default_marker();
    format!(
        "<|turn>user\n{marker}\n{instruction}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )
}

/// Text generation: tokenize + prefill the prompt, then run the decode loop.
///
/// Prefill is done in `n_batch`-sized chunks so a long (multi-turn) prompt can
/// be far larger than a single batch — feeding it all at once is what overflowed
/// the old fixed-512 batch ("Insufficient Space of 512"). `n_predict` is clamped
/// so prompt + reply always fit the `n_ctx` KV cache.
fn run_generation(
    model: &LlamaModel,
    prompt: &str,
    n_ctx: u32,
    n_batch: i32,
    n_predict: i32,
    tx: &Sender<Result<String, LlmError>>,
) -> Result<(), LlmError> {
    let backend = backend();
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch.max(1) as u32);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| LlmError::InferenceFailed(format!("create context: {e}")))?;

    // The chat template may already emit a literal `<bos>`; avoid doubling it.
    let add_bos = if prompt.trim_start().starts_with("<bos>") {
        AddBos::Never
    } else {
        AddBos::Always
    };
    let tokens = model
        .str_to_token(prompt, add_bos)
        .map_err(|e| LlmError::InferenceFailed(format!("tokenize: {e}")))?;

    // Leave room for at least a short reply; refuse rather than crash mid-decode
    // when the KV cache is full.
    let room = n_ctx as i32 - tokens.len() as i32 - 1;
    if room < 16 {
        return Err(LlmError::InferenceFailed(format!(
            "conversazione troppo lunga per il contesto da {n_ctx} token \
             ({} token nel prompt) — apri una nuova chat o aumenta n_ctx",
            tokens.len()
        )));
    }
    let n_predict = n_predict.min(room);

    // Prefill in chunks; only the very last token needs its logits computed
    // (that's where sampling starts).
    let cap = n_batch.max(1) as usize;
    let mut batch = LlamaBatch::new(cap, 1);
    let last = tokens.len() - 1;
    let mut pos = 0usize;
    while pos < tokens.len() {
        let end = (pos + cap).min(tokens.len());
        batch.clear();
        for i in pos..end {
            batch
                .add(tokens[i], i as i32, &[0], i == last)
                .map_err(|e| LlmError::InferenceFailed(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::InferenceFailed(format!("decode prompt: {e}")))?;
        pos = end;
    }

    decode_loop(model, &mut ctx, tokens.len() as i32, n_predict, tx)
}

/// Multimodal generation: prefill text+media chunks via `mtmd`, then decode.
fn run_mtmd_generation(
    model: &LlamaModel,
    mtmd: &MtmdContext,
    prompt: &str,
    bitmap: MtmdBitmap,
    n_ctx: u32,
    n_batch: i32,
    n_predict: i32,
    tx: &Sender<Result<String, LlmError>>,
) -> Result<(), LlmError> {
    let backend = backend();
    // Image/audio embeddings are submitted as one non-causal ubatch, so the
    // (u)batch must be large enough to hold them — Gemma 4 vision can emit up to
    // ~1120 tokens for a single image, well over the default 512 (which aborts).
    let mm_batch = n_batch.max(2048) as u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(mm_batch)
        .with_n_ubatch(mm_batch);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| LlmError::InferenceFailed(format!("create context: {e}")))?;

    let input = MtmdInputText {
        text: prompt.to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd
        .tokenize(input, &[&bitmap])
        .map_err(|e| LlmError::InferenceFailed(format!("mtmd tokenize: {e:?}")))?;

    // Prefill all chunks (text + encoded media embeddings) into the KV cache.
    let n_past = chunks
        .eval_chunks(mtmd, &ctx, 0, 0, mm_batch as i32, true)
        .map_err(|e| LlmError::InferenceFailed(format!("mtmd eval: {e:?}")))?;

    decode_loop(model, &mut ctx, n_past, n_predict, tx)
}

/// Greedy autoregressive decode, streaming each detokenized piece into `tx`.
/// Shared by text and multimodal paths; starts sampling from the last prefilled
/// position. Returns early (Ok) if the receiver is dropped.
fn decode_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    mut n_past: i32,
    n_predict: i32,
    tx: &Sender<Result<String, LlmError>>,
) -> Result<(), LlmError> {
    let mut batch = LlamaBatch::new(512, 1);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let eos = model.token_eos();
    let stop_at = n_past + n_predict;

    while n_past <= stop_at {
        // `-1`: sample from the logits of the last evaluated position.
        let token = sampler.sample(ctx, -1);
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
            .add(token, n_past, &[0], true)
            .map_err(|e| LlmError::InferenceFailed(format!("batch add: {e}")))?;
        n_past += 1;
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

    fn gemma_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("gemma")
    }
    fn model_path() -> PathBuf {
        gemma_dir().join("gemma-4-12b-it-Q4_0.gguf")
    }
    fn mmproj_path() -> PathBuf {
        gemma_dir().join("mmproj-F16.gguf")
    }

    #[test]
    fn audio_chunking_splits_into_windows() {
        let sr = 16_000;
        // 70 s of mono samples → three 30 s windows (30 + 30 + 10).
        let samples = vec![0.0f32; sr as usize * 70];
        let chunks = chunk_audio_pcm(&samples, sr, 30);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), sr as usize * 30);
        assert_eq!(chunks[2].len(), sr as usize * 10);
        assert!(chunk_audio_pcm(&[], sr, 30).is_empty());
    }

    #[test]
    fn media_prompt_has_marker_and_primes_thought() {
        let p = format_gemma4_media_prompt("Describe this image.");
        assert!(p.contains(mtmd_default_marker()));
        assert!(p.ends_with("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn gemma4_prompt_primes_empty_thought_channel() {
        let p = format_gemma4_prompt("Hi", "");
        assert!(p.starts_with("<|turn>user\nHi<turn|>"));
        // Thinking disabled: prompt ends with a pre-filled empty thought channel
        // so the model emits only the final answer.
        assert!(p.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
        let with_ctx = format_gemma4_prompt("Q", "DOC");
        assert!(with_ctx.contains("DOC\n\nQ"));
    }

    #[test]
    fn gemma4_chat_includes_every_prior_turn() {
        let history = vec![
            ChatTurn { role: ChatRole::User, text: "Come ti chiami?".into() },
            ChatTurn { role: ChatRole::Model, text: "Sono Gemma.".into() },
            ChatTurn { role: ChatRole::User, text: "E quanti anni hai?".into() },
        ];
        let p = format_gemma4_chat(&history, "");
        // All three turns are present, in order, each in its own turn block:
        // the model can only answer the follow-up if it sees the earlier ones.
        assert!(p.contains("<|turn>user\nCome ti chiami?<turn|>"));
        assert!(p.contains("<|turn>model\nSono Gemma.<turn|>"));
        assert!(p.contains("<|turn>user\nE quanti anni hai?<turn|>"));
        let first = p.find("Come ti chiami?").unwrap();
        let second = p.find("Sono Gemma.").unwrap();
        let third = p.find("E quanti anni hai?").unwrap();
        assert!(first < second && second < third, "turns must stay ordered");
        // Ends primed for a fresh answer with thinking disabled.
        assert!(p.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
        // Context lands on the FIRST user turn only.
        let with_ctx = format_gemma4_chat(&history, "CTX");
        assert!(with_ctx.contains("<|turn>user\nCTX\n\nCome ti chiami?<turn|>"));
        assert_eq!(with_ctx.matches("CTX").count(), 1);
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
            n_predict: 64,
            ..LlamaConfig::default()
        };
        let engine = LlamaEngine::load(&path, cfg).expect("load Gemma");
        assert_eq!(engine.backend_id(), "llama-cpp-vulkan");

        let stream = engine
            .generate_text("How many letter r are in the word strawberry? Answer briefly.", "")
            .await
            .expect("start generation");
        let output: String = stream
            .filter_map(|r| async move { r.ok() })
            .collect::<Vec<_>>()
            .await
            .concat();

        eprintln!("Gemma said: {output:?}");
        assert!(!output.trim().is_empty(), "model produced no text");
        // With thinking disabled, the reasoning-channel markup must NOT leak.
        assert!(
            !output.contains("<|channel>") && !output.contains("<channel|>"),
            "reasoning-channel markup leaked into output: {output:?}"
        );
    }

    // mtmd's image/audio decoders trip a debug-CRT assertion; run multimodal
    // tests in release: `cargo test --release --lib analyzes_image_on_gpu -- --ignored`.
    #[tokio::test]
    #[ignore = "multimodal needs a release build (debug-CRT assertion in stb/miniaudio)"]
    async fn analyzes_image_on_gpu() {
        if !model_path().exists() || !mmproj_path().exists() {
            eprintln!("skipping: Gemma model + mmproj not downloaded");
            return;
        }
        let cfg = LlamaConfig {
            n_predict: 32,
            mmproj_path: Some(mmproj_path()),
            ..LlamaConfig::default()
        };
        let engine = LlamaEngine::load(&model_path(), cfg).expect("load Gemma + mmproj");
        assert!(engine.supports_multimodal());

        // A 64×64 solid-red RGB image, fed as a raw bitmap (no PNG decode needed).
        let red: Vec<u8> = std::iter::repeat([255u8, 0, 0])
            .take(64 * 64)
            .flatten()
            .collect();
        let prompt = format_gemma4_media_prompt("What is the main color of this image? One word.");
        let stream = engine
            .spawn_mtmd(prompt, move |_ctx| {
                MtmdBitmap::from_image_data(64, 64, &red)
                    .map_err(|e| LlmError::InferenceFailed(format!("{e:?}")))
            })
            .expect("start image analysis");
        let output: String = stream
            .filter_map(|r| async move { r.ok() })
            .collect::<Vec<_>>()
            .await
            .concat();

        eprintln!("vision said: {output:?}");
        assert!(!output.trim().is_empty(), "vision produced no text");
        if !output.to_lowercase().contains("red") {
            eprintln!("note: model did not say 'red' for a solid-red image");
        }
    }

    /// Demo over real media in `<repo>/tests/` (PNG screenshots + MP3 clips).
    /// Prints what Gemma 4 makes of each; run with:
    ///   cargo test --lib describe_real_media -- --nocapture --ignored
    #[tokio::test]
    #[ignore = "exploratory demo; needs model, mmproj, and tests/ media"]
    async fn describe_real_media() {
        if !model_path().exists() || !mmproj_path().exists() {
            eprintln!("skipping: Gemma model + mmproj not downloaded");
            return;
        }
        let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests");
        if !tests_dir.exists() {
            eprintln!("skipping: {} not present", tests_dir.display());
            return;
        }

        let cfg = LlamaConfig {
            n_predict: 200,
            mmproj_path: Some(mmproj_path()),
            ..LlamaConfig::default()
        };
        let engine = LlamaEngine::load(&model_path(), cfg).expect("load Gemma + mmproj");

        async fn drain(stream: TokenStream) -> String {
            stream
                .filter_map(|r| async move {
                    match r {
                        Ok(t) => Some(t),
                        Err(e) => Some(format!("[ERR: {e}]")),
                    }
                })
                .collect::<Vec<_>>()
                .await
                .concat()
        }

        // Every .png / .mp3 in tests/, sorted for stable order.
        let mut files: Vec<_> = std::fs::read_dir(&tests_dir)
            .expect("read tests dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                    Some(ref e) if e == "png" || e == "mp3"
                )
            })
            .collect();
        files.sort();

        for path in files {
            let bytes = std::fs::read(&path).expect("read media");
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            eprintln!("\n=========== {name} ({} bytes) ===========", bytes.len());

            let result = if ext.eq_ignore_ascii_case("png") {
                engine
                    .analyze_image(&bytes, "Describe what you see in this image in detail.")
                    .await
            } else {
                engine.analyze_audio(&bytes, "Transcribe this audio verbatim.").await
            };
            match result {
                Ok(stream) => eprintln!("→ {}", drain(stream).await.trim()),
                Err(e) => eprintln!("→ start error: {e}"),
            }
        }
    }
}
