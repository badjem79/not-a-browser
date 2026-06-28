//! Local CPU embedder: `all-MiniLM-L6-v2` via ONNX Runtime (specs §3.1).
//!
//! Produces the 384-dim sentence vectors stored in every LanceDB table
//! (specs §5). Backend-independent and never routed to the cloud.
//!
//! Pipeline: WordPiece tokenize → ONNX forward pass → mean-pool the token
//! embeddings weighted by the attention mask → L2-normalize. The normalization
//! means cosine similarity reduces to a dot product downstream.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ai::engine::{Embedder, LlmError, EMBEDDING_DIM};

/// Which MiniLM weights to load. Both produce 384-dim vectors; they differ in
/// size/speed vs. precision. The setup wizard (UC-05) lets the user pick:
/// quantized for maximum CPU performance, fp32 for maximum fidelity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniLmVariant {
    /// Full-precision float32 (~90 MB). Highest retrieval fidelity.
    Fp32,
    /// Dynamic INT8, AVX2-optimized (~23 MB). Faster, ~67 MB lighter on disk,
    /// with a negligible accuracy drop. Default — AVX2 is universal since 2013.
    Quint8Avx2,
}

impl MiniLmVariant {
    /// The variant chosen by default (smallest footprint, CPU-friendly).
    pub const DEFAULT: MiniLmVariant = MiniLmVariant::Quint8Avx2;

    /// File name of the ONNX weights for this variant.
    pub const fn filename(self) -> &'static str {
        match self {
            MiniLmVariant::Fp32 => "model.onnx",
            MiniLmVariant::Quint8Avx2 => "model_quint8_avx2.onnx",
        }
    }

    /// Version tag stored as `embedding_model_version` (specs §5). Distinct per
    /// variant so quantized and fp32 vectors are never mixed in one index —
    /// switching variants triggers a re-index.
    pub const fn version(self) -> &'static str {
        match self {
            MiniLmVariant::Fp32 => "all-MiniLM-L6-v2",
            MiniLmVariant::Quint8Avx2 => "all-MiniLM-L6-v2-quint8-avx2",
        }
    }
}

/// Version tag of the default embedder variant (see [`MiniLmVariant::version`]).
pub const MINILM_MODEL_VERSION: &str = MiniLmVariant::DEFAULT.version();

/// Default on-disk location of the model files, relative to the crate. The
/// hardware wizard (UC-05) will later download these into a user data dir.
pub fn default_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("minilm")
}

/// MiniLM embedder. The ONNX [`Session`] is behind a [`Mutex`] because
/// `Session::run` needs `&mut self`; the AI Execution Engine serializes
/// inference anyway, so contention is not a concern in practice.
pub struct MiniLmEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_version: String,
}

impl MiniLmEmbedder {
    /// Load the [`MiniLmVariant::DEFAULT`] weights + `tokenizer.json` from `dir`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, LlmError> {
        Self::from_dir_with(dir, MiniLmVariant::DEFAULT)
    }

    /// Load a specific `variant`'s weights + `tokenizer.json` from `dir`.
    pub fn from_dir_with(dir: impl AsRef<Path>, variant: MiniLmVariant) -> Result<Self, LlmError> {
        let dir = dir.as_ref();
        let model_path = dir.join(variant.filename());
        let tokenizer_path = dir.join("tokenizer.json");

        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| LlmError::InferenceFailed("non-UTF8 model path".into()))?;

        let session = Session::builder()
            .map_err(map_ort)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_ort)?
            .commit_from_file(model_path_str)
            .map_err(map_ort)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LlmError::InferenceFailed(format!("tokenizer load: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_version: variant.version().to_string(),
        })
    }

    /// Synchronous forward pass. Kept private; the async trait method wraps it.
    fn embed_sync(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LlmError::InferenceFailed(format!("tokenize: {e}")))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&i| i as i64)
            .collect();
        let seq_len = ids.len();
        let type_ids = vec![0i64; seq_len];

        let shape = [1usize, seq_len];
        let input_ids = Tensor::from_array((shape, ids)).map_err(map_ort)?;
        let attention_mask = Tensor::from_array((shape, mask.clone())).map_err(map_ort)?;
        let token_type_ids = Tensor::from_array((shape, type_ids)).map_err(map_ort)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| LlmError::InferenceFailed("embedder mutex poisoned".into()))?;

        let output_name = session.outputs()[0].name().to_string();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
            .map_err(map_ort)?;

        // Token embeddings, flat: shape [batch=1, seq_len, hidden=384].
        let (out_shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(map_ort)?;
        if out_shape.len() != 3 || out_shape[2] as usize != EMBEDDING_DIM {
            return Err(LlmError::InferenceFailed(format!(
                "unexpected model output shape {out_shape:?}, expected [1, seq, {EMBEDDING_DIM}]"
            )));
        }

        Ok(mean_pool_normalize(data, seq_len, EMBEDDING_DIM, &mask))
    }
}

#[async_trait]
impl Embedder for MiniLmEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        // CPU-bound and fast; runs inline. The engine serializes inference, so
        // this does not stall concurrent generation. Offload to spawn_blocking
        // if it ever shares a runtime with latency-sensitive work.
        self.embed_sync(text)
    }

    fn model_version(&self) -> &str {
        &self.model_version
    }
}

/// Mean-pool token embeddings weighted by the attention mask, then L2-normalize.
///
/// `hidden` is the flat `[1, seq_len, dim]` row-major buffer, so token `t`'s
/// embedding component `d` lives at `t * dim + d`.
fn mean_pool_normalize(hidden: &[f32], seq_len: usize, dim: usize, mask: &[i64]) -> Vec<f32> {
    let mut pooled = vec![0.0f32; dim];
    let mut mask_sum = 0.0f32;
    for t in 0..seq_len {
        let m = mask[t] as f32;
        if m == 0.0 {
            continue;
        }
        mask_sum += m;
        let base = t * dim;
        for d in 0..dim {
            pooled[d] += hidden[base + d] * m;
        }
    }
    let denom = mask_sum.max(1e-9);
    for v in pooled.iter_mut() {
        *v /= denom;
    }

    let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for v in pooled.iter_mut() {
        *v /= norm;
    }
    pooled
}

fn map_ort<T>(e: ort::Error<T>) -> LlmError {
    LlmError::InferenceFailed(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Load the embedder, skipping (not failing) if the model isn't present.
    fn load() -> Option<MiniLmEmbedder> {
        let dir = default_model_dir();
        let model = dir.join(MiniLmVariant::DEFAULT.filename());
        if !model.exists() {
            eprintln!("skipping: {} not downloaded", model.display());
            return None;
        }
        Some(MiniLmEmbedder::from_dir(&dir).expect("load MiniLM"))
    }

    #[test]
    fn default_variant_is_quantized() {
        assert_eq!(MiniLmVariant::DEFAULT, MiniLmVariant::Quint8Avx2);
        assert_eq!(MINILM_MODEL_VERSION, "all-MiniLM-L6-v2-quint8-avx2");
    }

    #[tokio::test]
    async fn quantized_vector_matches_fp32_closely() {
        // Cross-check: if both variants are present, the INT8 vector should be
        // nearly identical to fp32 (cosine > 0.99) — the "minimal imprecision".
        let dir = default_model_dir();
        if !dir.join(MiniLmVariant::Fp32.filename()).exists()
            || !dir.join(MiniLmVariant::Quint8Avx2.filename()).exists()
        {
            return;
        }
        let fp32 = MiniLmEmbedder::from_dir_with(&dir, MiniLmVariant::Fp32).unwrap();
        let q8 = MiniLmEmbedder::from_dir_with(&dir, MiniLmVariant::Quint8Avx2).unwrap();
        let text = "background indexing of a browsing session";
        let a = fp32.embed(text).await.unwrap();
        let b = q8.embed(text).await.unwrap();
        let cos = dot(&a, &b); // both unit-normalized
        assert!(cos > 0.99, "fp32 vs quint8 cosine too low: {cos:.4}");
    }

    #[tokio::test]
    async fn embeds_to_unit_384_vector() {
        let Some(emb) = load() else { return };
        let v = emb.embed("the quick brown fox").await.unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        let norm: f32 = dot(&v, &v).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit vector, got {norm}");
    }

    #[tokio::test]
    async fn semantically_similar_texts_score_higher() {
        let Some(emb) = load() else { return };
        let cat = emb.embed("a small domestic cat").await.unwrap();
        let kitten = emb.embed("a young kitten playing").await.unwrap();
        let finance = emb.embed("quarterly interest rate policy").await.unwrap();

        let related = dot(&cat, &kitten); // cosine, vectors are normalized
        let unrelated = dot(&cat, &finance);
        assert!(
            related > unrelated,
            "cat~kitten ({related:.3}) should beat cat~finance ({unrelated:.3})"
        );
    }
}
