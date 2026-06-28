//! Speech synthesis — the **Speech** sensory channel (specs §3.1, UC-02).
//!
//! Resolves the open TTS decision (specs §9) in favour of **Piper** (VITS, ONNX):
//! lightweight, runs on the **CPU** via the same `ort` runtime as the MiniLM
//! embedder (so the GPU stays free for Gemma), good multilingual voices, fully
//! local. The [`Tts`] trait is the backend-independent seam, mirroring
//! [`crate::ai::engine::Embedder`].

use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use piper_rs::Piper;

use crate::ai::engine::LlmError;

/// A synthesized mono audio clip: PCM `f32` samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl TtsAudio {
    /// Duration of the clip in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    /// Encode as a 16-bit mono PCM WAV (RIFF) byte buffer, ready to write to disk
    /// or feed the HUD media player.
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        let data_len = (self.samples.len() * 2) as u32;
        let mut buf = Vec::with_capacity(44 + data_len as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio format: PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels: mono
        buf.extend_from_slice(&self.sample_rate.to_le_bytes());
        buf.extend_from_slice(&(self.sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for &s in &self.samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }
}

/// Backend-independent text-to-speech. Local and CPU-side, like the embedder.
#[async_trait]
pub trait Tts: Send + Sync {
    /// Synthesize `text` into a mono audio clip.
    async fn synthesize(&self, text: &str) -> Result<TtsAudio, LlmError>;

    /// Identifier of the active voice (e.g. `"it_IT-paola-medium"`).
    fn voice_id(&self) -> &str;
}

/// Piper TTS backend. The ONNX [`Piper`] session needs `&mut self` to run, so it
/// lives behind a [`Mutex`] (synthesis is serialized like the embedder).
pub struct PiperTts {
    piper: Mutex<Piper>,
    voice_id: String,
}

impl PiperTts {
    /// Load a Piper voice from its `model.onnx` and `model.onnx.json` config.
    pub fn from_files(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        voice_id: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let piper = Piper::new(model_path.as_ref(), config_path.as_ref())
            .map_err(|e| LlmError::InferenceFailed(format!("load piper voice: {e}")))?;
        Ok(Self {
            piper: Mutex::new(piper),
            voice_id: voice_id.into(),
        })
    }
}

#[async_trait]
impl Tts for PiperTts {
    async fn synthesize(&self, text: &str) -> Result<TtsAudio, LlmError> {
        let mut piper = self
            .piper
            .lock()
            .map_err(|_| LlmError::InferenceFailed("piper mutex poisoned".into()))?;
        // (text, is_phonemes=false, speaker_id, length_scale, noise_scale, noise_w)
        let (samples, sample_rate) = piper
            .create(text, false, None, None, None, None)
            .map_err(|e| LlmError::InferenceFailed(format!("synthesize: {e}")))?;
        Ok(TtsAudio {
            samples,
            sample_rate,
        })
    }

    fn voice_id(&self) -> &str {
        &self.voice_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn voice_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("piper")
    }

    #[test]
    fn wav_header_is_well_formed() {
        let audio = TtsAudio {
            samples: vec![0.0, 0.5, -0.5, 1.0],
            sample_rate: 22_050,
        };
        let wav = audio.to_wav_bytes();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 4 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 8);
    }

    #[tokio::test]
    async fn synthesizes_italian_speech() {
        let model = voice_dir().join("it_IT-paola-medium.onnx");
        let config = voice_dir().join("it_IT-paola-medium.onnx.json");
        if !model.exists() || !config.exists() {
            eprintln!("skipping: Piper voice not downloaded at {}", voice_dir().display());
            return;
        }
        let tts = PiperTts::from_files(&model, &config, "it_IT-paola-medium").expect("load voice");
        let audio = tts
            .synthesize("Ciao, sono il tuo browser. Come posso aiutarti?")
            .await
            .expect("synthesize");

        assert!(!audio.samples.is_empty(), "no audio produced");
        assert!(audio.sample_rate >= 16_000, "unexpected sample rate");

        // Write a WAV next to the voice so it can be listened to.
        let out = voice_dir().join("sample_output.wav");
        std::fs::write(&out, audio.to_wav_bytes()).expect("write wav");
        eprintln!(
            "synthesized {:.2}s of audio at {} Hz -> {}",
            audio.duration_secs(),
            audio.sample_rate,
            out.display()
        );
    }
}
