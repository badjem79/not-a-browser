pub mod ai;

use std::sync::Arc;

use futures::StreamExt;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::Mutex;

use crate::ai::engine::LlmEngine;
use crate::ai::llama::{LlamaConfig, LlamaEngine};

/// Dev-time model location (baked at compile time). In production the setup
/// wizard (UC-05) supplies this; for the spike we point at the downloaded GGUF.
const MODEL_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/models/gemma/gemma-4-12b-it-Q4_0.gguf");

/// App state: the generation engine, lazily loaded on first use (loading ~7 GB
/// into VRAM takes a few seconds, so we don't block app startup).
#[derive(Default)]
struct AppState {
    engine: Mutex<Option<Arc<LlamaEngine>>>,
}

impl AppState {
    /// Return the engine, loading it (off the async runtime) on first call.
    async fn engine(&self, app: &AppHandle) -> Result<Arc<LlamaEngine>, String> {
        let mut guard = self.engine.lock().await;
        if let Some(engine) = guard.as_ref() {
            return Ok(engine.clone());
        }
        let _ = app.emit("ai-status", "loading-model");
        let engine = tokio::task::spawn_blocking(|| {
            LlamaEngine::load(
                MODEL_PATH,
                LlamaConfig {
                    n_predict: 256,
                    ..LlamaConfig::default()
                },
            )
        })
        .await
        .map_err(|e| format!("load task panicked: {e}"))?
        .map_err(|e| e.to_string())?;
        let engine = Arc::new(engine);
        *guard = Some(engine.clone());
        Ok(engine)
    }
}

/// Ask the local model. Tokens are streamed back to the UI as `ai-token` events;
/// `ai-status` reports phase, `ai-done` signals completion, `ai-error` a failure.
#[tauri::command]
async fn ask(app: AppHandle, state: State<'_, AppState>, prompt: String) -> Result<(), String> {
    let engine = state.engine(&app).await?;
    let _ = app.emit("ai-status", "generating");

    let mut stream = engine
        .generate_text(&prompt, "")
        .await
        .map_err(|e| e.to_string())?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(token) => {
                let _ = app.emit("ai-token", token);
            }
            Err(e) => {
                let _ = app.emit("ai-error", e.to_string());
                break;
            }
        }
    }
    let _ = app.emit("ai-done", ());
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![ask])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
