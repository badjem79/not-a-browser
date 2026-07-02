pub mod ai;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use tauri::webview::WebviewBuilder;
use tauri::window::WindowBuilder;
use tauri::{
    AppHandle, Emitter, LogicalPosition, LogicalSize, Manager, State, WebviewUrl, WindowEvent,
};
use tokio::sync::Mutex;

use crate::ai::engine::LlmEngine;
use crate::ai::llama::{LlamaConfig, LlamaEngine};

/// Dev-time model location (baked at compile time). In production the setup
/// wizard (UC-05) supplies this; for the spike we point at the downloaded GGUF.
const MODEL_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/models/gemma/gemma-4-12b-it-Q4_0.gguf");

/// Height (logical px) of the top bar. The chrome webview spans the whole window
/// (so it can also draw the side dashboard); the content webview is inset below
/// this bar — the chrome shows through above it.
const TOPBAR_H: f64 = 64.0;

/// Width (logical px) of the left dashboard panel. When open, the content webview
/// is shifted right by this much so the chrome's left column (the panel) shows.
const PANEL_W: f64 = 300.0;

/// App state: the generation engine (lazily loaded — ~7 GB into VRAM) plus the
/// current layout flags, so the window resize handler can re-inset the content
/// webview correctly.
#[derive(Default)]
struct AppState {
    engine: Mutex<Option<Arc<LlamaEngine>>>,
    /// true while a page is shown (a node webview visible, inset below the bar).
    browsing: AtomicBool,
    /// true while the left dashboard panel is open (content shifted right).
    panel_open: AtomicBool,
    /// Id of the page-node whose webview is currently shown (label `page-<id>`).
    active_page: std::sync::Mutex<Option<String>>,
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
                    n_predict: 1024,
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

/// Logical inner size of the main window, accounting for HiDPI scale.
fn logical_inner(app: &AppHandle) -> Option<(f64, f64)> {
    let win = app.get_window("main")?;
    let size = win.inner_size().ok()?;
    let scale = win.scale_factor().ok()?;
    Some((size.width as f64 / scale, size.height as f64 / scale))
}

/// The rectangle (logical px) the content webview should occupy, given the
/// current panel state: below the top bar, and shifted right when the panel is open.
fn content_rect(app: &AppHandle) -> Option<(f64, f64, f64, f64)> {
    let (w, h) = logical_inner(app)?;
    let left = if app.state::<AppState>().panel_open.load(Ordering::SeqCst) {
        PANEL_W
    } else {
        0.0
    };
    Some((left, TOPBAR_H, (w - left).max(1.0), (h - TOPBAR_H).max(1.0)))
}

/// The label of the active page-node's webview, if any.
fn active_label(app: &AppHandle) -> Option<String> {
    app.state::<AppState>()
        .active_page
        .lock()
        .unwrap()
        .clone()
        .map(|id| format!("page-{id}"))
}

/// Hide every page-node webview except `keep`.
fn hide_other_pages(app: &AppHandle, keep: &str) {
    for (label, wv) in app.webviews() {
        if label.starts_with("page-") && label != keep {
            let _ = wv.hide();
        }
    }
}

/// Keep the webviews sized to the window. The chrome always fills the whole
/// window (it draws the top bar and, when open, the left dashboard); the active
/// page webview is inset so the chrome shows through above it and to its left.
fn relayout(app: &AppHandle) {
    let Some((w, h)) = logical_inner(app) else {
        return;
    };
    if let Some(chrome) = app.get_webview("chrome") {
        let _ = chrome.set_position(LogicalPosition::new(0.0, 0.0));
        let _ = chrome.set_size(LogicalSize::new(w, h));
    }
    if let (Some(label), Some((x, y, cw, ch))) = (active_label(app), content_rect(app)) {
        if let Some(wv) = app.get_webview(&label) {
            let _ = wv.set_position(LogicalPosition::new(x, y));
            let _ = wv.set_size(LogicalSize::new(cw, ch));
        }
    }
}

/// Show the page-node `id`, creating its webview (navigating to `url`) on first
/// use, or just revealing it (history/scroll preserved) if it already exists.
/// Every other page webview is hidden — instant switching between sessions.
#[tauri::command]
async fn show_page(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
    url: String,
) -> Result<(), String> {
    let label = format!("page-{id}");
    state.browsing.store(true, Ordering::SeqCst);
    *state.active_page.lock().unwrap() = Some(id);
    hide_other_pages(&app, &label);

    if let Some(wv) = app.get_webview(&label) {
        wv.show().map_err(|e| e.to_string())?;
    } else {
        let target: tauri::Url = url.parse().map_err(|e| format!("URL non valido: {e}"))?;
        let (x, y, cw, ch) = content_rect(&app).ok_or("finestra non disponibile")?;
        let win = app.get_window("main").ok_or("finestra non disponibile")?;
        win.add_child(
            WebviewBuilder::new(label.as_str(), WebviewUrl::External(target)),
            LogicalPosition::new(x, y),
            LogicalSize::new(cw, ch),
        )
        .map_err(|e| e.to_string())?;
    }

    relayout(&app);
    Ok(())
}

/// Permanently close a page-node's webview (used when pruning the tree).
#[tauri::command]
async fn close_page(app: AppHandle, state: State<'_, AppState>, id: String) -> Result<(), String> {
    if let Some(wv) = app.get_webview(&format!("page-{id}")) {
        let _ = wv.close();
    }
    let mut active = state.active_page.lock().unwrap();
    if active.as_deref() == Some(id.as_str()) {
        *active = None;
    }
    Ok(())
}

/// Return to the launcher / ask view: hide all page webviews (the full-window
/// chrome then shows the omnibox / answer again). Nodes persist for re-opening.
#[tauri::command]
async fn home(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    state.browsing.store(false, Ordering::SeqCst);
    for (label, wv) in app.webviews() {
        if label.starts_with("page-") {
            let _ = wv.hide();
        }
    }
    *state.active_page.lock().unwrap() = None;
    relayout(&app);
    Ok(())
}

/// Open/close the left dashboard panel. Re-insets the active page webview (the
/// panel itself is drawn by the chrome); when not browsing there's nothing to move.
#[tauri::command]
async fn set_panel(app: AppHandle, state: State<'_, AppState>, open: bool) -> Result<(), String> {
    state.panel_open.store(open, Ordering::SeqCst);
    relayout(&app);
    Ok(())
}

/// Drive the active page's session history (back / forward / reload) by running
/// JS in its webview — works across origins since it's the webview's own history.
#[tauri::command]
async fn page_nav(app: AppHandle, action: String) -> Result<(), String> {
    let label = active_label(&app).ok_or("nessuna pagina attiva")?;
    let wv = app.get_webview(&label).ok_or("pagina non trovata")?;
    let js = match action.as_str() {
        "back" => "history.back()",
        "forward" => "history.forward()",
        "reload" => "location.reload()",
        other => return Err(format!("azione sconosciuta: {other}")),
    };
    wv.eval(js).map_err(|e| e.to_string())
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
        .invoke_handler(tauri::generate_handler![
            ask, show_page, close_page, home, set_panel, page_nav
        ])
        .setup(|app| {
            let window = WindowBuilder::new(app, "main")
                .title("!aBrowser")
                .inner_size(1100.0, 720.0)
                .min_inner_size(720.0, 480.0)
                .build()?;

            let (w, h) = {
                let size = window.inner_size()?;
                let scale = window.scale_factor()?;
                (size.width as f64 / scale, size.height as f64 / scale)
            };

            // The chrome (our Svelte UI) starts filling the whole window — the
            // launcher state. The content webview is created lazily on first navigate.
            window.add_child(
                WebviewBuilder::new("chrome", WebviewUrl::App("index.html".into())),
                LogicalPosition::new(0.0, 0.0),
                LogicalSize::new(w, h),
            )?;

            // Keep the webviews sized to the window as it resizes.
            let handle = app.handle().clone();
            window.on_window_event(move |event| {
                if matches!(event, WindowEvent::Resized(_)) {
                    relayout(&handle);
                }
            });

            // Eager-load the model in the background so the first question is
            // answered immediately instead of paying a cold load on first ask.
            // If the user asks mid-load, `engine()` serializes on the same Mutex,
            // so the ask awaits this warm-up rather than starting a second load.
            let warm = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match warm.state::<AppState>().engine(&warm).await {
                    Ok(_) => {
                        let _ = warm.emit("model-ready", ());
                    }
                    Err(e) => {
                        let _ = warm.emit("ai-error", e);
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
