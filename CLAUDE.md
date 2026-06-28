# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**!aBrowser** ("not a browser" — `!` = the NOT operator; it's an AI shell over the OS WebView, not a real browser engine). The design spec is `specs.md` (v1.1, Italian) — the authoritative source of intent; read it fully, especially §9 (open decisions) and §10 (roadmap). Phase 0 (scaffolding) is **done**: a Tauri v2 + vanilla-TS app lives at the repo root and compiles clean.

Naming: product/display name is `!aBrowser`; the technical slug (crate, npm package, folder) is `not-a-browser` (the `!` is invalid in crate/npm names). Tauri identifier `com.notabrowser.app`.

**Build order is AI-core-first:** the AI Execution Engine + inference router are built and tested headless (no UI) before the browser shell. See `specs.md` §10. We are entering **Phase 1** (headless AI engine: `LlmEngine`/`Embedder` traits + router, llama.cpp Vulkan backend, MiniLM embedder, LanceDB RAG, Privacy Guard).

## Layout

The Tauri v2 app lives **at the repo root** (no subfolder). Docs (`specs.md`, `CLAUDE.md`) sit alongside it.

- `src/` — vanilla-TS frontend (Vite): `main.ts`, `styles.css`; `index.html` at root.
- `src-tauri/` — Rust core: `src/lib.rs` (app logic, `#[tauri::command]`s, `run()`), `src/main.rs` (thin entrypoint calling `not_a_browser_lib::run()`), `tauri.conf.json`, `capabilities/` (permissions), `Cargo.toml` (crate `not-a-browser`, lib `not_a_browser_lib`).

## Commands

Run from the repo root. `~/.cargo/bin` is on the persistent user PATH (new shells pick it up).

- `npm install` — frontend deps.
- `npm run tauri dev` — run the app (Vite dev server + Rust core, hot reload).
- `npm run tauri build` — production bundle.
- `cargo check` / `cargo build` / `cargo test` (from `src-tauri/`) — Rust-only compile/test, faster than a full Tauri build for headless engine work.
- `cargo test <name>` (from `src-tauri/`) — run a single Rust test.

Toolchain: Rust 1.96 (stable, MSVC), Node 24, MSVC Build Tools 2022 + Windows SDK, WebView2 runtime — all installed and verified.

## What !aBrowser is

A privacy-first web browser with local + hybrid AI woven into the browsing loop through three "sensory channels": **Vision** (vision-LLM), **Listening** (speech-to-text), and **Speech** (text-to-speech). Design goal: minimal disk/RAM footprint with GPU-accelerated inference and total local privacy. It is "not a browser" in that it rides the OS WebView rather than shipping its own engine.

## Intended architecture (from specs.md)

**Stack:** Rust (stable) + Tauri v2 core (tokio async), `wry` rendering (WebView2 on Windows / WebKitGTK on Linux / WKWebView on macOS), `llama.cpp` + `whisper.cpp` for the AI kernel, `LanceDB` (Apache Arrow columnar) as the embedded vector DB, and `all-MiniLM-L6-v2` (384-dim embeddings) run on CPU via ONNX Runtime / Burn. **GPU backend default is Vulkan** (reliable on consumer Radeon/Windows); ROCm/CUDA/Metal are loaded at runtime via `libloading`. **Vision/generation model is Gemma 4 12B-it QAT Q4_0** (official Google GGUF, Apache 2.0, encoder-free unified multimodal: text/image/audio≤30s/video input, text output; ~7 GB; 256K context, use 32K in practice). Vision in `llama.cpp` needs `llama-mtmd-cli` + a separate **mmproj** GGUF. **TTS engine is still undecided** (`specs.md` §9).

**Multi-process design** — the graphics process is deliberately isolated from the compute core so the WebView can stay smooth (144Hz target) while inference runs:
- **Core process (Rust):** a Tauri command router communicates over **MPSC channels** with an **AI Execution Engine** (llama.cpp vision model, whisper.cpp audio, LanceDB). Each request carries a `oneshot` return channel or emits Tauri events for token streaming back to the UI (MPSC alone is one-way). llama.cpp is not freely concurrent — the engine serializes requests through a single-/few-slot queue. Model weights are `mlock`ed into VRAM.
- **WebView process (`wry`):** the navigation UI (minimal top bar + collapsible side HUD) and the navigated web app.
- The two communicate via Tauri IPC.

**Hybrid inference router** — model access is abstracted behind a Rust trait so cloud fallback (e.g. Gemini Flash, DeepSeek) can transparently replace local inference on weak hardware. The seam is the `LlmEngine` trait (`generate_text`, `analyze_image` — both **streaming**, returning a `BoxStream`) with an `LlmError` enum. Embedding is a **separate** `Embedder` trait (CPU, backend-independent). New backends implement these traits; callers depend only on them. The router picks local-vs-cloud based on hardware, the current tab's consent, and the Privacy Guard's URL classification.

**GPU library loading** — CUDA/ROCm/Metal compute libraries are **not** statically linked. The setup wizard downloads the GPU-specific dynamic library (`.dll`/`.so`) and the app loads it at boot via `libloading` based on the detected GPU. Keep this dynamic-loading boundary intact; do not hardcode a single GPU backend.

**Privacy Guard (business rule, enforce in Rust before any embedding or remote call):** AI consent is granted **per browsing tab** (privacy-safe default), with a global "disable at your own risk" override. Every URL is classified and excluded from indexing/inference if it matches financial domains (bank, fineco, …), checkout pages, local addresses (`localhost`, `127.0.0.1`), or other sensitive strings — prefer categories + a maintained list over a bare regex blocklist. This gate must run *before* embedding or remote inference, never after, and cloud fallback is never used for non-consented tabs or blocked URLs. RAG history at rest holds sensitive cleartext and is slated for on-disk encryption (mechanism TBD, §9).

## Data model (LanceDB)

Three tables, each with a `VECTOR(384)` column and an `embedding_model_version` field (so vectors can be re-indexed when the embedding model changes), used by the "omnicomprehensive semantic history" (multimodal RAG) feature:
- `web_history` — clean page text extracted from the DOM via JS injection, plus url/timestamp.
- `image_history` — AI-generated descriptions of analyzed images, source url, local thumbnail path.
- `chat_history` — embedded prompt+response chunks with the active context url.

## Key use cases driving the design

- **UC-02 voice "hands-free":** mic activation → command on the page → spoken response, with a minimal media player in the HUD.
- **UC-03 multimodal vision:** `Alt+Drag` to select a screen region, or right-click an `<img>`, → AI transcribes/explains/translates the visual.
- **UC-04 semantic history:** background indexing of pages, image descriptions, and chats, queried via abstract natural-language search in the command bar.
- **UC-05 hardware wizard:** on first launch, detect PC architecture and download only the quantized models + compute libraries matching the detected GPU.
