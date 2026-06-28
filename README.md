# !aBrowser

> **!aBrowser** — *"not a browser"* (the `!` is the **NOT** operator). It is **not** a browser engine: it's a privacy-first AI shell that rides the operating system's WebView, with local + hybrid AI woven into the browsing loop through three sensory channels — **Vision** (vision-LLM), **Listening** (speech-to-text), and **Speech** (text-to-speech).

Local-first, GPU-accelerated inference; nothing leaves the machine without per-tab consent. Built in Rust + Tauri v2 over `wry`/WebView2, with `llama.cpp` (Vulkan) for generation/vision, `LanceDB` for an "omnicomprehensive semantic history" (multimodal RAG), and `all-MiniLM-L6-v2` (ONNX, CPU) for embeddings.

The authoritative design spec is [`specs.md`](./specs.md) (Italian). Architecture and contributor notes are in [`CLAUDE.md`](./CLAUDE.md).

**Status:** building AI-core-first (headless engine before the browser shell). Phase 1 (AI Execution Engine) is largely implemented and tested headless: the `LlmEngine`/`Embedder` trait seam + hybrid router, the Privacy Guard, the MiniLM embedder, the LanceDB RAG pipeline, and the Gemma/llama.cpp Vulkan generation backend.

## License

Licensed under the **[Apache License 2.0](./LICENSE)** (see also [`NOTICE`](./NOTICE)). Note that **model weights** (Gemma 4, MiniLM) carry their **own** licenses, separate from this repository's code, and are downloaded at setup time — they are never committed here.

---

## Architecture at a glance

```
src/                     # Vanilla-TS frontend (Vite): main.ts, styles.css
src-tauri/               # Rust core
  src/lib.rs             # Tauri commands + run()
  src/ai/
    engine.rs            # LlmEngine / Embedder traits + LlmError (the seam)
    privacy.rs           # Privacy Guard: per-tab consent + URL classification
    router.rs            # Hybrid local/cloud inference router (Guard-gated)
    embedder.rs          # MiniLM embedder (ONNX Runtime, CPU, 384-dim)
    llama.rs             # Gemma generation backend (llama.cpp, Vulkan, streaming)
    rag/                 # LanceDB semantic history (schema, store, pipeline)
  models/                # Downloaded weights (gitignored, see below)
```

The AI Execution Engine is deliberately isolated from the graphics process so the WebView stays smooth while inference runs. Generation streams tokens back over channels; the Privacy Guard runs **before** any embedding, local inference, or cloud call.

---

## Building on Windows

> The AI stack compiles `llama.cpp` (with Vulkan shaders) and `lance`/`onnxruntime` from source via build scripts, so the toolchain is heavier than a typical Rust project. Everything below is required for a full build of the GPU backend. (Linux/macOS instructions are TODO — contributions welcome.)

### 1. Prerequisites

| Tool | Why | Notes |
|------|-----|-------|
| **Rust** (stable, MSVC) ≥ 1.96 | core | `rustup` default `x86_64-pc-windows-msvc` |
| **Node** ≥ 24 | frontend | |
| **Visual Studio 2022 Build Tools** | C/C++ compiler + Windows SDK | "Desktop development with C++" workload |
| **CMake** ≥ 3.21 | builds `llama.cpp` | `winget install Kitware.CMake` |
| **Ninja** | CMake generator (avoids VS-generator issues) | `winget install Ninja-build.Ninja` |
| **LLVM / Clang** | `libclang` for `bindgen` | `winget install LLVM.LLVM` |
| **Vulkan SDK** | builds the GPU backend (`glslc` shader compiler) | `winget install KhronosGroup.VulkanSDK` |
| **protoc** | Protobuf, for `lancedb`/`lance` | [protobuf releases](https://github.com/protocolbuffers/protobuf/releases) → set `PROTOC` |
| GPU + driver with **Vulkan** | runtime inference | e.g. AMD Radeon (Vulkan is the default backend) |

### 2. Long-path note (important)

`llama.cpp`'s nested CMake build produces paths that exceed Windows' 260-char `MAX_PATH`, which breaks compilation. Do **one** of:

- Enable long paths (admin, once):
  ```powershell
  Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" LongPathsEnabled 1
  ```
- **and/or** build under a short target dir, e.g. map a drive to `target/`:
  ```powershell
  subst B: "C:\path\to\not-a-browser\src-tauri\target"
  $env:CARGO_TARGET_DIR = "B:\"
  ```

### 3. Build environment

The build needs the MSVC environment (`vcvars64`) plus several env vars. A reusable PowerShell setup:

```powershell
# Import the MSVC build environment (INCLUDE / LIB / cl.exe)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}

$env:CMAKE_GENERATOR = "Ninja"
$env:LIBCLANG_PATH   = "C:\Program Files\LLVM\bin"
$env:VULKAN_SDK      = "C:\VulkanSDK\<version>"
$env:PROTOC          = "C:\path\to\protoc\bin\protoc.exe"
$env:CARGO_TARGET_DIR = "B:\"      # see long-path note above
```

### 4. Download models

Weights are gitignored and fetched into `src-tauri/models/`.

```powershell
# Embedder: all-MiniLM-L6-v2 (ONNX, CPU) -> src-tauri/models/minilm/
#   model_quint8_avx2.onnx (default, ~23 MB) + tokenizer.json
#   from huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Generation: Gemma 4 12B-it Q4_0 (+ mmproj for vision) -> src-tauri/models/gemma/
#   gemma-4-12b-it-Q4_0.gguf + mmproj-F16.gguf
#   from an ungated mirror, e.g. huggingface.co/unsloth/gemma-4-12b-it-GGUF
```

> The official Google QAT GGUFs (`google/gemma-4-*-qat-q4_0-gguf`) are license-gated on Hugging Face; ungated community mirrors provide equivalent Q4_0 files. Eventually the in-app **setup wizard** (UC-05) will detect your GPU and download the right models/compute libraries automatically.

### 5. Build & test

```powershell
npm install
cargo test --lib                 # headless AI-engine tests (from src-tauri/)
cargo build --release --lib      # release build
npm run tauri dev                # run the app (once the shell exists)
```

Model-dependent tests skip automatically when the weights aren't present, so `cargo test` is green without any downloads. With the Gemma weights in place, the GPU generation test runs real Vulkan inference.

---

## Contributing

Contributions are welcome. Before opening a PR:

- Read [`specs.md`](./specs.md) (esp. §9 open decisions, §10 roadmap) and [`CLAUDE.md`](./CLAUDE.md).
- Keep the **Privacy Guard** gate intact: it must run before any embedding/inference/cloud call.
- Keep the **dynamic GPU-library** boundary intact; don't hardcode a single GPU backend.
- Run `cargo test --lib` (and `cargo fmt` / `cargo clippy`) before submitting.

By contributing, you agree that your contributions are licensed under the project's Apache-2.0 license.
