# Specifiche Tecniche e Funzionali: !aBrowser

> **!aBrowser** ("not a browser", dove `!` è l'operatore NOT): non è un motore browser, ma un guscio AI sopra la WebView del sistema operativo. Slug tecnico: `not-a-browser`.

**Versione:** 1.1
**Data:** Giugno 2026
**Stato:** In revisione — vedi §9 (Decisioni aperte) e §10 (Roadmap)
**Target Hardware Ottimale:** CPU x86/ARM, GPU AMD RDNA 3 (Radeon RX 7800 XT 16GB VRAM) / NVIDIA RTX / Apple Silicon

---

## 1. Visione del Prodotto e Filosofia

!aBrowser è un browser web incentrato sulla privacy e sull'intelligenza artificiale locale e ibrida. A differenza dei prodotti concorrenti che relegano l'AI a una chat d'accompagnamento isolata, !aBrowser integra l'AI direttamente nel ciclo di vita della navigazione tramite tre canali sensoriali: **Vista** (Vision-LLM), **Ascolto** (Speech-to-Text) e **Parola** (Text-to-Speech). Il sistema è strutturato per essere una piuma sul disco e sulla memoria del PC, garantendo al contempo prestazioni d'eccellenza e privacy totale.

**Principio guida sulla privacy:** nessun dato di navigazione lascia la macchina senza consenso. Il consenso all'uso dell'AI (locale e, a maggior ragione, cloud) è concesso **per singola tab di navigazione**. Esiste un interruttore globale che disattiva il filtro di privacy, da usare a rischio e pericolo dell'utente.

---

## 2. Specifiche Funzionali (Casi d'Uso)

### UC-01: Navigazione Web Convenzionale
* **Descrizione:** L'utente naviga sul web visualizzando correttamente applicativi complessi (es. YouTube, Facebook, LinkedIn, siti di news) sfruttando l'accelerazione hardware della GPU per il rendering delle pagine.
* **Interfaccia:** Barra di navigazione minimale superiore, gestione a schede (tab) isolate e HUD (Heads-Up Display) laterale a scomparsa per le interazioni AI.

### UC-02: Interazione Vocale "Hands-Free"
* **Descrizione:** L'utente attiva il microfono integrato per impartire comandi vocali sulla pagina (es. *"Riassumi i punti chiave"*). L'AI risponde sintetizzando il testo a voce.
* **Controlli:** Player multimediale minimale (Play, Pausa, Velocità di riproduzione) integrato nell'HUD.
* **Attivazione microfono:** push-to-talk come default privacy-safe; wake-word opzionale e disattivabile (vedi §9).

### UC-03: Interazione Visiva e Multimodale
* **Descrizione:** L'utente seleziona un'area dello schermo (`Alt + Trascina`) o fa click destro su un elemento `<img>`. L'AI analizza i pixel ed esegue compiti di trascrizione, spiegazione o traduzione dell'elemento visivo.

### UC-04: Cronologia Semantica Onnicomprensiva (RAG Multimodale)
* **Descrizione:** Il browser indicizza in background le sessioni di navigazione (pagine utili), le descrizioni delle immagini analizzate e lo storico delle chat con l'AI. L'utente può eseguire ricerche astratte nella barra dei comandi (es. *"Trova quel grafico sui consumi energetici dei chip visto ieri"*).
* **Indicizzazione:** l'embedding di una pagina viene generato a fine caricamento, con debounce, e solo se la tab ha il consenso AI attivo e l'URL supera il Privacy Guard (§6).

### UC-05: Setup Hardware Wizard Automatizzato
* **Descrizione:** Al primo avvio, l'applicativo analizza l'architettura del PC e scarica autonomamente solo le librerie di calcolo e i modelli quantizzati adatti alla GPU rilevata. I download sono verificati per integrità (checksum/firma) prima del caricamento.

---

## 3. Specifiche Tecniche e Architettura

### 3.1 Stack Tecnologico
* **Runtime Core & Orchestrazione:** Rust (Stable Edition) + Tauri v2, runtime async `tokio`.
* **Motore di Rendering (Frontend):** `wry` → WebView2 (Chromium) su Windows, WebKitGTK su Linux, WKWebView su macOS.
* **Kernel Computazionale AI:** `llama.cpp` + `whisper.cpp`.
  * **Backend GPU di default:** **Vulkan** (massima affidabilità su Radeon consumer/Windows).
  * **Backend alternativi caricati a runtime via `libloading`:** ROCm/HIP (AMD), CUDA (NVIDIA), Metal (Apple Silicon).
* **Sintesi Vocale (TTS):** *da definire* (vedi §9). Candidati: Piper (ONNX, leggero), Kokoro/OuteTTS (qualità superiore), TTS di sistema.
* **Database Vettoriale Embedded:** `LanceDB` (formato colonnare basato su Apache Arrow), schemi definiti come Arrow schema in Rust (vedi §5).
* **Modello di Embedding Locale (CPU):** `all-MiniLM-L6-v2` (384 dim) via ONNX Runtime / Burn (~100 MB RAM).
* **Modello Vision/Generazione:** **Gemma 4 12B-it QAT Q4_0** (GGUF ufficiale Google, Apache 2.0, rilasciato giugno 2026). Architettura **encoder-free unificata**: input testo/immagine/audio(≤30s)/video(≤60s @1fps), output testo. File ~7 GB, contesto fino a 256K (consigliato 32K per reattività). **Vision in `llama.cpp`:** richiede `llama-mtmd-cli` + file **mmproj** GGUF separato. L'**audio nativo (≤30s)** può coprire comandi vocali brevi anche senza Whisper.

### 3.2 Architettura Multi-Processo e Flusso Dati

Il sistema isola il processo grafico dal core di computazione per garantire stabilità a 144Hz. Le richieste verso l'engine AI passano per un canale MPSC; ogni richiesta porta con sé un canale di ritorno (`oneshot`) o produce eventi Tauri per lo streaming incrementale dei token verso la UI.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              CORE PROCESS (Rust)                          │
│                                                                          │
│  ┌─────────────────────────┐   MPSC + oneshot   ┌──────────────────────┐ │
│  │  Tauri Command Router   ├───────────────────►│  AI Execution Engine │ │
│  └────────────▲────────────┘   eventi (stream)  ├──────────────────────┤ │
│               │  ◄───────────────────────────── │ 1. llama.cpp (Vision)│ │
│               │ IPC (Tauri)                     │ 2. whisper.cpp (STT) │ │
│               │                                 │ 3. TTS engine        │ │
│               ▼                                 │ 4. LanceDB (vettori) │ │
│  ┌─────────────────────────┐                    └──────────┬───────────┘ │
│  │ WEBVIEW PROCESS (wry)   │                               │ pesi mlock  │
│  ├─────────────────────────┤                               ▼             │
│  │ UI (Barra + HUD)        │                      ┌──────────────────┐   │
│  │ App Navigata (WebView)  │                      │ VRAM / GPU       │   │
│  └─────────────────────────┘                      └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Concorrenza dell'inferenza:** `llama.cpp` non è banalmente concorrente. L'AI Execution Engine serializza le richieste con una coda a slot singolo (o pochi slot dedicati), così la UI resta reattiva mentre l'inferenza procede.

**Isolamento tab:** ogni tab è un'istanza WebView isolata con profilo/sessione separati. Da progettare esplicitamente (cookie, storage, processi) perché WebView2 è nato per l'embedding, non come motore browser multi-tab.

### 3.3 Router di Inferenza Ibrido (Trait Rust)

L'architettura astrae l'accesso ai modelli per consentire il Fallback Cloud (es. Google Gemini Flash o DeepSeek a bassissimo costo) su macchine con hardware limitato. Il trait espone metodi **streaming** (per la reattività della UI) ed embedding.

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;

#[derive(Debug)]
pub enum LlmError {
    InferenceFailed(String),
    NetworkError(String),
    HardwareUnavailable,
}

#[async_trait]
pub trait LlmEngine: Send + Sync {
    /// Generazione testuale in streaming (token incrementali).
    async fn generate_text(
        &self,
        prompt: &str,
        context: &str,
    ) -> Result<BoxStream<'static, Result<String, LlmError>>, LlmError>;

    /// Analisi multimodale di un'immagine.
    async fn analyze_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<BoxStream<'static, Result<String, LlmError>>, LlmError>;

    /// Identità del backend, per logging/telemetria locale e routing.
    fn backend_id(&self) -> &str;
}

/// L'embedding è separato dal generation engine: gira su CPU ed è
/// indipendente dal backend GPU/cloud.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError>; // len == 384
    fn model_version(&self) -> &str;
}
```

Il **router** sceglie l'implementazione (locale GPU vs cloud) in base a: hardware disponibile, consenso della tab corrente, e classificazione dell'URL dal Privacy Guard. Il fallback cloud non viene mai invocato per tab senza consenso o per URL sensibili.

---

## 4. Gestione dei Dati e Memoria (Stime di Peso)

### 4.1 Modello e Cache VRAM (Esempio AMD Radeon 7800 XT, 16 GB)
* **Gemma 4 12B-it QAT Q4_0:** ~7 GB di pesi residenti in VRAM (`mlock`).
* **mmproj (proiettore multimodale):** ~0.5–1 GB aggiuntivi per l'input immagine.
* **KV Cache:** dipende dal contesto. A 32K token è contenuta; i 256K teorici richiederebbero molta più VRAM, quindi 32K è il default pratico. Valutare la quantizzazione della KV cache per contesti lunghi.
* **TTS Context:** ~0.5–1 GB VRAM / RAM (engine da definire, §9).
* **Margine:** con ~8 GB di modello+mmproj e KV a 32K, restano diversi GB liberi sui 16 GB — budget comodo, non più al limite.

### 4.2 Stima Spazio su Disco (RAG Semantico su base Annua)
Stima per navigazione intensa di 50 pagine testuali indicizzate al giorno (~2.000 parole pulite a pagina, ~14 KB di testo in chiaro):
* Testo in chiaro memorizzato: ~700 KB / giorno
* Vettori di embedding (384 dim f32): ~375 KB / giorno
* Metadati e log di chat: ~125 KB / giorno
* **Totale giornaliero:** ~1.2 MB / giorno → **~438 MB / anno**

I dati a riposo (testo, descrizioni, chat, miniature) sono candidati alla **cifratura su disco** (vedi §6).

---

## 5. Schema Strutturale delle Tabelle del Database (LanceDB)

Schemi logici (rappresentazione illustrativa; l'implementazione reale è un Arrow schema in Rust). Ogni tabella include `embedding_model_version` per gestire la reindicizzazione quando cambia il modello di embedding.

### table.web_history
* `id: VARCHAR (PK)`
* `vector: VECTOR(384)` — coordinate semantiche del testo della pagina
* `text_content: TEXT` — testo pulito estratto dal DOM via JS injection
* `url: VARCHAR` — URL sorgente
* `embedding_model_version: VARCHAR`
* `timestamp: TIMESTAMP`

### table.image_history
* `id: VARCHAR (PK)`
* `vector: VECTOR(384)` — embedding della descrizione prodotta dall'AI
* `description: TEXT` — descrizione generata da modello Vision / Cloud API
* `source_url: VARCHAR` — URL della pagina ospite
* `thumbnail_path: VARCHAR` — path locale alla miniatura in cache
* `embedding_model_version: VARCHAR`
* `timestamp: TIMESTAMP`

### table.chat_history
* `id: VARCHAR (PK)`
* `vector: VECTOR(384)` — embedding dell'intero blocco (Prompt + Risposta)
* `conversation_chunk: TEXT` — log testuale dello scambio
* `context_url: VARCHAR` — eventuale URL attivo durante la chat
* `embedding_model_version: VARCHAR`
* `timestamp: TIMESTAMP`

**Indici e retention:** definire la configurazione dell'indice vettoriale (es. IVF_PQ) e una policy di cancellazione/TTL coerente con il diritto utente alla rimozione dello storico.

---

## 6. Sicurezza e Regole di Business (Privacy Guard)

* **Consenso per tab:** ogni tab decide se l'AI (locale e cloud) può accedere ai suoi contenuti. Default privacy-safe. Interruttore globale di disattivazione del Guard disponibile, esplicitamente "a rischio e pericolo dell'utente".
* **Analisi URL Locale:** prima di qualsiasi embedding o inferenza (locale o remota), il modulo Rust classifica l'URL ed esclude domini finanziari (bank, fineco, ecc.), pagine di checkout, URL locali (`localhost`, `127.0.0.1`) e stringhe sensibili. La blocklist a regex è il minimo; preferire categorie + lista mantenuta + override utente. Il gate gira **prima** di embedding e inferenza remota, mai dopo.
* **Fallback cloud e privacy:** l'invio di contenuti a servizi cloud avviene solo per tab con consenso esplicito e mai per URL bloccati dal Guard. Da documentare chiaramente all'utente quali dati escono.
* **Cifratura a riposo:** lo storico RAG e le cache contengono dati di navigazione sensibili in chiaro; vanno cifrati su disco. *(Da definire il meccanismo — vedi §9.)*
* **Dynamic Loading delle Librerie (`libloading`):** l'app non linka staticamente i compilati Vulkan/CUDA/ROCm. Al boot carica la libreria dinamica (.dll/.so) adatta alla GPU attiva, scaricata dal setup wizard e verificata per integrità, azzerando i conflitti di driver.

---

## 7. Requisiti Non Funzionali (bozza)

* **Reattività UI:** target 144Hz; l'inferenza non deve mai bloccare il thread UI (streaming + coda dedicata).
* **Footprint:** RAM base contenuta; modelli caricati on-demand secondo l'hardware.
* **Latenza percepita:** primo token rapido grazie allo streaming; budget di latenza da definire per STT/TTS/vision.
* **Offline-first:** tutte le funzioni core devono funzionare senza rete (cloud è solo fallback opzionale).

---

## 8. Rischi Tecnici Principali

1. **WebView2 come browser multi-tab** — isolamento sessioni/processi non nativo: complessità sottostimata.
2. **Budget VRAM** — ~14 GB su 16 GB è stretto; dipende dal modello reale e dalla KV cache.
3. **Backend GPU AMD su Windows** — mitigato scegliendo Vulkan come default.
4. **Integrazione mmproj/`llama-mtmd-cli`** per la vision in `llama.cpp` (pipeline diversa dalla sola generazione testo).
5. **JS injection cross-origin** per estrazione DOM — vincoli CSP/sicurezza da gestire.

---

## 9. Decisioni Aperte

* **Motore TTS:** *non ancora deciso.* (Piper / Kokoro-OuteTTS / TTS di sistema.)
* **Meccanismo di cifratura a riposo:** scelta tra cifratura applicativa, file-based, o affidamento a cifratura disco di sistema.
* **Wake-word per UC-02:** se/come implementarla mantenendo il default push-to-talk.

---

## 10. Roadmap (approccio: AI-core headless first)

L'engine AI viene costruito e testato **senza UI** prima di integrare il browser.

* **Fase 0 — Scaffolding:** progetto Tauri v2 (`src-tauri/` core Rust + frontend), CI, struttura cartelle.
* **Fase 1 — AI Execution Engine headless:**
  * Trait `LlmEngine` + `Embedder` e router di inferenza.
  * Backend locale via `llama.cpp` (Vulkan) per `generate_text` in streaming.
  * `Embedder` CPU con `all-MiniLM-L6-v2`.
  * Integrazione LanceDB (3 tabelle) + pipeline RAG (chunking, embed, top-k).
  * Privacy Guard (classificazione URL) + consenso per tab a livello di API.
  * Test end-to-end via CLI/test harness, nessuna UI.
* **Fase 2 — Vision & Audio:** `analyze_image`, `whisper.cpp` STT, scelta e integrazione TTS.
* **Fase 3 — Browser shell:** UI Tauri (barra + HUD), tab isolati, JS injection per estrazione DOM, wiring degli use-case UC-01..04.
* **Fase 4 — Setup Wizard (UC-05):** detect hardware, download verificato di modelli e librerie GPU, `libloading`.
* **Fase 5 — Hardening:** cifratura a riposo, fallback cloud con consenso, retention/cancellazione, packaging multi-OS.
