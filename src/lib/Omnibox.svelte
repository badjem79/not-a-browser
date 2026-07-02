<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  type Status = "idle" | "loading-model" | "generating" | "done";
  type View = "launcher" | "ask" | "browse";

  let query = $state("");
  let response = $state("");
  let status = $state<Status>("idle");
  let view = $state<View>("launcher");
  let panelOpen = $state(false);
  let inputEl: HTMLInputElement;

  // Page-node tree (step 2: flat roots). Each node = a place with its own webview.
  type PageNode = { id: string; title: string; url: string };
  let pages = $state<PageNode[]>([]);
  let activeId = $state<string | null>(null);
  let seq = 0;

  // Node queued to open once the bar finishes sliding up from the launcher.
  let pendingShow: PageNode | null = null;

  const mode = $derived<"navigate" | "ask">(looksLikeUrl(query) ? "navigate" : "ask");
  const busy = $derived(status === "loading-model" || status === "generating");
  const pinned = $derived(view !== "launcher");

  function looksLikeUrl(q: string): boolean {
    const s = q.trim();
    if (!s || /\s/.test(s)) return false;
    if (/^https?:\/\//i.test(s)) return true;
    if (/^localhost(:\d+)?(\/.*)?$/i.test(s)) return true;
    return /^[a-z0-9-]+(\.[a-z0-9-]+)+(:\d+)?(\/.*)?$/i.test(s);
  }

  function normalizeUrl(q: string): string {
    return /^https?:\/\//i.test(q) ? q : `https://${q}`;
  }

  function titleFor(url: string): string {
    try {
      return new URL(url).hostname.replace(/^www\./, "");
    } catch {
      return url;
    }
  }

  onMount(() => {
    inputEl?.focus();
    const subs: Promise<UnlistenFn>[] = [
      listen<string>("ai-token", (e) => {
        response += e.payload;
      }),
      listen<string>("ai-status", (e) => {
        status = e.payload as Status;
      }),
      listen("ai-done", () => {
        status = "done";
      }),
      listen<string>("ai-error", (e) => {
        response += `\n\n⚠ ${e.payload}`;
        status = "done";
      }),
    ];
    return () => subs.forEach((p) => p.then((un) => un()));
  });

  async function submit() {
    const q = query.trim();
    if (!q || busy) return;

    if (mode === "navigate") {
      const url = normalizeUrl(q);
      const node: PageNode = { id: String(++seq), title: titleFor(url), url };
      pages = [...pages, node];
      activeId = node.id;
      query = url;
      if (view === "launcher") {
        pendingShow = node;
        view = "browse";
      } else {
        view = "browse";
        showPage(node);
      }
      return;
    }

    if (view === "browse") await invoke("home").catch(() => {});
    view = "ask";
    response = "";
    status = "generating";
    try {
      await invoke("ask", { prompt: q });
    } catch (e) {
      response = `Errore: ${e}`;
      status = "done";
    }
  }

  async function showPage(node: PageNode) {
    try {
      await invoke("show_page", { id: node.id, url: node.url });
    } catch (e) {
      view = "ask";
      response = `Errore di navigazione: ${e}`;
      status = "done";
    }
  }

  function switchNode(node: PageNode) {
    activeId = node.id;
    query = node.url;
    view = "browse";
    showPage(node);
  }

  function onBarTransitionEnd(e: TransitionEvent) {
    if (e.propertyName === "top" && pendingShow) {
      const node = pendingShow;
      pendingShow = null;
      showPage(node);
    }
  }

  async function goHome() {
    await invoke("home").catch(() => {});
    view = "launcher";
    query = "";
    response = "";
    status = "idle";
    activeId = null;
    pendingShow = null;
    queueMicrotask(() => inputEl?.focus());
  }

  async function togglePanel() {
    panelOpen = !panelOpen;
    await invoke("set_panel", { open: panelOpen }).catch(() => {});
  }

  async function pageNav(action: "back" | "forward" | "reload") {
    await invoke("page_nav", { action }).catch(() => {});
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") {
      if (view === "launcher") query = "";
      else goHome();
    }
  }
</script>

<div class="chrome" data-view={view} data-mode={mode} class:panel-open={panelOpen}>
  <!-- Persistent burger: always visible, toggles the side dashboard. -->
  <button class="burger" class:active={panelOpen} onclick={togglePanel} aria-label="Menu" title="Sessioni e controlli">
    <span class="bang">!</span>
  </button>

  <!-- Left dashboard panel -->
  <aside class="panel" aria-hidden={!panelOpen}>
    <div class="panel-head">
      <button class="pbtn" onclick={goHome} title="Nuova sessione" aria-label="Home">⌂</button>
      <button class="pbtn" onclick={() => pageNav("back")} disabled={view !== "browse"} title="Indietro" aria-label="Indietro">◀</button>
      <button class="pbtn" onclick={() => pageNav("forward")} disabled={view !== "browse"} title="Avanti" aria-label="Avanti">▶</button>
      <button class="pbtn" onclick={() => pageNav("reload")} disabled={view !== "browse"} title="Ricarica" aria-label="Ricarica">⟳</button>
      <span class="spacer"></span>
      <button class="pbtn" onclick={togglePanel} title="Chiudi" aria-label="Chiudi">✕</button>
    </div>
    <div class="panel-body">
      {#if pages.length === 0}
        <p class="empty">Nessuna sessione.<br />Naviga o chiedi per iniziare.</p>
      {:else}
        <ul class="tree">
          {#each pages as node (node.id)}
            <li>
              <button
                class="node"
                class:active={node.id === activeId && view === "browse"}
                onclick={() => switchNode(node)}
                title={node.url}
              >
                <span class="ic">📄</span>
                <span class="label">
                  <span class="t">{node.title}</span>
                  <span class="u">{node.url}</span>
                </span>
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </aside>

  <div class="bar" class:pinned ontransitionend={onBarTransitionEnd}>
    <div class="field" class:busy>
      <span class="chip" title={mode === "navigate" ? "Vai all'indirizzo" : "Chiedi all'AI"}>
        <span class="dot"></span>
        {mode === "navigate" ? "vai" : "chiedi"}
      </span>
      <input
        bind:this={inputEl}
        bind:value={query}
        onkeydown={onKey}
        placeholder="Cerca, naviga o chiedi…"
        aria-label="Indirizzo o domanda"
        spellcheck="false"
        autocomplete="off"
      />
      <button class="go" onclick={submit} disabled={busy || !query.trim()} aria-label="Invio">
        ↵
      </button>
    </div>
  </div>

  {#if view === "launcher"}
    <p class="sub">
      Invio per <b>{mode === "navigate" ? "andare" : "chiedere"}</b> · il modello gira
      in locale sulla tua GPU
    </p>
  {/if}

  {#if view === "ask"}
    <div class="results">
      <div class="card ask">
        <div class="card-head">
          {#if status === "loading-model"}
            carico Gemma… <span class="muted">(qualche secondo al primo avvio)</span>
          {:else if status === "generating" && !response}
            <span class="dots"><i></i><i></i><i></i></span>
          {:else}
            risposta
          {/if}
        </div>
        {#if response}
          <div class="answer">{response}{#if status === "generating"}<span class="caret"></span>{/if}</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .chrome {
    position: fixed;
    inset: 0;
    --accent: var(--ask);
    --panel-offset: 0px;
  }
  .chrome[data-mode="navigate"] {
    --accent: var(--nav);
  }
  .chrome.panel-open {
    --panel-offset: 300px;
  }

  /* --- Persistent burger --- */
  .burger {
    position: fixed;
    left: 12px;
    top: 12px;
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border: none;
    border-radius: 11px;
    background: transparent;
    font-size: 20px;
    font-weight: 800;
    z-index: 6;
    -webkit-app-region: no-drag;
    transition: background 0.15s ease;
  }
  .burger:hover {
    background: var(--bg-elev);
  }
  .burger.active {
    background: color-mix(in oklab, var(--ask) 16%, transparent);
  }
  .burger .bang {
    color: var(--ask);
  }

  /* --- Side dashboard --- */
  .panel {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    width: 300px;
    display: flex;
    flex-direction: column;
    background: var(--bg-elev);
    border-right: 1px solid var(--line);
    transform: translateX(-100%);
    transition: transform 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    z-index: 5;
  }
  .chrome.panel-open .panel {
    transform: none;
  }
  .panel-head {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 64px;
    padding: 0 10px 0 60px; /* leave room for the burger */
    border-bottom: 1px solid var(--line);
  }
  .panel-head .spacer {
    flex: 1;
  }
  .pbtn {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: var(--text);
    font-size: 15px;
    transition: background 0.12s ease;
  }
  .pbtn:hover:not(:disabled) {
    background: color-mix(in oklab, var(--text) 9%, transparent);
  }
  .pbtn:disabled {
    opacity: 0.28;
    cursor: default;
  }
  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
  }
  .empty {
    font-size: 12.5px;
    line-height: 1.6;
    color: var(--muted);
    text-align: center;
    padding: 24px 8px;
  }
  .tree {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .node {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 8px 10px;
    border: none;
    border-radius: 10px;
    background: transparent;
    text-align: left;
    transition: background 0.12s ease;
  }
  .node:hover {
    background: color-mix(in oklab, var(--text) 7%, transparent);
  }
  .node.active {
    background: color-mix(in oklab, var(--nav) 18%, transparent);
  }
  .node .ic {
    font-size: 14px;
    flex: none;
  }
  .node .label {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .node .t {
    font-size: 13px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .node .u {
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .node.active .t {
    color: var(--nav);
    font-weight: 600;
  }

  /* --- Omnibar (slides center → top) --- */
  .bar {
    position: fixed;
    left: var(--panel-offset);
    right: 0;
    top: 34vh;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 0 24px;
    height: 64px;
    transition:
      top 0.34s cubic-bezier(0.22, 1, 0.36, 1),
      left 0.28s cubic-bezier(0.22, 1, 0.36, 1),
      padding 0.34s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .bar.pinned {
    top: 0;
    justify-content: flex-start;
    padding: 0 14px 0 60px; /* clear the burger on the left */
    -webkit-app-region: drag;
    border-bottom: 1px solid var(--line);
    background: color-mix(in oklab, var(--bg-elev) 60%, transparent);
    backdrop-filter: blur(12px);
  }

  .field {
    -webkit-app-region: no-drag;
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 0 1 640px;
    max-width: 640px;
    padding: 8px 8px 8px 14px;
    background: var(--bg-elev);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    transition:
      border-color 0.2s ease,
      box-shadow 0.2s ease,
      max-width 0.34s cubic-bezier(0.22, 1, 0.36, 1),
      flex-basis 0.34s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .bar.pinned .field {
    flex: 1 1 auto;
    max-width: none;
    border-radius: 12px;
  }
  .field:focus-within {
    border-color: color-mix(in oklab, var(--accent) 70%, var(--line));
    box-shadow: 0 0 0 4px color-mix(in oklab, var(--accent) 18%, transparent);
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: var(--accent);
    background: color-mix(in oklab, var(--accent) 14%, transparent);
    border-radius: 999px;
    user-select: none;
    transition: color 0.2s ease, background 0.2s ease;
  }
  .chip .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
  }

  input {
    flex: 1;
    min-width: 0;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font-size: 16px;
    padding: 8px 2px;
    caret-color: var(--accent);
  }
  input::placeholder {
    color: var(--muted);
  }

  .go {
    border: none;
    background: var(--accent);
    color: #0b0b0f;
    font-weight: 800;
    font-size: 16px;
    width: 40px;
    height: 40px;
    flex: none;
    border-radius: 12px;
    transition: opacity 0.15s ease, transform 0.05s ease;
  }
  .go:disabled {
    opacity: 0.35;
    cursor: default;
  }
  .go:not(:disabled):active {
    transform: scale(0.94);
  }

  .sub {
    position: fixed;
    left: var(--panel-offset);
    right: 0;
    top: calc(34vh + 64px);
    text-align: center;
    font-size: 12.5px;
    color: var(--muted);
    animation: fade 0.3s ease;
    transition: left 0.28s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .sub b {
    color: color-mix(in oklab, var(--accent) 80%, var(--text));
    font-weight: 650;
  }

  .results {
    position: fixed;
    top: 64px;
    left: var(--panel-offset);
    right: 0;
    bottom: 0;
    overflow-y: auto;
    padding: 18px;
    display: flex;
    justify-content: center;
    transition: left 0.28s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .card {
    width: min(720px, 100%);
    height: fit-content;
    background: var(--bg-elev);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 14px 16px;
    animation: rise 0.18s ease;
  }
  .card-head {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .answer {
    font-size: 15px;
    line-height: 1.55;
    white-space: pre-wrap;
  }
  .muted {
    color: var(--muted);
  }

  .caret {
    display: inline-block;
    width: 8px;
    height: 1.05em;
    margin-left: 2px;
    vertical-align: text-bottom;
    background: var(--accent);
    animation: blink 1s steps(2, start) infinite;
  }

  .dots {
    display: inline-flex;
    gap: 5px;
  }
  .dots i {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    opacity: 0.4;
    animation: pulse 1.1s ease-in-out infinite;
  }
  .dots i:nth-child(2) {
    animation-delay: 0.18s;
  }
  .dots i:nth-child(3) {
    animation-delay: 0.36s;
  }

  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
  @keyframes pulse {
    0%,
    100% {
      opacity: 0.3;
      transform: translateY(0);
    }
    50% {
      opacity: 1;
      transform: translateY(-3px);
    }
  }
</style>
