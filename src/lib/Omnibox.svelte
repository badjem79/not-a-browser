<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  type Status = "idle" | "loading-model" | "generating" | "done";

  let query = $state("");
  let response = $state("");
  let status = $state<Status>("idle");
  let navTarget = $state<string | null>(null);
  let inputEl: HTMLInputElement;

  // Live mode detection: a URL-ish string navigates; anything else asks the AI.
  const mode = $derived<"navigate" | "ask">(looksLikeUrl(query) ? "navigate" : "ask");
  const busy = $derived(status === "loading-model" || status === "generating");

  function looksLikeUrl(q: string): boolean {
    const s = q.trim();
    if (!s || /\s/.test(s)) return false;
    if (/^https?:\/\//i.test(s)) return true;
    if (/^localhost(:\d+)?(\/.*)?$/i.test(s)) return true;
    // domain.tld (optionally with a path)
    return /^[a-z0-9-]+(\.[a-z0-9-]+)+(:\d+)?(\/.*)?$/i.test(s);
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
      // Spike: no WebView yet — just show what we'd navigate to.
      navTarget = /^https?:\/\//i.test(q) ? q : `https://${q}`;
      response = "";
      status = "idle";
      return;
    }

    // Ask the local model; tokens stream back via events.
    navTarget = null;
    response = "";
    status = "generating";
    try {
      await invoke("ask", { prompt: q });
    } catch (e) {
      response = `Errore: ${e}`;
      status = "done";
    }
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") {
      query = "";
    }
  }
</script>

<section class="omni" data-mode={mode}>
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

  <p class="sub">
    Invio per <b>{mode === "navigate" ? "andare" : "chiedere"}</b> · il modello gira
    in locale sulla tua GPU
  </p>

  {#if navTarget}
    <div class="card nav">
      <div class="card-head">navigazione</div>
      <div class="nav-url">{navTarget}</div>
      <div class="card-note">la WebView arriva nel prossimo step dello shell</div>
    </div>
  {/if}

  {#if status !== "idle" && mode === "ask" && !navTarget}
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
  {/if}
</section>

<style>
  .omni {
    width: min(720px, 92vw);
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
    /* accent follows the active mode */
    --accent: var(--ask);
  }
  .omni[data-mode="navigate"] {
    --accent: var(--nav);
  }

  .field {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 8px 8px 14px;
    background: var(--bg-elev);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    transition:
      border-color 0.2s ease,
      box-shadow 0.2s ease;
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
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font-size: 17px;
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
    text-align: center;
    font-size: 12.5px;
    color: var(--muted);
  }
  .sub b {
    color: color-mix(in oklab, var(--accent) 80%, var(--text));
    font-weight: 650;
  }

  .card {
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
  .card-note {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
  }
  .nav-url {
    font-size: 15px;
    color: var(--nav);
    word-break: break-all;
  }

  .answer {
    font-size: 15px;
    line-height: 1.55;
    white-space: pre-wrap;
    max-height: 38vh;
    overflow-y: auto;
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
