<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { renderMarkdown } from "./markdown";

  type Status = "idle" | "loading-model" | "generating" | "done";
  type View = "launcher" | "ask" | "browse";

  // Context tree (§11). A node is a "place/session" with its own history.
  // Step 3: types page|chat. Branching happens only on context jumps (§11.1):
  //  · question from a page  ⇒ child chat (parent = active page)
  //  · question from launcher ⇒ root chat
  //  · follow-up in a chat    ⇒ stays in the node (no branch)
  //  · new URL in the omnibox ⇒ new root page
  type NodeType = "page" | "chat";
  type Msg = { role: "user" | "model"; text: string };
  type Node = {
    id: string;
    type: NodeType;
    title: string;
    parentId: string | null;
    url?: string; // page
    messages?: Msg[]; // chat
  };

  let query = $state("");
  let status = $state<Status>("idle");
  let view = $state<View>("launcher");
  let panelOpen = $state(false);
  let inputEl: HTMLInputElement;

  let nodes = $state<Node[]>([]);
  let activeId = $state<string | null>(null);
  let seq = 0;

  // Node queued to open once the bar finishes sliding up from the launcher.
  let pendingShow: Node | null = null;
  // Chat whose model reply is currently being streamed into (token target).
  let streamingChatId: string | null = null;

  const mode = $derived<"navigate" | "ask">(looksLikeUrl(query) ? "navigate" : "ask");
  const busy = $derived(status === "loading-model" || status === "generating");
  const pinned = $derived(view !== "launcher");
  const roots = $derived(nodes.filter((n) => n.parentId === null));
  const activeChat = $derived(
    nodes.find((n) => n.id === activeId && n.type === "chat") ?? null,
  );

  function childrenOf(id: string): Node[] {
    return nodes.filter((n) => n.parentId === id);
  }

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

  // §11.4/§11.2: walk the current ancestors of `node`, collecting the page
  // lineage as lightweight grounding. (Full RAG over page DOM text is the later
  // UC-04 indexing step; here we ground on page identity — title + URL.)
  function buildContext(node: Node): string {
    const pages: string[] = [];
    let cur: Node | null =
      node.parentId != null ? nodes.find((n) => n.id === node.parentId) ?? null : null;
    while (cur) {
      if (cur.type === "page" && cur.url) pages.unshift(`- ${cur.title} (${cur.url})`);
      cur = cur.parentId != null ? nodes.find((n) => n.id === cur!.parentId) ?? null : null;
    }
    return pages.length
      ? `L'utente sta consultando queste pagine (contesto):\n${pages.join("\n")}`
      : "";
  }

  onMount(() => {
    inputEl?.focus();
    const subs: Promise<UnlistenFn>[] = [
      listen<string>("ai-token", (e) => {
        const chat = nodes.find((n) => n.id === streamingChatId);
        const last = chat?.messages?.[chat.messages.length - 1];
        if (last?.role === "model") last.text += e.payload;
      }),
      listen<string>("ai-status", (e) => {
        status = e.payload as Status;
      }),
      listen("ai-done", () => {
        status = "done";
        streamingChatId = null;
      }),
      listen<string>("ai-error", (e) => {
        const chat = nodes.find((n) => n.id === streamingChatId);
        const last = chat?.messages?.[chat.messages.length - 1];
        if (last?.role === "model") last.text += `\n\n⚠ ${e.payload}`;
        status = "done";
        streamingChatId = null;
      }),
      // A page node follows navigation inside its webview (§11.1): update its
      // label to the current URL, and the omnibox too while it's the active page.
      listen<{ id: string; url: string }>("page-loaded", (e) => {
        const n = nodes.find((x) => x.id === e.payload.id && x.type === "page");
        if (!n) return;
        n.url = e.payload.url;
        n.title = titleFor(e.payload.url);
        if (activeId === n.id && view === "browse") query = e.payload.url;
      }),
    ];
    return () => subs.forEach((p) => p.then((un) => un()));
  });

  async function submit() {
    const q = query.trim();
    if (!q || busy) return;

    // --- Navigate: a URL always opens a new root page (§11.1). ---
    if (mode === "navigate") {
      const url = normalizeUrl(q);
      const node: Node = {
        id: String(++seq),
        type: "page",
        title: titleFor(url),
        url,
        parentId: null,
      };
      nodes = [...nodes, node];
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

    // --- Ask: branch or follow-up per §11.1. ---
    const active = nodes.find((n) => n.id === activeId) ?? null;
    let chat: Node;
    if (active?.type === "chat") {
      chat = active; // follow-up: stays in this node
    } else {
      // question from a page ⇒ child chat; from launcher ⇒ root chat
      const parentId = active?.type === "page" ? active.id : null;
      chat = {
        id: String(++seq),
        type: "chat",
        title: q.length > 42 ? q.slice(0, 42) + "…" : q,
        parentId,
        messages: [],
      };
      nodes = [...nodes, chat];
    }

    if (view === "browse") await invoke("home").catch(() => {});
    activeId = chat.id;
    view = "ask";

    chat.messages!.push({ role: "user", text: q });
    // Full ordered history (prior turns + this question) gives the model memory
    // of the conversation; the empty model placeholder below is not sent.
    const history = chat.messages!.map((m) => ({ role: m.role, text: m.text }));
    chat.messages!.push({ role: "model", text: "" });
    streamingChatId = chat.id;
    query = "";
    status = "generating";

    try {
      await invoke("ask", { history, context: buildContext(chat) });
    } catch (e) {
      const last = chat.messages![chat.messages!.length - 1];
      last.text = `Errore: ${e}`;
      status = "done";
      streamingChatId = null;
    }
  }

  async function showPage(node: Node) {
    try {
      await invoke("show_page", { id: node.id, url: node.url });
    } catch (e) {
      view = "ask";
      status = "done";
    }
  }

  function openNode(node: Node) {
    activeId = node.id;
    if (node.type === "page") {
      query = node.url ?? "";
      view = "browse";
      showPage(node);
    } else {
      // reveal an existing chat: hide any page webview, show its conversation
      if (view === "browse") invoke("home").catch(() => {});
      query = "";
      view = "ask";
      status = "done";
    }
  }

  // Open `url` as a page node. From within a chat it becomes a child of that
  // chat (§11.1: chat → link ⇒ child page); otherwise a new root.
  function openUrl(url: string) {
    const parentId = activeChat ? activeChat.id : null;
    const node: Node = {
      id: String(++seq),
      type: "page",
      title: titleFor(url),
      url,
      parentId,
    };
    nodes = [...nodes, node];
    activeId = node.id;
    query = url;
    view = "browse";
    showPage(node);
  }

  // Links rendered in a model answer must not hijack the chrome webview: catch
  // the click, keep the chrome put, and route http(s) links through openUrl.
  function onAnswerClick(e: MouseEvent) {
    const a = (e.target as HTMLElement)?.closest?.("a");
    const href = a?.getAttribute("href");
    if (!href) return;
    e.preventDefault();
    if (/^https?:\/\//i.test(href)) openUrl(href);
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
      {#if nodes.length === 0}
        <p class="empty">Nessuna sessione.<br />Naviga o chiedi per iniziare.</p>
      {:else}
        <ul class="tree">
          {#each roots as node (node.id)}
            {@render treeNode(node, 0)}
          {/each}
        </ul>
      {/if}
    </div>
  </aside>

  {#snippet treeNode(node: Node, depth: number)}
    <li>
      <button
        class="node"
        class:active={node.id === activeId}
        onclick={() => openNode(node)}
        title={node.url ?? node.title}
        style="padding-left:{10 + depth * 16}px"
      >
        <span class="ic">{node.type === "page" ? "📄" : "💬"}</span>
        <span class="label">
          <span class="t">{node.title}</span>
          {#if node.url}<span class="u">{node.url}</span>{/if}
        </span>
      </button>
      {#each childrenOf(node.id) as child (child.id)}
        {@render treeNode(child, depth + 1)}
      {/each}
    </li>
  {/snippet}

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
        placeholder={activeChat ? "Continua la conversazione…" : "Cerca, naviga o chiedi…"}
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

  {#if view === "ask" && activeChat}
    <div class="results">
      <div class="chat">
        {#each activeChat.messages ?? [] as m, i (i)}
          {#if m.role === "user"}
            <div class="msg user"><div class="bubble">{m.text}</div></div>
          {:else}
            <div class="msg model">
              {#if !m.text && status !== "done"}
                {#if status === "loading-model"}
                  <span class="muted">carico Gemma… <span class="muted">(qualche secondo al primo avvio)</span></span>
                {:else}
                  <span class="dots"><i></i><i></i><i></i></span>
                {/if}
              {:else}
                <div class="answer" role="presentation" onclick={onAnswerClick}>
                  {@html renderMarkdown(m.text)}
                </div>
                {#if i === (activeChat.messages?.length ?? 0) - 1 && status === "generating"}
                  <span class="caret"></span>
                {/if}
              {/if}
            </div>
          {/if}
        {/each}
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
  .tree li {
    list-style: none;
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

  .chat {
    width: min(760px, 100%);
    height: fit-content;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    display: flex;
    animation: rise 0.18s ease;
  }
  .msg.user {
    justify-content: flex-end;
  }
  .msg.user .bubble {
    max-width: 78%;
    background: color-mix(in oklab, var(--nav) 20%, var(--bg-elev));
    border: 1px solid color-mix(in oklab, var(--nav) 30%, var(--line));
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px;
    font-size: 15px;
    line-height: 1.5;
    white-space: pre-wrap;
  }
  .msg.model {
    flex-direction: column;
  }

  .answer {
    font-size: 15px;
    line-height: 1.6;
  }
  /* Injected Markdown HTML isn't touched by Svelte's scoped styles, so target
     it with :global under .answer. Tight vertical rhythm; no margin on the
     first/last child so the card padding stays even. */
  .answer :global(> *:first-child) {
    margin-top: 0;
  }
  .answer :global(> *:last-child) {
    margin-bottom: 0;
  }
  .answer :global(p) {
    margin: 0 0 0.7em;
  }
  .answer :global(h1),
  .answer :global(h2),
  .answer :global(h3),
  .answer :global(h4) {
    margin: 1.1em 0 0.5em;
    line-height: 1.3;
    font-weight: 650;
  }
  .answer :global(h1) {
    font-size: 1.4em;
  }
  .answer :global(h2) {
    font-size: 1.25em;
  }
  .answer :global(h3) {
    font-size: 1.1em;
  }
  .answer :global(ul),
  .answer :global(ol) {
    margin: 0 0 0.7em;
    padding-left: 1.5em;
  }
  .answer :global(li) {
    margin: 0.2em 0;
  }
  .answer :global(li::marker) {
    color: var(--muted);
  }
  .answer :global(a) {
    color: var(--nav);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .answer :global(strong) {
    font-weight: 680;
  }
  .answer :global(em) {
    font-style: italic;
  }
  .answer :global(code) {
    font-family: ui-monospace, "Cascadia Code", "Consolas", monospace;
    font-size: 0.88em;
    background: color-mix(in oklab, var(--text) 10%, transparent);
    padding: 0.12em 0.4em;
    border-radius: 6px;
  }
  .answer :global(pre) {
    margin: 0 0 0.7em;
    padding: 12px 14px;
    background: color-mix(in oklab, var(--text) 8%, transparent);
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow-x: auto;
  }
  .answer :global(pre code) {
    background: none;
    padding: 0;
    font-size: 0.86em;
    line-height: 1.5;
  }
  .answer :global(blockquote) {
    margin: 0 0 0.7em;
    padding: 0.1em 0 0.1em 0.9em;
    border-left: 3px solid color-mix(in oklab, var(--nav) 50%, var(--line));
    color: var(--muted);
  }
  .answer :global(hr) {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1em 0;
  }
  .answer :global(table) {
    border-collapse: collapse;
    margin: 0 0 0.7em;
    font-size: 0.92em;
  }
  .answer :global(th),
  .answer :global(td) {
    border: 1px solid var(--line);
    padding: 5px 10px;
    text-align: left;
  }
  .answer :global(th) {
    background: color-mix(in oklab, var(--text) 6%, transparent);
    font-weight: 600;
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
