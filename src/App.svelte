<script lang="ts">
  import Omnibox from "./lib/Omnibox.svelte";
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      <span class="bang">!</span>aBrowser
    </div>
    <div class="hint">spike · omnibox fuso</div>
  </header>

  <main class="stage">
    <Omnibox />
  </main>
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    -webkit-app-region: drag; /* let the bar move the window */
    user-select: none;
  }

  .brand {
    font-weight: 650;
    letter-spacing: 0.2px;
    font-size: 15px;
  }
  .brand .bang {
    color: var(--ask);
    font-weight: 800;
  }

  .hint {
    font-size: 12px;
    color: var(--muted);
  }

  .stage {
    flex: 1;
    display: grid;
    place-items: center;
    padding: 24px;
    /* nudge the field slightly above the optical center */
    padding-bottom: 12vh;
  }
</style>
