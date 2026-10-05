/**
 * ALETHEX Universal Content Script (Manifest V3)
 * Performance-optimized, non-blocking, responsive memory reconciliation guard.
 *
 * Supported Platforms: ChatGPT, Claude, Gemini, Copilot, DeepSeek, Perplexity, Grok, Poe, Mistral, Local WebUIs.
 */

(function () {
  if (window.__ALETHEX_GUARD_INITIALIZED__) {
    try {
      if (chrome.runtime && chrome.runtime.id) {
        return;
      }
    } catch (e) {}
  }
  window.__ALETHEX_GUARD_INITIALIZED__ = true;

  // Platform detection
  const host = window.location.hostname.toLowerCase();
  let platform = "AI Chat";
  if (host.includes("chatgpt") || host.includes("openai")) platform = "ChatGPT";
  else if (host.includes("claude.ai")) platform = "Claude.ai";
  else if (host.includes("gemini.google")) platform = "Google Gemini";
  else if (host.includes("copilot.microsoft")) platform = "Copilot";
  else if (host.includes("deepseek")) platform = "DeepSeek";
  else if (host.includes("perplexity")) platform = "Perplexity";
  else if (host.includes("grok") || host.includes("x.com")) platform = "Grok";
  else if (host.includes("poe.com")) platform = "Poe";
  else if (host.includes("mistral")) platform = "Mistral";
  else if (host.includes("localhost") || host.includes("127.0.0.1")) platform = "Local WebUI";

  // State variables
  let currentBeliefs = [];
  let currentConflicts = [];
  let currentSuperseded = [];
  let currentCI = 1.0;
  let isHighlightingEnabled = true;
  let lastMessageCount = 0;
  let isDrawerOpen = false;
  let activeTab = "overview"; // "overview" | "facts" | "conflicts"
  let factSearchQuery = "";
  let userThemeSetting = "auto"; // "auto" | "light" | "dark"
  let soundEnabled = true;
  let previousBeliefCount = 0;
  let previousConflictCount = 0;

  // ==========================================
  // Web Audio Synthesis Engine for ALETHEX
  // ==========================================
  class AlethexAudioEngine {
    constructor() {
      this.ctx = null;
      this.enabled = true;
      this.lastPlayTime = 0;
      this.minInterval = 350;
    }

    init() {
      if (!this.ctx && typeof window !== "undefined") {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          try {
            this.ctx = new AudioCtx();
          } catch (e) {}
        }
      }
    }

    ensureContext() {
      this.init();
      if (this.ctx && this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }
    }

    playVerify() {
      if (!this.enabled) return;
      const now = Date.now();
      if (now - this.lastPlayTime < this.minInterval) return;
      this.lastPlayTime = now;

      this.ensureContext();
      if (!this.ctx) return;

      try {
        const t = this.ctx.currentTime;
        const osc1 = this.ctx.createOscillator();
        const osc2 = this.ctx.createOscillator();
        const gain1 = this.ctx.createGain();
        const gain2 = this.ctx.createGain();
        const filter = this.ctx.createBiquadFilter();

        filter.type = "lowpass";
        filter.frequency.setValueAtTime(2600, t);

        osc1.type = "sine";
        osc1.frequency.setValueAtTime(587.33, t);

        osc2.type = "sine";
        osc2.frequency.setValueAtTime(880.00, t + 0.04);

        gain1.gain.setValueAtTime(0.0001, t);
        gain1.gain.exponentialRampToValueAtTime(0.08, t + 0.01);
        gain1.gain.exponentialRampToValueAtTime(0.0001, t + 0.24);

        gain2.gain.setValueAtTime(0.0001, t);
        gain2.gain.setValueAtTime(0.0001, t + 0.04);
        gain2.gain.exponentialRampToValueAtTime(0.09, t + 0.055);
        gain2.gain.exponentialRampToValueAtTime(0.0001, t + 0.32);

        osc1.connect(gain1);
        osc2.connect(gain2);
        gain1.connect(filter);
        gain2.connect(filter);
        filter.connect(this.ctx.destination);

        osc1.start(t);
        osc2.start(t + 0.04);
        osc1.stop(t + 0.25);
        osc2.stop(t + 0.33);
      } catch (e) {}
    }

    playConflict() {
      if (!this.enabled) return;
      const now = Date.now();
      if (now - this.lastPlayTime < this.minInterval) return;
      this.lastPlayTime = now;

      this.ensureContext();
      if (!this.ctx) return;

      try {
        const t = this.ctx.currentTime;
        const osc1 = this.ctx.createOscillator();
        const osc2 = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        const filter = this.ctx.createBiquadFilter();

        filter.type = "lowpass";
        filter.frequency.setValueAtTime(1400, t);

        osc1.type = "triangle";
        osc1.frequency.setValueAtTime(520, t);
        osc1.frequency.exponentialRampToValueAtTime(390, t + 0.28);

        osc2.type = "sine";
        osc2.frequency.setValueAtTime(488, t);
        osc2.frequency.exponentialRampToValueAtTime(366, t + 0.28);

        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.09, t + 0.012);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.30);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(filter);
        filter.connect(this.ctx.destination);

        osc1.start(t);
        osc2.start(t);
        osc1.stop(t + 0.31);
        osc2.stop(t + 0.31);
      } catch (e) {}
    }

    playAudit() {
      if (!this.enabled) return;
      this.ensureContext();
      if (!this.ctx) return;

      try {
        const t = this.ctx.currentTime;
        const oscSweep = this.ctx.createOscillator();
        const gainSweep = this.ctx.createGain();

        oscSweep.type = "sine";
        oscSweep.frequency.setValueAtTime(260, t);
        oscSweep.frequency.exponentialRampToValueAtTime(740, t + 0.18);

        gainSweep.gain.setValueAtTime(0.0001, t);
        gainSweep.gain.exponentialRampToValueAtTime(0.07, t + 0.02);
        gainSweep.gain.exponentialRampToValueAtTime(0.0001, t + 0.19);

        const oscPing = this.ctx.createOscillator();
        const gainPing = this.ctx.createGain();

        oscPing.type = "sine";
        oscPing.frequency.setValueAtTime(880, t + 0.18);

        gainPing.gain.setValueAtTime(0.0001, t);
        gainPing.gain.setValueAtTime(0.0001, t + 0.17);
        gainPing.gain.exponentialRampToValueAtTime(0.08, t + 0.19);
        gainPing.gain.exponentialRampToValueAtTime(0.0001, t + 0.38);

        oscSweep.connect(gainSweep);
        gainSweep.connect(this.ctx.destination);
        oscPing.connect(gainPing);
        gainPing.connect(this.ctx.destination);

        oscSweep.start(t);
        oscSweep.stop(t + 0.20);
        oscPing.start(t + 0.18);
        oscPing.stop(t + 0.40);
      } catch (e) {}
    }

    playClick() {
      if (!this.enabled) return;
      this.ensureContext();
      if (!this.ctx) return;

      try {
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(1200, t);
        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.04, t + 0.002);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.025);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(t);
        osc.stop(t + 0.03);
      } catch (e) {}
    }
  }

  const soundEngine = new AlethexAudioEngine();

  // Passive unlock on first user gesture
  const unlockAudio = () => {
    soundEngine.ensureContext();
    window.removeEventListener("pointerdown", unlockAudio);
    window.removeEventListener("keydown", unlockAudio);
  };
  window.addEventListener("pointerdown", unlockAudio, { passive: true });
  window.addEventListener("keydown", unlockAudio, { passive: true });

  // NLI Worker — Transformers.js model running in a Web Worker (non-blocking)
  let nliWorker = null;
  let nliWorkerReady = false;
  let nliCallbacks = {}; // id -> {resolve, reject}
  let nliCallId = 0;

  function initNLIWorker() {
    if (nliWorker) return;
    try {
      nliWorker = new Worker(chrome.runtime.getURL("nli_worker.js"));
      nliWorker.onmessage = (e) => {
        const { id, type, results, error, message } = e.data;
        if (type === "status") {
          if (message && message.includes("ready")) nliWorkerReady = true;
          return;
        }
        if (type === "error") { return; }
        if (type === "pong") { return; }
        if (id && nliCallbacks[id]) {
          nliCallbacks[id].resolve(results || []);
          delete nliCallbacks[id];
        }
      };
      nliWorker.onerror = () => {};
      nliWorker.postMessage({ id: 0, type: "ping" });
    } catch (e) {
      // Worker fallback
    }
  }

  function nliInfer(pairs, threshold) {
    return new Promise((resolve) => {
      if (!nliWorker || pairs.length === 0) { resolve([]); return; }
      const id = ++nliCallId;
      nliCallbacks[id] = { resolve };
      nliWorker.postMessage({ id, type: "nli_batch", data: { pairs, threshold } });
      setTimeout(() => {
        if (nliCallbacks[id]) { delete nliCallbacks[id]; resolve([]); }
      }, 30000);
    });
  }

  // ==========================================
  // 1. Theme Management (Light & Dark)
  // ==========================================
  function detectHostTheme() {
    if (userThemeSetting === "light" || userThemeSetting === "dark") {
      return userThemeSetting;
    }
    const docEl = document.documentElement;
    const body = document.body;

    // Check host class or data attributes (ChatGPT, Claude, etc.)
    const isDarkClass = docEl.classList.contains("dark") ||
                        body.classList.contains("dark") ||
                        docEl.getAttribute("data-theme") === "dark" ||
                        docEl.getAttribute("data-color-mode") === "dark";

    if (isDarkClass) return "dark";

    // Check system preference
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }

    return "light";
  }

  function syncTheme() {
    const root = document.getElementById("alethex-root");
    if (!root) return;
    const effective = detectHostTheme();
    root.setAttribute("data-theme", effective);

    const themeToggleBtn = document.getElementById("alethex-theme-btn");
    if (themeToggleBtn) {
      themeToggleBtn.title = `Current Theme: ${effective.toUpperCase()} (Click to toggle)`;
      themeToggleBtn.innerHTML = effective === "light"
        ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line></svg>`
        : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
    }
  }

  // ==========================================
  // 2. Inject Styles (Human-Crafted Dual Theme)
  // ==========================================
  function injectStyles() {
    if (document.getElementById("alethex-styles")) return;
    const style = document.createElement("style");
    style.id = "alethex-styles";
    style.textContent = `
      #alethex-root {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 2147483647;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Inter", sans-serif;
        pointer-events: none;
        -webkit-font-smoothing: antialiased;
      }

      /* Dark Theme Tokens */
      #alethex-root[data-theme="dark"] {
        --al-bg-hud: rgba(18, 22, 30, 0.88);
        --al-bg-drawer: #0f131a;
        --al-bg-header: #151a24;
        --al-bg-surface: #171d27;
        --al-bg-surface-elevated: #1f2633;
        --al-bg-surface-hover: #262f40;
        --al-border-subtle: rgba(255, 255, 255, 0.08);
        --al-border-strong: rgba(255, 255, 255, 0.16);
        --al-text-primary: #f1f5f9;
        --al-text-secondary: #94a3b8;
        --al-text-tertiary: #64748b;
        --al-accent: #2563eb;
        --al-accent-hover: #1d4ed8;
        --al-accent-subtle: rgba(37, 99, 235, 0.12);
        --al-accent-text: #60a5fa;
        --al-green: #10b981;
        --al-green-text: #34d399;
        --al-green-subtle: rgba(16, 185, 129, 0.12);
        --al-amber: #f59e0b;
        --al-amber-text: #fbbf24;
        --al-amber-subtle: rgba(245, 158, 11, 0.12);
        --al-red: #ef4444;
        --al-red-text: #f87171;
        --al-red-subtle: rgba(239, 68, 68, 0.12);
        --al-shadow-hud: 0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 2px 6px -1px rgba(0, 0, 0, 0.3);
        --al-shadow-drawer: 0 20px 45px -10px rgba(0, 0, 0, 0.75), 0 0 0 1px var(--al-border-subtle);
        --al-tab-bg: #11151d;
      }

      /* Light Theme Tokens */
      #alethex-root[data-theme="light"] {
        --al-bg-hud: rgba(255, 255, 255, 0.92);
        --al-bg-drawer: #ffffff;
        --al-bg-header: #f8fafc;
        --al-bg-surface: #f8fafc;
        --al-bg-surface-elevated: #f1f5f9;
        --al-bg-surface-hover: #e2e8f0;
        --al-border-subtle: rgba(15, 23, 42, 0.08);
        --al-border-strong: rgba(15, 23, 42, 0.16);
        --al-text-primary: #0f172a;
        --al-text-secondary: #475569;
        --al-text-tertiary: #94a3b8;
        --al-accent: #2563eb;
        --al-accent-hover: #1d4ed8;
        --al-accent-subtle: rgba(37, 99, 235, 0.08);
        --al-accent-text: #1d4ed8;
        --al-green: #059669;
        --al-green-text: #059669;
        --al-green-subtle: rgba(5, 150, 105, 0.08);
        --al-amber: #d97706;
        --al-amber-text: #d97706;
        --al-amber-subtle: rgba(217, 119, 6, 0.08);
        --al-red: #dc2626;
        --al-red-text: #dc2626;
        --al-red-subtle: rgba(220, 38, 38, 0.08);
        --al-shadow-hud: 0 4px 20px -2px rgba(15, 23, 42, 0.12), 0 2px 6px -1px rgba(15, 23, 42, 0.06);
        --al-shadow-drawer: 0 20px 45px -10px rgba(15, 23, 42, 0.18), 0 0 0 1px var(--al-border-subtle);
        --al-tab-bg: #e2e8f0;
      }

      /* Floating HUD Capsule */
      #alethex-hud {
        pointer-events: auto;
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--al-bg-hud);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--al-border-subtle);
        box-shadow: var(--al-shadow-hud);
        border-radius: 9999px;
        padding: 6px 13px;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        user-select: none;
        color: var(--al-text-primary);
      }
      #alethex-hud:hover {
        transform: translateY(-2px);
        border-color: var(--al-border-strong);
        box-shadow: var(--al-shadow-hud), 0 0 0 3px var(--al-accent-subtle);
      }
      .alethex-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--al-green);
        position: relative;
        transition: background-color 0.25s ease;
      }
      .alethex-dot.amber { background: var(--al-amber); }
      .alethex-dot.red { background: var(--al-red); }
      .alethex-dot.gray { background: var(--al-text-tertiary); }

      .alethex-hud-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: -0.1px;
        color: var(--al-text-primary);
      }
      .alethex-hud-sub {
        font-size: 11px;
        color: var(--al-text-secondary);
        font-weight: 500;
      }

      /* Inspector Drawer / Slide-Over Modal */
      #alethex-drawer {
        pointer-events: auto;
        position: absolute;
        bottom: 46px;
        right: 0;
        width: 384px;
        max-width: calc(100vw - 32px);
        max-height: 560px;
        background: var(--al-bg-drawer);
        border: 1px solid var(--al-border-subtle);
        box-shadow: var(--al-shadow-drawer);
        border-radius: 14px;
        display: none;
        flex-direction: column;
        overflow: hidden;
        animation: alethex-pop 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        color: var(--al-text-primary);
      }
      @keyframes alethex-pop {
        from { opacity: 0; transform: translateY(8px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }

      /* Drawer Header */
      .alethex-drawer-header {
        padding: 12px 16px;
        background: var(--al-bg-header);
        border-bottom: 1px solid var(--al-border-subtle);
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .alethex-brand-lockup {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .alethex-brand-badge {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: var(--al-accent-subtle);
        border: 1px solid var(--al-border-subtle);
        color: var(--al-accent-text);
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .alethex-drawer-title {
        font-size: 13px;
        font-weight: 700;
        color: var(--al-text-primary);
        line-height: 1.2;
      }
      .alethex-platform-pill {
        font-size: 10px;
        color: var(--al-text-secondary);
        font-weight: 500;
      }
      .alethex-header-actions {
        display: flex;
        align-items: center;
        gap: 5px;
      }
      .alethex-tool-btn {
        background: var(--al-bg-surface);
        border: 1px solid var(--al-border-subtle);
        color: var(--al-text-secondary);
        width: 26px;
        height: 26px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.15s ease;
      }
      .alethex-tool-btn:hover {
        background: var(--al-bg-surface-hover);
        color: var(--al-text-primary);
        border-color: var(--al-border-strong);
      }

      /* Segmented Tabs Control */
      .alethex-tabs {
        display: flex;
        gap: 4px;
        padding: 8px 16px;
        background: var(--al-bg-header);
        border-bottom: 1px solid var(--al-border-subtle);
      }
      .alethex-tab-btn {
        flex: 1;
        background: transparent;
        border: none;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        font-family: inherit;
        color: var(--al-text-secondary);
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
      }
      .alethex-tab-btn:hover {
        color: var(--al-text-primary);
        background: var(--al-bg-surface);
      }
      .alethex-tab-btn.active {
        background: var(--al-bg-surface-elevated);
        color: var(--al-text-primary);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
      }
      .alethex-tab-count {
        font-size: 10px;
        font-family: var(--font-mono);
        padding: 1px 5px;
        border-radius: 999px;
        background: var(--al-bg-surface);
        color: var(--al-text-secondary);
      }
      .alethex-tab-btn.active .alethex-tab-count {
        background: var(--al-accent-subtle);
        color: var(--al-accent-text);
      }

      /* Drawer Content Body */
      .alethex-drawer-body {
        padding: 14px 16px;
        overflow-y: auto;
        max-height: 400px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        font-size: 12px;
      }

      /* Card Elements */
      .alethex-card {
        background: var(--al-bg-surface);
        border: 1px solid var(--al-border-subtle);
        border-radius: 8px;
        padding: 10px 12px;
        transition: border-color 0.15s ease;
      }
      .alethex-card:hover {
        border-color: var(--al-border-strong);
      }

      /* Badges */
      .alethex-badge {
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        letter-spacing: 0.2px;
      }
      .alethex-badge.green { background: var(--al-green-subtle); color: var(--al-green-text); border: 1px solid rgba(16, 185, 129, 0.2); }
      .alethex-badge.amber { background: var(--al-amber-subtle); color: var(--al-amber-text); border: 1px solid rgba(245, 158, 11, 0.2); }
      .alethex-badge.red { background: var(--al-red-subtle); color: var(--al-red-text); border: 1px solid rgba(239, 68, 68, 0.2); }

      /* Search Input */
      .alethex-search-box {
        position: relative;
        margin-bottom: 4px;
      }
      .alethex-search-input {
        width: 100%;
        background: var(--al-bg-surface);
        border: 1px solid var(--al-border-subtle);
        border-radius: 6px;
        padding: 7px 10px 7px 28px;
        font-size: 11px;
        font-family: inherit;
        color: var(--al-text-primary);
        outline: none;
        transition: border-color 0.15s ease;
      }
      .alethex-search-input:focus {
        border-color: var(--al-accent);
      }
      .alethex-search-icon {
        position: absolute;
        left: 8px;
        top: 8px;
        color: var(--al-text-tertiary);
        pointer-events: none;
      }

      /* Buttons */
      .alethex-action-btn {
        width: 100%;
        background: var(--al-accent);
        color: #ffffff;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        font-family: inherit;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        transition: background-color 0.15s ease;
      }
      .alethex-action-btn:hover { background: var(--al-accent-hover); }

      .alethex-sub-btn {
        width: 100%;
        background: var(--al-bg-surface);
        border: 1px solid var(--al-border-subtle);
        color: var(--al-text-primary);
        padding: 7px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        font-family: inherit;
        cursor: pointer;
        margin-top: 4px;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
      }
      .alethex-sub-btn:hover {
        background: var(--al-bg-surface-hover);
        border-color: var(--al-border-strong);
      }

      /* In-Chat Message Highlighting (Subtle & Non-Disruptive) */
      .alethex-highlight {
        position: relative !important;
        border-left: 3.5px solid var(--al-red) !important;
        background: var(--al-red-subtle) !important;
        border-radius: 0 8px 8px 0 !important;
        transition: background-color 0.2s ease;
      }
      .alethex-highlight-amber {
        position: relative !important;
        border-left: 3.5px solid var(--al-amber) !important;
        background: var(--al-amber-subtle) !important;
        border-radius: 0 8px 8px 0 !important;
        transition: background-color 0.2s ease;
      }
    `;
    document.head.appendChild(style);
  }

  // ==========================================
  // 3. Initialize HUD and In-Page Drawer
  // ==========================================
  function initDOM() {
    injectStyles();

    let root = document.getElementById("alethex-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "alethex-root";
      root.setAttribute("data-theme", detectHostTheme());
      root.innerHTML = `
        <div id="alethex-hud" title="Click to open ALETHEX Truth Drawer">
          <span class="alethex-dot" id="alethex-dot"></span>
          <span class="alethex-hud-title">ALETHEX</span>
          <span style="color:var(--al-text-tertiary); font-size:10px;">·</span>
          <span class="alethex-hud-sub" id="alethex-hud-status">Guarding</span>
        </div>
        <div id="alethex-drawer">
          <div class="alethex-drawer-header">
            <div class="alethex-brand-lockup">
              <div class="alethex-brand-badge" style="overflow:hidden; display:flex; align-items:center; justify-content:center; padding:1px;">
                <img src="${chrome.runtime.getURL('icon48.png')}" style="width:14px; height:14px; border-radius:3px; display:block;" alt="ALETHEX">
              </div>
              <div>
                <div class="alethex-drawer-title">ALETHEX Inspector</div>
                <div class="alethex-platform-pill">${platform} Active</div>
              </div>
            </div>
            <div class="alethex-header-actions">
              <button class="alethex-tool-btn" id="alethex-sound-btn" title="Toggle Acoustic Cues (Sound On/Off)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
                </svg>
              </button>
              <button class="alethex-tool-btn" id="alethex-theme-btn" title="Toggle Light / Dark Theme">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="5"></circle>
                  <line x1="12" y1="1" x2="12" y2="3"></line>
                  <line x1="12" y1="21" x2="12" y2="23"></line>
                </svg>
              </button>
              <button class="alethex-tool-btn" id="alethex-close" title="Close Drawer">&times;</button>
            </div>
          </div>

          <!-- Tab Bar -->
          <div class="alethex-tabs">
            <button class="alethex-tab-btn active" id="tab-btn-overview">
              <span>Overview</span>
            </button>
            <button class="alethex-tab-btn" id="tab-btn-facts">
              <span>Facts</span>
              <span class="alethex-tab-count" id="tab-count-facts">0</span>
            </button>
            <button class="alethex-tab-btn" id="tab-btn-conflicts">
              <span>Conflicts</span>
              <span class="alethex-tab-count" id="tab-count-conflicts">0</span>
            </button>
          </div>

          <!-- Body -->
          <div class="alethex-drawer-body" id="alethex-drawer-body">
            <!-- Dynamic Content Injected Here -->
          </div>
        </div>
      `;
      document.body.appendChild(root);

      // Bind events
      document.getElementById("alethex-hud").addEventListener("click", toggleDrawer);
      document.getElementById("alethex-close").addEventListener("click", () => {
        soundEngine.playClick();
        isDrawerOpen = false;
        document.getElementById("alethex-drawer").style.display = "none";
      });

      // Sound button toggle
      const soundBtn = document.getElementById("alethex-sound-btn");
      if (soundBtn) {
        soundBtn.addEventListener("click", () => {
          soundEngine.enabled = !soundEngine.enabled;
          try {
            chrome.storage.local.set({ soundEnabled: soundEngine.enabled });
          } catch (e) {}
          renderDrawerSoundIcon();
          if (soundEngine.enabled) {
            soundEngine.playVerify();
          }
        });
      }

      // In-page Theme Toggle
      document.getElementById("alethex-theme-btn").addEventListener("click", () => {
        soundEngine.playClick();
        const cur = detectHostTheme();
        userThemeSetting = cur === "dark" ? "light" : "dark";
        try {
          chrome.storage.local.set({ userTheme: userThemeSetting });
        } catch (e) {}
        syncTheme();
      });

      // Tabs click listeners
      document.getElementById("tab-btn-overview").addEventListener("click", () => switchDrawerTab("overview"));
      document.getElementById("tab-btn-facts").addEventListener("click", () => switchDrawerTab("facts"));
      document.getElementById("tab-btn-conflicts").addEventListener("click", () => switchDrawerTab("conflicts"));
      renderDrawerSoundIcon();
    }
  }

  function renderDrawerSoundIcon() {
    const btn = document.getElementById("alethex-sound-btn");
    if (!btn) return;
    if (soundEngine.enabled) {
      btn.title = "Acoustic Cues: Enabled (Click to mute)";
      btn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
        </svg>
      `;
      btn.style.color = "var(--al-text-primary)";
    } else {
      btn.title = "Acoustic Cues: Muted (Click to enable)";
      btn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <line x1="23" y1="9" x2="17" y2="15"></line>
          <line x1="17" y1="9" x2="23" y2="15"></line>
        </svg>
      `;
      btn.style.color = "var(--al-text-tertiary)";
    }
  }

  function switchDrawerTab(tab) {
    soundEngine.playClick();
    activeTab = tab;
    document.querySelectorAll(".alethex-tab-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`tab-btn-${tab}`);
    if (activeBtn) activeBtn.classList.add("active");
    updateDrawerUI();
  }

  function toggleDrawer() {
    soundEngine.playClick();
    isDrawerOpen = !isDrawerOpen;
    const drawer = document.getElementById("alethex-drawer");
    if (drawer) {
      drawer.style.display = isDrawerOpen ? "flex" : "none";
      if (isDrawerOpen) {
        syncTheme();
        renderDrawerSoundIcon();
        updateDrawerUI();
      }
    }
  }

  // ==========================================
  // 4. Safe HTML Escaping
  // ==========================================
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ==========================================
  // 5. Message Scraper (Non-blocking)
  // ==========================================
  const GENERIC_SELECTOR = 'article, [class*="message"], [data-message-author-role], [class*="turn"], .prose, p';

  function getTurnNodes() {
    let selector = GENERIC_SELECTOR;
    if (platform === "ChatGPT") selector += ', [data-message-author-role], .whitespace-pre-wrap';
    else if (platform === "Claude.ai") selector += ', .font-claude-message, .font-user-message, div.prose, [data-testid*="message"]';
    else if (platform === "Google Gemini") selector += ', .user-query-container, .response-container, message-content';
    else if (platform === "DeepSeek") selector += ', .ds-markdown';
    else if (platform === "Perplexity") selector += ', [class*="answer"]';
    else if (platform === "Copilot") selector += ', cib-message-group, .ac-textBlock';

    const rawNodes = document.querySelectorAll(selector);
    const seen = new Set();
    const valid = [];
    rawNodes.forEach((node) => {
      if (node.closest && node.closest("#alethex-root")) return;
      if (seen.has(node)) return;
      const text = node.innerText ? node.innerText.trim() : "";
      if (text.length > 15 && text.length < 5000) {
        const isNestedDuplicate = valid.some((v) => v.node !== node && v.node.contains(node));
        if (isNestedDuplicate) return;
        seen.add(node);
        valid.push({ node, text });
      }
    });
    return valid;
  }

  // ==========================================
  // 6. Factual Reconciliation Logic
  // ==========================================
  function reconcileTurns(turns) {
    const claims = [];
    const entitySlots = {};
    const conflicts = [];
    const superseded = [];

    const SUBJ = "(?:i am|i'm|i|my name is|i'll be|i am now|i'm now)";
    const patterns = [
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,10}(?:now\\s+)?(?:work(?:ing)?\\s+(?:at|for)|employed (?:at|by)|joined|started (?:at|working at))\\s+([a-z0-9&. -]{2,40})`, "i"), pred: "works_at" },
      { regex: /\bmy (?:company|employer|workplace|office) is\s+([-a-z0-9&. ]{2,40})/i, pred: "works_at" },
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,10}(?:now\\s+)?(?:live(?:s|d)?\\s+in|based in|moved to|relocated to|living in|staying in|currently in|reside in|residing in|am from|grew up in)\\s+([-a-z0-9,. ]{2,40})`, "i"), pred: "lives_in" },
      { regex: /\bmy (?:city|town|country|location|home(?:town)?) is\s+([-a-z0-9,. ]{2,40})/i, pred: "lives_in" },
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,5}(?:now\\s+)?(?:use|uses|using|prefer|prefers|switched to|migrated to)\\s+([-a-z0-9#+. ]{2,30})`, "i"), pred: "uses" },
      { regex: /\bmy (?:main |primary |daily )?(?:language|stack|framework|editor|ide|os|browser|phone|laptop|computer) is\s+([-a-z0-9#+. ]{2,30})/i, pred: "uses" },
      { regex: /\bmy name is\s+([a-z][-a-z'. ]{1,25})/i, pred: "name" },
      { regex: /\bpeople call me\s+([a-z][-a-z'. ]{1,20})/i, pred: "name" },
      { regex: /\bi(?:'m| am)\s+(\d{1,3})\s*(?:years old|yo|years of age)?\b/i, pred: "age" },
      { regex: /\bmy age is\s+(\d{1,3})\b/i, pred: "age" },
      { regex: /\b(?:i am|i'm|i've become|i became|i went)\s+(?:a |an )?(vegetarian|vegan|pescatarian|meat eater|omnivore|carnivore|keto|paleo)\b/i, pred: "diet" },
      { regex: /\bi\s+(?:don'?t|do not|no longer)\s+(?:eat meat|eat animal|consume meat)\b/i, pred: "diet", forcedValue: "vegetarian" },
      { regex: /\bi\s+(?:eat meat|am back to eating meat)\b/i, pred: "diet", forcedValue: "omnivore" },
      { regex: /\bi\s+(?:like|love|enjoy|adore|am a fan of)\s+([-a-z0-9 ]{2,25})/i, pred: "likes" },
      { regex: /\bi\s+(?:dislike|hate|despise|don'?t like|do not like|can'?t stand|no longer like)\s+([-a-z0-9 ]{2,25})/i, pred: "dislikes" },
      { regex: /\bmy (?:job|role|title|position|profession|occupation) is\s+(?:a |an )?([-a-z0-9 ]{2,30})/i, pred: "role" },
      { regex: /\bi work as (?:a |an )?([-a-z0-9 ]{2,25})/i, pred: "role" },
      { regex: /\bi(?:'m| am) (?:a |an )?([-a-z0-9 ]{2,30}(?:developer|engineer|designer|manager|analyst|scientist|researcher|student|teacher|writer|founder|cto|ceo|coo|cfo))\b/i, pred: "role" },
      { regex: /\bi(?:'m| am)\s+(single|married|engaged|divorced|in a relationship)\b/i, pred: "relationship" },
      { regex: /\bmy (?:native |first |primary )?language is\s+([-a-z ]{2,20})/i, pred: "language" },
      { regex: /\bi(?:'m| am) (?:a )?(?:native |fluent )?([a-z]{3,15}) speaker\b/i, pred: "language" },
    ];

    const TRIM_AT = /\b(?:and|but|so|because|which|who|that|while|although|as a|as an)\b.*$/i;

    function normalizeValue(raw) {
      let v = raw.replace(TRIM_AT, "").trim();
      v = v.replace(/[.,!?]+$/, "").trim();
      return v.toLowerCase();
    }

    function valuesMatch(a, b) {
      if (!a || !b) return false;
      if (a === b) return true;
      return a.length > 2 && b.length > 2 && (a.includes(b) || b.includes(a));
    }

    const months = { jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5, jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11 };

    function extractDate(text, fallback) {
      const m = text.match(/\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:(\d{4}))?\b/i);
      if (m) {
        const mKey = m[1].substring(0, 3).toLowerCase();
        const year = m[2] ? parseInt(m[2]) : fallback.getFullYear();
        return new Date(year, months[mKey] || 0, 1);
      }
      return fallback;
    }

    turns.forEach((turn, idx) => {
      const now = new Date(Date.now() - (turns.length - idx) * 60000);
      const sentences = turn.text.split(/(?<=[.!?])\s+/);

      sentences.forEach((sent) => {
        patterns.forEach((pat) => {
          const match = sent.match(pat.regex);
          if (match) {
            const rawVal = pat.forcedValue || match[1] || "";
            const val = normalizeValue(rawVal);
            if (!val || val.split(/\s+/).length > 4) return;
            if (/^(?:the|a|an|it|its|this|that|these|those|my|your|our|their)\b/i.test(val)) return;

            const date = extractDate(sent, now);
            const claim = {
              turnIdx: idx,
              node: turn.node,
              predicate: pat.pred,
              value: val,
              start: date,
              end: null,
              raw: sent.trim()
            };
            claims.push(claim);

            if (!entitySlots[pat.pred]) entitySlots[pat.pred] = [];
            entitySlots[pat.pred].push(claim);
          }
        });
      });
    });

    const OPPOSITES = { likes: "dislikes", dislikes: "likes" };

    Object.keys(entitySlots).forEach((pred) => {
      const slotClaims = entitySlots[pred];
      slotClaims.sort((a, b) => a.turnIdx - b.turnIdx);

      for (let i = 0; i < slotClaims.length; i++) {
        for (let j = i + 1; j < slotClaims.length; j++) {
          const c1 = slotClaims[i];
          const c2 = slotClaims[j];
          if (valuesMatch(c1.value, c2.value)) continue;

          if (c2.turnIdx > c1.turnIdx) {
            c1.end = c2.start;
            if (!superseded.includes(c1)) superseded.push(c1);
          } else {
            conflicts.push({ claimA: c1, claimB: c2, reason: `Contradiction: '${c1.value}' vs '${c2.value}'` });
          }
        }
      }

      const oppositePred = OPPOSITES[pred];
      if (oppositePred && entitySlots[oppositePred]) {
        slotClaims.forEach((c1) => {
          entitySlots[oppositePred].forEach((c2) => {
            if (valuesMatch(c1.value, c2.value)) {
              conflicts.push({ claimA: c1, claimB: c2, reason: `Opposing values for '${c1.value}'` });
            }
          });
        });
      }
    });

    const active = claims.filter((c) => !superseded.includes(c));
    const totalClaims = Math.max(1, claims.length);
    const issueScore = conflicts.length + superseded.length * 0.5;
    const ci = claims.length === 0 ? null : Math.max(0, 1.0 - issueScore / totalClaims);

    return { active, superseded, conflicts, ci };
  }

  // ==========================================
  // 7. Update HUD UI
  // ==========================================
  function updateHUD() {
    const dot = document.getElementById("alethex-dot");
    const status = document.getElementById("alethex-hud-status");
    if (!dot || !status) return;

    if (currentConflicts.length > 0) {
      dot.className = "alethex-dot red";
      status.innerText = `${currentConflicts.length} Conflict${currentConflicts.length > 1 ? "s" : ""}`;
    } else if (currentSuperseded.length > 0) {
      dot.className = "alethex-dot amber";
      status.innerText = `${currentSuperseded.length} Updated`;
    } else if (currentCI === null) {
      dot.className = "alethex-dot gray";
      status.innerText = "Guarding";
    } else {
      dot.className = "alethex-dot";
      status.innerText = `${currentBeliefs.length} Fact${currentBeliefs.length === 1 ? "" : "s"} Verified`;
    }

    const countFactsEl = document.getElementById("tab-count-facts");
    if (countFactsEl) countFactsEl.innerText = currentBeliefs.length;

    const countConflictsEl = document.getElementById("tab-count-conflicts");
    if (countConflictsEl) countConflictsEl.innerText = currentConflicts.length + currentSuperseded.length;
  }

  // ==========================================
  // 8. Update Drawer UI (Multi-Tab Architecture)
  // ==========================================
  let lastAuditNotice = null;

  function updateDrawerUI() {
    const body = document.getElementById("alethex-drawer-body");
    if (!body) return;

    let contentHtml = "";

    // ------------------------------------------
    // TAB: OVERVIEW
    // ------------------------------------------
    if (activeTab === "overview") {
      let badgeClass = "green";
      let badgeText = currentCI === null ? "No Claims Yet" : `Healthy (${Math.round((currentCI || 1) * 100)}%)`;
      if (currentCI === null) badgeClass = "amber";
      if (currentConflicts.length > 0) {
        badgeClass = "red";
        badgeText = `Conflict (${Math.round(currentCI * 100)}%)`;
      } else if (currentSuperseded.length > 0) {
        badgeClass = "amber";
        badgeText = `Updated (${Math.round(currentCI * 100)}%)`;
      }

      const ciScoreVal = currentCI === null ? 100 : Math.round(currentCI * 100);

      contentHtml = `
        ${lastAuditNotice ? `<div class="alethex-card" style="color:var(--al-amber-text);">${lastAuditNotice}</div>` : ""}

        <!-- Consistency Gauge -->
        <div class="alethex-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:11px; font-weight:600; color:var(--al-text-secondary); text-transform:uppercase; letter-spacing:0.3px;">Memory Coherence</span>
            <span class="alethex-badge ${badgeClass}">${badgeText}</span>
          </div>
          <div style="height:6px; background:var(--al-bg-surface-elevated); border-radius:999px; overflow:hidden; margin-bottom:8px;">
            <div style="height:100%; width:${ciScoreVal}%; background:var(--al-${badgeClass === 'green' ? 'green' : (badgeClass === 'amber' ? 'amber' : 'red')}); border-radius:999px;"></div>
          </div>
          <div style="font-size:11px; color:var(--al-text-secondary); line-height:1.4;">
            Autonomous neuro-symbolic verification active. Conflicting and superseded statements are continuously audited.
          </div>
        </div>

        <!-- Metric Summary -->
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px;">
          <div class="alethex-card" style="text-align:center; padding:8px 4px;">
            <div style="font-size:16px; font-weight:700; font-family:var(--font-mono); color:var(--al-text-primary);">${currentBeliefs.length}</div>
            <div style="font-size:10px; color:var(--al-text-secondary); margin-top:2px;">Active Facts</div>
          </div>
          <div class="alethex-card" style="text-align:center; padding:8px 4px;">
            <div style="font-size:16px; font-weight:700; font-family:var(--font-mono); color:${currentSuperseded.length > 0 ? 'var(--al-amber-text)' : 'var(--al-text-primary)'};">${currentSuperseded.length}</div>
            <div style="font-size:10px; color:var(--al-text-secondary); margin-top:2px;">Superseded</div>
          </div>
          <div class="alethex-card" style="text-align:center; padding:8px 4px;">
            <div style="font-size:16px; font-weight:700; font-family:var(--font-mono); color:${currentConflicts.length > 0 ? 'var(--al-red-text)' : 'var(--al-green-text)'};">${currentConflicts.length}</div>
            <div style="font-size:10px; color:var(--al-text-secondary); margin-top:2px;">Conflicts</div>
          </div>
        </div>

        <!-- Controls -->
        <button id="alethex-rescan-btn" class="alethex-action-btn">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
          <span>Run Live Audit</span>
        </button>

        <button id="alethex-toggle-hl-btn" class="alethex-sub-btn">
          <span>${isHighlightingEnabled ? "Hide In-Chat Ribbons" : "Show In-Chat Ribbons"}</span>
        </button>
      `;
    }

    // ------------------------------------------
    // TAB: FACTS (Active Beliefs Frontier)
    // ------------------------------------------
    else if (activeTab === "facts") {
      let filtered = currentBeliefs;
      if (factSearchQuery.trim()) {
        const q = factSearchQuery.toLowerCase();
        filtered = currentBeliefs.filter(b => b.predicate.toLowerCase().includes(q) || b.value.toLowerCase().includes(q) || b.raw.toLowerCase().includes(q));
      }

      const factItemsHtml = filtered.length > 0
        ? filtered.map((b) => `
            <div class="alethex-card" style="display:flex; flex-direction:column; gap:4px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="alethex-badge green">${escapeHtml(b.predicate.replace('_', ' '))}</span>
                <span style="font-size:10px; color:var(--al-text-tertiary); font-family:var(--font-mono);">Turn ${b.turnIdx + 1}</span>
              </div>
              <div style="font-size:13px; font-weight:600; color:var(--al-text-primary); margin-top:2px;">${escapeHtml(b.value)}</div>
              <div style="font-size:10px; color:var(--al-text-secondary); font-style:italic;">&ldquo;${escapeHtml(b.raw)}&rdquo;</div>
            </div>
          `).join("")
        : `<div style="text-align:center; padding:24px 12px; color:var(--al-text-tertiary);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="margin:0 auto 6px auto; opacity:0.6; display:block;">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            <div style="font-weight:500;">No matching facts found.</div>
            <div style="font-size:10px; margin-top:2px;">Factual assertions in the conversation appear here automatically.</div>
           </div>`;

      contentHtml = `
        <div class="alethex-search-box">
          <svg class="alethex-search-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="alethex-fact-search" class="alethex-search-input" placeholder="Search verified facts..." value="${escapeHtml(factSearchQuery)}" />
        </div>
        <div style="display:flex; flex-direction:column; gap:6px;">
          ${factItemsHtml}
        </div>
        ${currentBeliefs.length > 0 ? `<button id="alethex-copy-facts-btn" class="alethex-sub-btn">Copy Facts JSON</button>` : ""}
      `;
    }

    // ------------------------------------------
    // TAB: CONFLICTS (Contradiction & Supersession)
    // ------------------------------------------
    else if (activeTab === "conflicts") {
      const allIssues = [];
      currentConflicts.forEach(c => allIssues.push({ type: "conflict", data: c }));
      currentSuperseded.forEach(s => allIssues.push({ type: "superseded", data: s }));

      const issuesHtml = allIssues.length > 0
        ? allIssues.map((item) => {
            if (item.type === "conflict") {
              const c = item.data;
              return `
                <div class="alethex-card" style="border-left:3.5px solid var(--al-red);">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span class="alethex-badge red">Contradiction</span>
                    <button class="alethex-tool-btn" data-jump-turn="${c.claimB ? c.claimB.turnIdx : 0}" title="Scroll to Turn" style="width:20px; height:20px; font-size:9px;">↗</button>
                  </div>
                  <div style="font-weight:600; color:var(--al-red-text); font-size:11px; margin-bottom:4px;">${escapeHtml(c.reason)}</div>
                  <div style="font-size:11px; color:var(--al-text-secondary); background:var(--al-bg-surface-elevated); padding:6px 8px; border-radius:4px; margin-bottom:4px;">
                    &ldquo;${escapeHtml(c.claimA.raw)}&rdquo; <br>
                    <span style="color:var(--al-red-text); font-weight:600;">vs</span> <br>
                    &ldquo;${escapeHtml(c.claimB.raw)}&rdquo;
                  </div>
                </div>
              `;
            } else {
              const s = item.data;
              return `
                <div class="alethex-card" style="border-left:3.5px solid var(--al-amber);">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span class="alethex-badge amber">Superseded</span>
                    <button class="alethex-tool-btn" data-jump-turn="${s.turnIdx}" title="Scroll to Turn" style="width:20px; height:20px; font-size:9px;">↗</button>
                  </div>
                  <div style="font-size:11px; color:var(--al-text-primary); margin-bottom:2px;">
                    <span style="color:var(--al-text-tertiary); text-decoration:line-through;">${escapeHtml(s.predicate)}: ${escapeHtml(s.value)}</span>
                  </div>
                  <div style="font-size:10px; color:var(--al-text-secondary); font-style:italic;">
                    Updated in subsequent turn.
                  </div>
                </div>
              `;
            }
          }).join("")
        : `<div style="text-align:center; padding:28px 12px; color:var(--al-text-secondary);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin:0 auto 6px auto; color:var(--al-green-text); display:block;"><polyline points="20 6 9 17 4 12"></polyline></svg>
            <div style="font-weight:600; color:var(--al-text-primary);">Zero Contradictions Found</div>
            <div style="font-size:11px; color:var(--al-text-tertiary); margin-top:2px;">All extracted assertions across chat turns remain logically consistent.</div>
           </div>`;

      contentHtml = `
        <div style="display:flex; flex-direction:column; gap:6px;">
          ${issuesHtml}
        </div>
      `;
    }

    body.innerHTML = contentHtml;
    lastAuditNotice = null;

    // Attach dynamic listeners
    const rescanBtn = body.querySelector("#alethex-rescan-btn");
    if (rescanBtn) {
      rescanBtn.addEventListener("click", () => {
        soundEngine.playAudit();
        rescanBtn.innerText = "Scanning...";
        setTimeout(() => runAudit(true), 50);
      });
    }

    const toggleHlBtn = body.querySelector("#alethex-toggle-hl-btn");
    if (toggleHlBtn) {
      toggleHlBtn.addEventListener("click", () => {
        soundEngine.playClick();
        isHighlightingEnabled = !isHighlightingEnabled;
        applyHighlights();
        updateDrawerUI();
        try {
          chrome.storage.local.set({ highlightConflicts: isHighlightingEnabled });
        } catch (e) {}
      });
    }

    const searchInput = body.querySelector("#alethex-fact-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        factSearchQuery = e.target.value;
        updateDrawerUI();
        // Restore focus to input after re-render
        const reInput = document.getElementById("alethex-fact-search");
        if (reInput) {
          reInput.focus();
          reInput.setSelectionRange(reInput.value.length, reInput.value.length);
        }
      });
    }

    const copyBtn = body.querySelector("#alethex-copy-facts-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const json = JSON.stringify(currentBeliefs.map(b => ({ predicate: b.predicate, value: b.value, source: b.raw })), null, 2);
        navigator.clipboard.writeText(json).then(() => {
          copyBtn.innerText = "Copied to Clipboard!";
          setTimeout(() => { copyBtn.innerText = "Copy Facts JSON"; }, 1500);
        });
      });
    }

    // Scroll to turn click handlers
    body.querySelectorAll("[data-jump-turn]").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const turnIdx = parseInt(e.currentTarget.getAttribute("data-jump-turn"));
        const turns = getTurnNodes();
        if (turns[turnIdx] && turns[turnIdx].node) {
          turns[turnIdx].node.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    });
  }

  // ==========================================
  // 9. In-Chat Highlighting (Refined Left-Accent)
  // ==========================================
  function applyHighlights() {
    document.querySelectorAll(".alethex-highlight").forEach(el => el.classList.remove("alethex-highlight"));
    document.querySelectorAll(".alethex-highlight-amber").forEach(el => el.classList.remove("alethex-highlight-amber"));
    if (!isHighlightingEnabled) return;

    currentConflicts.forEach(conf => {
      if (conf.claimA && conf.claimA.node) conf.claimA.node.classList.add("alethex-highlight");
      if (conf.claimB && conf.claimB.node) conf.claimB.node.classList.add("alethex-highlight");
    });
    currentSuperseded.forEach(c => {
      if (c.node) c.node.classList.add("alethex-highlight-amber");
    });
  }

  // ==========================================
  // 10. Semantic NLI Worker Integration
  // ==========================================
  async function runSemanticNLI(turns) {
    if (!nliWorker) return;

    const FIRST_PERSON = /^(?:i |my |i'm |i've |i am |i was |i will |i do |i don't |i work|i live|i use)/i;
    const MIN_LEN = 15;
    const MAX_LEN = 200;

    const sentencesByTurn = turns.map((turn, idx) => {
      const sents = turn.text.split(/(?<=[.!?])\s+/).filter(s =>
        s.length >= MIN_LEN && s.length <= MAX_LEN && FIRST_PERSON.test(s.trim())
      );
      return { turnIdx: idx, node: turn.node, sents };
    }).filter(t => t.sents.length > 0);

    if (sentencesByTurn.length < 2) return;

    const pairs = [];
    for (let i = 0; i < sentencesByTurn.length; i++) {
      for (let j = i + 1; j < sentencesByTurn.length; j++) {
        for (const s1 of sentencesByTurn[i].sents) {
          for (const s2 of sentencesByTurn[j].sents) {
            if (s1.slice(0, 30) === s2.slice(0, 30)) continue;
            pairs.push({
              premise: s1,
              hypothesis: s2,
              meta: {
                claimA: { node: sentencesByTurn[i].node, turnIdx: sentencesByTurn[i].turnIdx, raw: s1 },
                claimB: { node: sentencesByTurn[j].node, turnIdx: sentencesByTurn[j].turnIdx, raw: s2 },
              }
            });
          }
        }
      }
    }

    if (pairs.length === 0) return;
    const sample = pairs.length > 50 ? pairs.sort(() => Math.random() - 0.5).slice(0, 50) : pairs;
    const nliConflicts = await nliInfer(sample, 0.65);

    if (nliConflicts.length > 0) {
      const existingRaws = new Set(currentConflicts.map(c => c.claimA.raw + c.claimB.raw));
      const newConflicts = nliConflicts.filter(c => !existingRaws.has(c.claimA.raw + c.claimB.raw));

      if (newConflicts.length > 0) {
        currentConflicts = [...currentConflicts, ...newConflicts];
        previousConflictCount = currentConflicts.length;
        const totalClaims = Math.max(1, currentBeliefs.length + currentSuperseded.length + currentConflicts.length);
        const issueScore = currentConflicts.length + currentSuperseded.length * 0.5;
        currentCI = Math.max(0, 1.0 - issueScore / totalClaims);
        soundEngine.playConflict();
        updateHUD();
        applyHighlights();
        if (isDrawerOpen) updateDrawerUI();
      }
    }
  }

  // ==========================================
  // 11. Master Audit Loop
  // ==========================================
  function runAudit(manual = false) {
    const turns = getTurnNodes();
    if (turns.length === 0) {
      lastAuditNotice = "No conversation text detected on this page yet. Try scrolling the chat into view and click Run Live Audit.";
      if (manual && isDrawerOpen) updateDrawerUI();
      return;
    }

    const result = reconcileTurns(turns);
    currentBeliefs = result.active;
    currentSuperseded = result.superseded;
    currentConflicts = result.conflicts;
    currentCI = result.ci;

    // Acoustic truth feedback based on state delta
    if (manual) {
      soundEngine.playAudit();
    } else {
      if (currentConflicts.length > previousConflictCount) {
        soundEngine.playConflict();
      } else if (currentBeliefs.length > previousBeliefCount && currentConflicts.length === 0) {
        soundEngine.playVerify();
      }
    }
    previousConflictCount = currentConflicts.length;
    previousBeliefCount = currentBeliefs.length;

    updateHUD();
    applyHighlights();
    runSemanticNLI(turns);

    try {
      chrome.runtime.sendMessage({
        action: "update_stats",
        turns: 1,
        conflicts: currentConflicts.length,
        activeBeliefs: currentBeliefs.length
      }, () => {
        if (chrome.runtime.lastError) { /* ignore */ }
      });
    } catch (e) {}

    if (manual && isDrawerOpen) {
      updateDrawerUI();
    }
  }

  // ==========================================
  // 12. Mutation Observer & Theme Watcher
  // ==========================================
  let auditDebounce = null;
  const observer = new MutationObserver((mutations) => {
    const external = mutations.some(m => !m.target.closest || !m.target.closest("#alethex-root"));
    if (!external) return;

    const currentTurns = getTurnNodes();
    if (currentTurns.length !== lastMessageCount) {
      lastMessageCount = currentTurns.length;
      clearTimeout(auditDebounce);
      auditDebounce = setTimeout(() => {
        runAudit(false);
      }, 2500);
    }
  });

  // Watch for host theme switches (e.g., ChatGPT or Claude class toggle)
  const themeObserver = new MutationObserver(() => {
    syncTheme();
  });

  // ==========================================
  // 13. Chrome Runtime Message Listener
  // ==========================================
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ping") {
      sendResponse({
        ok: true,
        version: "1.2.0",
        platform: platform,
        beliefsCount: currentBeliefs.length,
        conflictsCount: currentConflicts.length,
        isDrawerOpen: isDrawerOpen
      });
      return false;
    } else if (request.action === "audit_now") {
      runAudit(true);
      if (!isDrawerOpen) toggleDrawer();
      sendResponse({ ok: true });
      return false;
    } else if (request.action === "open_drawer") {
      soundEngine.playClick();
      if (!isDrawerOpen) toggleDrawer();
      sendResponse({ ok: true });
      return false;
    } else if (request.action === "toggle_highlights") {
      isHighlightingEnabled = !!request.enabled;
      applyHighlights();
      if (isDrawerOpen) updateDrawerUI();
      sendResponse({ ok: true });
      return false;
    } else if (request.action === "toggle_sound") {
      soundEngine.enabled = !!request.enabled;
      renderDrawerSoundIcon();
      if (soundEngine.enabled) soundEngine.playVerify();
      sendResponse({ ok: true });
      return false;
    } else if (request.action === "set_theme") {
      userThemeSetting = request.theme || "auto";
      syncTheme();
      sendResponse({ ok: true });
      return false;
    }
  });

  // Listen for OS media query changes
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      syncTheme();
    });
  }

  // Boot after document idle
  setTimeout(() => {
    try {
      chrome.storage.local.get(["highlightConflicts", "userTheme", "soundEnabled"], (res) => {
        if (res.highlightConflicts !== undefined) isHighlightingEnabled = res.highlightConflicts;
        if (res.userTheme) userThemeSetting = res.userTheme;
        if (res.soundEnabled !== undefined) soundEngine.enabled = res.soundEnabled;
        renderDrawerSoundIcon();
        syncTheme();
      });
    } catch (e) {}

    initDOM();
    syncTheme();
    initNLIWorker();
    const turns = getTurnNodes();
    lastMessageCount = turns.length;
    runAudit(false);

    observer.observe(document.body, { childList: true, subtree: true });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme", "data-color-mode"] });
  }, 1000);

})();
