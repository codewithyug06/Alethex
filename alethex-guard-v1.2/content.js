/**
 * ALETHEX Universal Content Script (Manifest V3)
 * Performance-optimized, non-blocking, responsive memory reconciliation guard.
 *
 * Supported Platforms: ChatGPT, Claude, Gemini, Copilot, DeepSeek, Perplexity, Grok, Poe, Mistral, Local WebUIs.
 */

(function () {
  if (window.__ALETHEX_GUARD_INITIALIZED__) return;
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

  // 1. Inject Styles Once
  function injectStyles() {
    if (document.getElementById("alethex-styles")) return;
    const style = document.createElement("style");
    style.id = "alethex-styles";
    style.textContent = `
      #alethex-root {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 2147483647;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #f8fafc;
        pointer-events: none;
      }
      #alethex-hud {
        pointer-events: auto;
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(11, 15, 25, 0.88);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px -3px rgba(56, 189, 248, 0.15);
        border-radius: 9999px;
        padding: 7px 14px;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        user-select: none;
      }
      #alethex-hud:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 14px 30px -5px rgba(0, 0, 0, 0.7), 0 0 20px -2px rgba(56, 189, 248, 0.3);
      }
      .alethex-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        transition: background-color 0.3s;
      }
      .alethex-dot.amber { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
      .alethex-dot.red { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
      
      .alethex-hud-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
        color: #ffffff;
      }
      .alethex-hud-sub {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 500;
      }

      /* Inspector Drawer */
      #alethex-drawer {
        pointer-events: auto;
        position: absolute;
        bottom: 50px;
        right: 0;
        width: 380px;
        max-width: calc(100vw - 40px);
        max-height: 520px;
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.85);
        border-radius: 16px;
        display: none;
        flex-direction: column;
        overflow: hidden;
        animation: alethex-slide 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      }
      @keyframes alethex-slide {
        from { opacity: 0; transform: translateY(8px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
      }
      .alethex-header {
        padding: 14px 16px;
        background: #0f172a;
        border-bottom: 1px solid #1e293b;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .alethex-brand {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .alethex-icon {
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 13px;
      }
      .alethex-body {
        padding: 16px;
        overflow-y: auto;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 12px;
        font-size: 12px;
      }
      .alethex-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
      }
      .alethex-badge {
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 9999px;
        text-transform: uppercase;
      }
      .alethex-badge.green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
      .alethex-badge.amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
      .alethex-badge.red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

      .alethex-btn {
        width: 100%;
        background: linear-gradient(135deg, #0284c7, #2563eb);
        color: #ffffff;
        border: none;
        padding: 9px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
      }
      .alethex-btn:hover { opacity: 0.92; }
      
      .alethex-btn-sub {
        width: 100%;
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
        padding: 7px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
        margin-top: 6px;
      }
      .alethex-btn-sub:hover { background: #334155; }

      .alethex-highlight {
        outline: 2px solid #ef4444 !important;
        background: rgba(239, 68, 68, 0.08) !important;
        border-radius: 6px;
      }
    `;
    document.head.appendChild(style);
  }

  // 2. Initialize HUD Once
  function initDOM() {
    injectStyles();

    let root = document.getElementById("alethex-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "alethex-root";
      root.innerHTML = `
        <div id="alethex-hud">
          <span class="alethex-dot" id="alethex-dot"></span>
          <span class="alethex-hud-title">ALETHEX</span>
          <span style="color:#475569;">·</span>
          <span class="alethex-hud-sub" id="alethex-hud-status">Guarding</span>
        </div>
        <div id="alethex-drawer">
          <div class="alethex-header">
            <div class="alethex-brand">
              <div class="alethex-icon">α</div>
              <div>
                <div style="font-weight:700; font-size:13px; color:#ffffff;">ALETHEX Truth Guard</div>
                <div style="font-size:10px; color:#94a3b8;">${platform} Active</div>
              </div>
            </div>
            <button id="alethex-close" style="background:none; border:none; color:#94a3b8; font-size:18px; cursor:pointer;">&times;</button>
          </div>
          <div class="alethex-body" id="alethex-drawer-body">
            <!-- Dynamic Content -->
          </div>
        </div>
      `;
      document.body.appendChild(root);

      document.getElementById("alethex-hud").addEventListener("click", toggleDrawer);
      document.getElementById("alethex-close").addEventListener("click", () => {
        isDrawerOpen = false;
        document.getElementById("alethex-drawer").style.display = "none";
      });
    }
  }

  function toggleDrawer() {
    isDrawerOpen = !isDrawerOpen;
    const drawer = document.getElementById("alethex-drawer");
    if (drawer) {
      drawer.style.display = isDrawerOpen ? "flex" : "none";
      if (isDrawerOpen) updateDrawerUI();
    }
  }

  // 3. Fast Turn Scraper with zero document-level thrashing
  function getTurnNodes() {
    let selector = "";
    if (platform === "ChatGPT") selector = '[data-message-author-role], .whitespace-pre-wrap';
    else if (platform === "Claude.ai") selector = '.font-claude-message, .font-user-message, div.prose';
    else if (platform === "Google Gemini") selector = '.user-query-container, .response-container, message-content';
    else if (platform === "DeepSeek") selector = '.ds-markdown, div[class*="message"]';
    else if (platform === "Perplexity") selector = '.prose, div[class*="answer"]';
    else if (platform === "Copilot") selector = 'cib-message-group, .ac-textBlock';
    else selector = 'article, div[class*="message"], .chat-bubble';

    const rawNodes = document.querySelectorAll(selector);
    const valid = [];
    rawNodes.forEach((node) => {
      // Ignore our own HUD elements
      if (node.closest && node.closest("#alethex-root")) return;
      const text = node.innerText ? node.innerText.trim() : "";
      if (text.length > 15 && text.length < 5000) {
        valid.push({ node, text });
      }
    });
    return valid;
  }

  // 4. Client-side Temporal Belief Reconciliation (Fast & Non-blocking)
  function reconcileTurns(turns) {
    const claims = [];
    const entitySlots = {}; // slot: "predicate" -> array of claims
    const conflicts = [];
    const superseded = [];

    // Factual statement patterns
    const patterns = [
      { regex: /\b(?:i am|i'm|alice is|bob is|user is)\s+(?:a|an)?\s*([a-zA-Z\s]+)(?:at|in|with)\s+([a-zA-Z\s]+)/i, pred: "role_at" },
      { regex: /\b(?:works? at|employed at|joined)\s+([a-zA-Z0-9\s]+)/i, pred: "works_at" },
      { regex: /\b(?:lives? in|moved to|relocated to)\s+([a-zA-Z0-9\s]+)/i, pred: "lives_in" },
      { regex: /\b(?:uses?|prefer|migrated to)\s+([a-zA-Z0-9\s]+)/i, pred: "uses" },
      { regex: /\b(?:am|became|is)\s+(vegetarian|vegan|pescatarian|meat-eater)/i, pred: "diet" },
      { regex: /\b(?:ate|eating|ordered)\s+(steak|burger|chicken|pork|beef|fish|salad)/i, pred: "eats" }
    ];

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
            const val = match[1].trim();
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

    // Check temporal ordering per predicate
    Object.keys(entitySlots).forEach((pred) => {
      const slotClaims = entitySlots[pred];
      slotClaims.sort((a, b) => a.start - b.start);

      for (let i = 0; i < slotClaims.length; i++) {
        for (let j = i + 1; j < slotClaims.length; j++) {
          const c1 = slotClaims[i];
          const c2 = slotClaims[j];
          if (c1.value.toLowerCase() === c2.value.toLowerCase()) continue;

          if (c2.start.getTime() > c1.start.getTime()) {
            c1.end = c2.start;
            if (!superseded.includes(c1)) superseded.push(c1);
          } else {
            conflicts.push({ claimA: c1, claimB: c2, reason: `Contradicting '${c1.value}' vs '${c2.value}'` });
          }
        }
      }
    });

    const active = claims.filter((c) => !superseded.includes(c));
    const totalPairs = Math.max(1, (claims.length * (claims.length - 1)) / 2);
    const ci = conflicts.length > 0 ? Math.max(0, 1.0 - conflicts.length / totalPairs) : 1.0;

    return { active, superseded, conflicts, ci };
  }

  // 5. Update UI Components without full re-render
  function updateHUD() {
    const dot = document.getElementById("alethex-dot");
    const status = document.getElementById("alethex-hud-status");
    if (!dot || !status) return;

    if (currentConflicts.length > 0) {
      dot.className = "alethex-dot red";
      status.innerText = `${currentConflicts.length} Conflict Alert`;
    } else if (currentSuperseded.length > 0) {
      dot.className = "alethex-dot amber";
      status.innerText = `${currentSuperseded.length} Updated Fact`;
    } else {
      dot.className = "alethex-dot";
      status.innerText = "100% Consistent";
    }
  }

  function updateDrawerUI() {
    const body = document.getElementById("alethex-drawer-body");
    if (!body) return;

    let badgeClass = "green";
    let badgeText = "Healthy (1.00)";
    if (currentConflicts.length > 0) {
      badgeClass = "red";
      badgeText = `Conflict (${currentCI.toFixed(2)})`;
    } else if (currentSuperseded.length > 0) {
      badgeClass = "amber";
      badgeText = `Superseded (${currentCI.toFixed(2)})`;
    }

    const activeList = currentBeliefs.length > 0
      ? currentBeliefs.map(b => `<div style="padding:4px 0; border-bottom:1px solid #1e293b;"><strong style="color:#38bdf8;">${b.predicate}:</strong> ${b.value}</div>`).join("")
      : "<div style='color:#64748b;'>No structured facts in conversation yet.</div>";

    const conflictList = currentConflicts.length > 0
      ? currentConflicts.map(c => `
          <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); padding:8px; border-radius:6px; margin-bottom:6px;">
            <div style="color:#f87171; font-weight:700;">⚠️ ${c.reason}</div>
            <div style="font-size:11px; color:#cbd5e1; margin-top:2px;">"${c.claimA.raw}" vs "${c.claimB.raw}"</div>
          </div>
        `).join("")
      : "<div style='color:#34d399;'>✓ Zero logical or temporal contradictions.</div>";

    body.innerHTML = `
      <div class="alethex-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase;">Consistency Score</span>
          <span class="alethex-badge ${badgeClass}">${badgeText}</span>
        </div>
        <div style="font-size:11px; color:#94a3b8;">Continuous real-time verification prevents persistent memory corruption.</div>
      </div>

      <div class="alethex-card">
        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">Active Beliefs Frontier</div>
        <div>${activeList}</div>
      </div>

      <div class="alethex-card">
        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom:6px;">Contradiction Report</div>
        <div>${conflictList}</div>
      </div>

      <button id="alethex-audit-now-btn" class="alethex-btn">Audit Active Conversation</button>
      <button id="alethex-toggle-hl-btn" class="alethex-btn-sub">${isHighlightingEnabled ? "Hide In-Chat Highlights" : "Highlight Conflicts in Chat"}</button>
    `;

    body.querySelector("#alethex-audit-now-btn").addEventListener("click", () => runAudit(true));
    body.querySelector("#alethex-toggle-hl-btn").addEventListener("click", () => {
      isHighlightingEnabled = !isHighlightingEnabled;
      applyHighlights();
      updateDrawerUI();
    });
  }

  function applyHighlights() {
    document.querySelectorAll(".alethex-highlight").forEach(el => el.classList.remove("alethex-highlight"));
    if (!isHighlightingEnabled || currentConflicts.length === 0) return;

    currentConflicts.forEach(conf => {
      if (conf.claimA && conf.claimA.node) conf.claimA.node.classList.add("alethex-highlight");
      if (conf.claimB && conf.claimB.node) conf.claimB.node.classList.add("alethex-highlight");
    });
  }

  // 6. Master Non-Blocking Audit
  function runAudit(manual = false) {
    const turns = getTurnNodes();
    if (turns.length === 0) return;

    const result = reconcileTurns(turns);
    currentBeliefs = result.active;
    currentSuperseded = result.superseded;
    currentConflicts = result.conflicts;
    currentCI = result.ci;

    updateHUD();
    applyHighlights();

    // Update background extension storage stats
    try {
      chrome.runtime.sendMessage({
        action: "update_stats",
        turns: 1,
        conflicts: currentConflicts.length,
        activeBeliefs: currentBeliefs.length
      });
    } catch (e) {
      // Ignored if background is dormant
    }

    if (manual && isDrawerOpen) {
      updateDrawerUI();
    }
  }

  // 7. Controlled, Non-Recursive Observer
  let auditDebounce = null;
  const observer = new MutationObserver((mutations) => {
    // IGNORE mutations that originate inside our own HUD/Drawer!
    const external = mutations.some(m => !m.target.closest || !m.target.closest("#alethex-root"));
    if (!external) return;

    const currentTurns = getTurnNodes();
    if (currentTurns.length !== lastMessageCount) {
      lastMessageCount = currentTurns.length;
      clearTimeout(auditDebounce);
      // Wait for streaming text to settle before running audit
      auditDebounce = setTimeout(() => {
        runAudit(false);
      }, 2500);
    }
  });

  // 8. Message Listener for Popup communication
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "audit_now") {
      runAudit(true);
      if (!isDrawerOpen) toggleDrawer();
      sendResponse({ ok: true });
    }
  });

  // Boot after document idle
  setTimeout(() => {
    initDOM();
    const turns = getTurnNodes();
    lastMessageCount = turns.length;
    runAudit(false);
    observer.observe(document.body, { childList: true, subtree: true });
  }, 1200);

})();
