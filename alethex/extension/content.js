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
          console.log("[ALETHEX NLI]", message);
          if (message && message.includes("ready")) nliWorkerReady = true;
          return;
        }
        if (type === "error") { console.warn("[ALETHEX NLI] Error:", message); return; }
        if (type === "pong") { return; }
        if (id && nliCallbacks[id]) {
          nliCallbacks[id].resolve(results || []);
          delete nliCallbacks[id];
        }
      };
      nliWorker.onerror = (e) => console.warn("[ALETHEX NLI] Worker error:", e.message);
      // Preload the model immediately so first audit doesn't wait for download
      nliWorker.postMessage({ id: 0, type: "ping" });
    } catch (e) {
      console.warn("[ALETHEX NLI] Worker unavailable:", e.message);
    }
  }

  function nliInfer(pairs, threshold) {
    return new Promise((resolve) => {
      if (!nliWorker || pairs.length === 0) { resolve([]); return; }
      const id = ++nliCallId;
      nliCallbacks[id] = { resolve };
      nliWorker.postMessage({ id, type: "nli_batch", data: { pairs, threshold } });
      // Timeout after 30s — fallback to regex-only results
      setTimeout(() => {
        if (nliCallbacks[id]) { delete nliCallbacks[id]; resolve([]); }
      }, 30000);
    });
  }

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
      .alethex-highlight-amber {
        outline: 2px dashed #f59e0b !important;
        background: rgba(245, 158, 11, 0.06) !important;
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
  //
  // Platform-specific selectors are guesses at each site's current DOM structure, which these
  // sites change often -- if a selector goes stale it silently finds zero turns, and runAudit()
  // would then no-op with no feedback at all ("Rescan" appearing to do nothing). To avoid that,
  // always OR in a broad generic fallback alongside the platform-specific selector instead of
  // only using it as a last resort, and de-duplicate matched nodes by reference.
  const GENERIC_SELECTOR = 'article, [class*="message"], [data-message-author-role], [class*="turn"], .prose, p';

  function getTurnNodes() {
    // Scan all turns (both user and AI). False positives from AI text are prevented downstream
    // by the article-start filter (rejects values starting with "the", "a", "an", "it", etc.)
    // and the 4-word length cap, which between them block advice/example phrases while passing
    // real personal facts like "Google", "Python", "London".
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

  // 4. Client-side Temporal Belief Reconciliation (Fast & Non-blocking)
  function reconcileTurns(turns) {
    const claims = [];
    const entitySlots = {}; // slot: "predicate" -> array of claims
    const conflicts = [];
    const superseded = [];

    // Factual statement patterns. Broadened from the original 6 toy patterns (which only matched
    // phrasing like "alice is"/"bob is" and almost never fired on real ChatGPT/Claude conversations)
    // to cover how people actually phrase these facts in first person.
    // NOTE: dash in character classes must be at start/end or escaped as \x2D — never \\- inside
    // a regex literal (\\- is parsed as a range from backslash char 92 to next char, causing
    // "range out of order" crashes). All captures use [-a-z0-9 ...] with dash first.
    const SUBJ = "(?:i am|i'm|i|my name is|i'll be|i am now|i'm now)";
    const patterns = [
      // Employment — must start with subject pronoun immediately, no long gaps
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,10}(?:now\\s+)?(?:work(?:ing)?\\s+(?:at|for)|employed (?:at|by)|joined|started (?:at|working at))\\s+([a-z0-9&. -]{2,40})`, "i"), pred: "works_at" },
      { regex: /\bmy (?:company|employer|workplace|office) is\s+([-a-z0-9&. ]{2,40})/i, pred: "works_at" },
      // Location
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,10}(?:now\\s+)?(?:live(?:s|d)?\\s+in|based in|moved to|relocated to|living in|staying in|currently in|reside in|residing in|am from|grew up in)\\s+([-a-z0-9,. ]{2,40})`, "i"), pred: "lives_in" },
      { regex: /\bmy (?:city|town|country|location|home(?:town)?) is\s+([-a-z0-9,. ]{2,40})/i, pred: "lives_in" },
      // Technology / tools — restricted to short values (tool names are rarely more than 3 words)
      { regex: new RegExp(`\\b(?:${SUBJ})\\s{1,5}(?:now\\s+)?(?:use|uses|using|prefer|prefers|switched to|migrated to)\\s+([-a-z0-9#+. ]{2,30})`, "i"), pred: "uses" },
      { regex: /\bmy (?:main |primary |daily )?(?:language|stack|framework|editor|ide|os|browser|phone|laptop|computer) is\s+([-a-z0-9#+. ]{2,30})/i, pred: "uses" },
      // Identity
      { regex: /\bmy name is\s+([a-z][-a-z'. ]{1,25})/i, pred: "name" },
      { regex: /\bpeople call me\s+([a-z][-a-z'. ]{1,20})/i, pred: "name" },
      { regex: /\bi(?:'m| am)\s+(\d{1,3})\s*(?:years old|yo|years of age)?\b/i, pred: "age" },
      { regex: /\bmy age is\s+(\d{1,3})\b/i, pred: "age" },
      // Diet — only closed-vocabulary values to avoid false positives
      { regex: /\b(?:i am|i'm|i've become|i became|i went)\s+(?:a |an )?(vegetarian|vegan|pescatarian|meat eater|omnivore|carnivore|keto|paleo)\b/i, pred: "diet" },
      { regex: /\bi\s+(?:don'?t|do not|no longer)\s+(?:eat meat|eat animal|consume meat)\b/i, pred: "diet", forcedValue: "vegetarian" },
      { regex: /\bi\s+(?:eat meat|am back to eating meat)\b/i, pred: "diet", forcedValue: "omnivore" },
      // Preferences — short objects only (2+ chars, max 4 words)
      { regex: /\bi\s+(?:like|love|enjoy|adore|am a fan of)\s+([-a-z0-9 ]{2,25})/i, pred: "likes" },
      { regex: /\bi\s+(?:dislike|hate|despise|don'?t like|do not like|can'?t stand|no longer like)\s+([-a-z0-9 ]{2,25})/i, pred: "dislikes" },
      // Role / job title — closed list of role suffixes to avoid matching arbitrary sentences
      { regex: /\bmy (?:job|role|title|position|profession|occupation) is\s+(?:a |an )?([-a-z0-9 ]{2,30})/i, pred: "role" },
      { regex: /\bi work as (?:a |an )?([-a-z0-9 ]{2,25})/i, pred: "role" },
      { regex: /\bi(?:'m| am) (?:a |an )?([-a-z0-9 ]{2,30}(?:developer|engineer|designer|manager|analyst|scientist|researcher|student|teacher|writer|founder|cto|ceo|coo|cfo))\b/i, pred: "role" },
      // Relationship status — closed vocabulary only
      { regex: /\bi(?:'m| am)\s+(single|married|engaged|divorced|in a relationship)\b/i, pred: "relationship" },
      // Language spoken
      { regex: /\bmy (?:native |first |primary )?language is\s+([-a-z ]{2,20})/i, pred: "language" },
      { regex: /\bi(?:'m| am) (?:a )?(?:native |fluent )?([a-z]{3,15}) speaker\b/i, pred: "language" },
    ];

    // Trailing-clause words that shouldn't be part of an extracted value (keeps "Google" and
    // "Google as a backend engineer" comparable instead of treated as unrelated strings).
    const TRIM_AT = /\b(?:and|but|so|because|which|who|that|while|although|as a|as an)\b.*$/i;

    function normalizeValue(raw) {
      let v = raw.replace(TRIM_AT, "").trim();
      v = v.replace(/[.,!?]+$/, "").trim();
      return v.toLowerCase();
    }

    // Loose equality: exact match, or one normalized value contains the other (handles
    // "google" vs "google cloud" style partial overlaps instead of false mismatches).
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
            if (!val) return;
            // Reject sentence fragments: more than 4 words → almost certainly not a named value.
            if (val.split(/\s+/).length > 4) return;
            // Reject values starting with articles/pronouns — real fact values (tool names,
            // places, companies) never start with "the", "a", "an", "it", "this", "that".
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

    // Opposing-predicate pairs: holding both simultaneously is a direct contradiction
    // regardless of when each was said (e.g. "I like pizza" vs "I dislike pizza").
    const OPPOSITES = { likes: "dislikes", dislikes: "likes" };

    // Check temporal ordering per predicate.
    // Two claims with different values for the same predicate:
    //   - Different turns: later one supersedes the earlier one (user updated their info).
    //   - Same turn: direct contradiction in one message.
    // Opposite-predicate pairs (likes/dislikes) are always conflicts regardless of turn order.
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
            conflicts.push({ claimA: c1, claimB: c2, reason: `Contradicting '${c1.value}' vs '${c2.value}'` });
          }
        }
      }

      const oppositePred = OPPOSITES[pred];
      if (oppositePred && entitySlots[oppositePred]) {
        slotClaims.forEach((c1) => {
          entitySlots[oppositePred].forEach((c2) => {
            if (valuesMatch(c1.value, c2.value)) {
              conflicts.push({ claimA: c1, claimB: c2, reason: `Opposing feelings about '${c1.value}'` });
            }
          });
        });
      }
    });

    const active = claims.filter((c) => !superseded.includes(c));
    // CI: conflicts are hard inconsistencies (weight 1.0), superseded are soft (weight 0.5 — user updated their info).
    // Returns null when no claims were found at all (conversation has no detectable factual statements).
    const totalClaims = Math.max(1, claims.length);
    const issueScore = conflicts.length + superseded.length * 0.5;
    const ci = claims.length === 0 ? null : Math.max(0, 1.0 - issueScore / totalClaims);

    return { active, superseded, conflicts, ci };
  }

  // 5. Update UI Components without full re-render
  function updateHUD() {
    const dot = document.getElementById("alethex-dot");
    const status = document.getElementById("alethex-hud-status");
    if (!dot || !status) return;

    if (currentConflicts.length > 0) {
      dot.className = "alethex-dot red";
      status.innerText = `${currentConflicts.length} Conflict${currentConflicts.length > 1 ? "s" : ""} Detected`;
    } else if (currentSuperseded.length > 0) {
      dot.className = "alethex-dot amber";
      status.innerText = `${currentSuperseded.length} Fact${currentSuperseded.length > 1 ? "s" : ""} Updated`;
    } else if (currentCI === null) {
      dot.className = "alethex-dot";
      status.innerText = "No Facts Found — Scan Chat";
    } else {
      dot.className = "alethex-dot";
      status.innerText = "Consistent ✓";
    }
  }

  // Claim text (predicate/value/raw sentence) originates from the chat page's own content,
  // which is untrusted -- a message literally containing HTML-like text must not be able to
  // inject markup into our drawer when interpolated into innerHTML below.
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function updateDrawerUI() {
    const body = document.getElementById("alethex-drawer-body");
    if (!body) return;

    let badgeClass = "green";
    let badgeText = currentCI === null ? "No Facts Yet" : `Healthy (${currentCI.toFixed(2)})`;
    if (currentCI === null) badgeClass = "amber";
    if (currentConflicts.length > 0) {
      badgeClass = "red";
      badgeText = `Conflict (${currentCI !== null ? currentCI.toFixed(2) : "?"})`;
    } else if (currentSuperseded.length > 0) {
      badgeClass = "amber";
      badgeText = `Updated Facts (${currentCI !== null ? currentCI.toFixed(2) : "?"})`;
    }

    const activeList = currentBeliefs.length > 0
      ? currentBeliefs.map(b => `<div style="padding:4px 0; border-bottom:1px solid #1e293b;"><strong style="color:#38bdf8;">${escapeHtml(b.predicate)}:</strong> ${escapeHtml(b.value)}</div>`).join("")
      : "<div style='color:#64748b;'>No structured facts in conversation yet.</div>";

    const conflictList = currentConflicts.length > 0
      ? currentConflicts.map(c => `
          <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); padding:8px; border-radius:6px; margin-bottom:6px;">
            <div style="color:#f87171; font-weight:700;">⚠️ ${escapeHtml(c.reason)}</div>
            <div style="font-size:11px; color:#cbd5e1; margin-top:2px;">"${escapeHtml(c.claimA.raw)}" vs "${escapeHtml(c.claimB.raw)}"</div>
          </div>
        `).join("")
      : "<div style='color:#34d399;'>✓ Zero logical or temporal contradictions.</div>";

    const noticeHtml = lastAuditNotice
      ? `<div class="alethex-card" style="color:#fbbf24;">${lastAuditNotice}</div>`
      : "";
    lastAuditNotice = null;

    body.innerHTML = `
      ${noticeHtml}
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

    body.querySelector("#alethex-audit-now-btn").addEventListener("click", (e) => {
      // Give immediate visible feedback on click so the button never *looks* like it did
      // nothing, even in the rare case where the audit finds no changes.
      const btn = e.currentTarget;
      btn.innerText = "Scanning...";
      btn.disabled = true;
      setTimeout(() => runAudit(true), 50);
    });
    body.querySelector("#alethex-toggle-hl-btn").addEventListener("click", () => {
      isHighlightingEnabled = !isHighlightingEnabled;
      applyHighlights();
      updateDrawerUI();
      // Keep the popup's switch (which reads chrome.storage) in sync with this in-page toggle.
      try {
        chrome.storage.local.set({ highlightConflicts: isHighlightingEnabled });
      } catch (e) {
        // chrome.storage unavailable; in-page state still updated above.
      }
    });
  }

  function applyHighlights() {
    document.querySelectorAll(".alethex-highlight").forEach(el => el.classList.remove("alethex-highlight"));
    document.querySelectorAll(".alethex-highlight-amber").forEach(el => el.classList.remove("alethex-highlight-amber"));
    if (!isHighlightingEnabled) return;

    // Previously only conflicts were ever outlined, but most real differences get classified as
    // "superseded" (a fact that changed over time) rather than "conflict" (held at the same time) --
    // see reconcileTurns()'s temporal-ordering logic. That meant the toggle usually had nothing to
    // hide/show and looked broken. Now both are highlighted (red = conflict, amber = superseded).
    currentConflicts.forEach(conf => {
      if (conf.claimA && conf.claimA.node) conf.claimA.node.classList.add("alethex-highlight");
      if (conf.claimB && conf.claimB.node) conf.claimB.node.classList.add("alethex-highlight");
    });
    currentSuperseded.forEach(c => {
      if (c.node) c.node.classList.add("alethex-highlight-amber");
    });
  }

  // 6. Semantic NLI Pass (runs in background after regex pass)
  // Extracts first-person sentences from all turns and runs cross-turn NLI inference.
  // Only sentences that start with "I " or "My " are candidates — this avoids processing
  // AI-generated prose while still catching user's self-description across turns.
  async function runSemanticNLI(turns) {
    if (!nliWorker) return; // Worker not initialized or unavailable

    const FIRST_PERSON = /^(?:i |my |i'm |i've |i am |i was |i will |i do |i don't |i work|i live|i use)/i;
    const MIN_LEN = 15;
    const MAX_LEN = 200;

    // Collect first-person sentences per turn
    const sentencesByTurn = turns.map((turn, idx) => {
      const sents = turn.text.split(/(?<=[.!?])\s+/).filter(s =>
        s.length >= MIN_LEN && s.length <= MAX_LEN && FIRST_PERSON.test(s.trim())
      );
      return { turnIdx: idx, node: turn.node, sents };
    }).filter(t => t.sents.length > 0);

    if (sentencesByTurn.length < 2) return; // Need at least 2 turns with first-person text

    // Build cross-turn pairs: all sentences from turn i vs all sentences from turn j (j > i)
    const pairs = [];
    for (let i = 0; i < sentencesByTurn.length; i++) {
      for (let j = i + 1; j < sentencesByTurn.length; j++) {
        for (const s1 of sentencesByTurn[i].sents) {
          for (const s2 of sentencesByTurn[j].sents) {
            // Skip pairs that are almost identical
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
    // Cap at 50 pairs to avoid very long inference times on long conversations
    const sample = pairs.length > 50 ? pairs.sort(() => Math.random() - 0.5).slice(0, 50) : pairs;

    const nliConflicts = await nliInfer(sample, 0.65);

    if (nliConflicts.length > 0) {
      // Merge NLI conflicts into current results, deduplicating against regex-found conflicts
      const existingRaws = new Set(currentConflicts.map(c => c.claimA.raw + c.claimB.raw));
      const newConflicts = nliConflicts.filter(c => !existingRaws.has(c.claimA.raw + c.claimB.raw));

      if (newConflicts.length > 0) {
        currentConflicts = [...currentConflicts, ...newConflicts];
        // Recompute CI
        const totalClaims = Math.max(1, currentBeliefs.length + currentSuperseded.length + currentConflicts.length);
        const issueScore = currentConflicts.length + currentSuperseded.length * 0.5;
        currentCI = Math.max(0, 1.0 - issueScore / totalClaims);
        updateHUD();
        applyHighlights();
        if (isDrawerOpen) updateDrawerUI();
      }
    }
  }

  // 7. Master Non-Blocking Audit
  let lastAuditNotice = null;

  function runAudit(manual = false) {
    const turns = getTurnNodes();
    if (turns.length === 0) {
      // Previously silently returned here with zero UI feedback, so a manual "Rescan" click
      // looked completely broken whenever selectors failed to match the page's current DOM.
      // Route through updateDrawerUI() (instead of a one-off DOM patch) so the Scanning...
      // button state set by the click handler always gets reset.
      lastAuditNotice = "No conversation text detected on this page yet. Try scrolling the chat into view, then Rescan again.";
      if (manual && isDrawerOpen) updateDrawerUI();
      return;
    }

    const result = reconcileTurns(turns);
    currentBeliefs = result.active;
    currentSuperseded = result.superseded;
    currentConflicts = result.conflicts;
    currentCI = result.ci;

    updateHUD();
    applyHighlights();

    // Semantic NLI pass — runs in background, updates UI when done.
    // Extract first-person sentences from all turns, build candidate pairs
    // across different turns, and run them through the NLI model.
    runSemanticNLI(turns);

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
    } else if (request.action === "toggle_highlights") {
      // Previously unhandled: the popup's "Highlight Drift in Chat" switch sent this
      // message but nothing here listened for it, so the switch silently did nothing.
      isHighlightingEnabled = !!request.enabled;
      applyHighlights();
      if (isDrawerOpen) updateDrawerUI();
      sendResponse({ ok: true });
    }
  });

  // Boot after document idle
  setTimeout(() => {
    // Respect the popup's persisted preference instead of always defaulting to enabled.
    try {
      chrome.storage.local.get(["highlightConflicts"], (res) => {
        if (res.highlightConflicts !== undefined) isHighlightingEnabled = res.highlightConflicts;
      });
    } catch (e) {
      // chrome.storage unavailable (e.g. extension context invalidated); keep default.
    }

    initDOM();
    initNLIWorker(); // Start model preload immediately so first audit doesn't stall
    const turns = getTurnNodes();
    lastMessageCount = turns.length;
    runAudit(false);
    observer.observe(document.body, { childList: true, subtree: true });
  }, 1200);

})();
