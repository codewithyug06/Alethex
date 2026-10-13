/**
 * Popup Script for ALETHEX Universal Chrome Extension (v1.2.0)
 * Dual Light/Dark Theme Support, Acoustic Truth Cues & Resilient Content Script Messaging.
 */

// ==========================================
// Web Audio Synthesis Engine for ALETHEX
// ==========================================
class AlethexAudioEngine {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.lastPlayTime = 0;
    this.minInterval = 300;
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

      // Chime: D5 (587.33 Hz) -> A5 (880.00 Hz)
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

/**
 * Resilient Tab Messaging Wrapper
 * Always consumes chrome.runtime.lastError to eliminate "Unchecked runtime.lastError" warnings.
 * Gracefully attempts dynamic script injection when on an active chat tab without a connected content script.
 */
function sendTabMessage(message, callback) {
  try {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const queryErr = chrome.runtime.lastError;
      if (queryErr || !tabs || !tabs[0] || !tabs[0].id) {
        if (typeof callback === "function") {
          callback(null, queryErr || new Error("No active tab found"));
        }
        return;
      }

      const activeTab = tabs[0];
      const tabUrl = (activeTab.url || "").toLowerCase();

      // Check for restricted internal browser URLs
      const isRestricted =
        tabUrl.startsWith("chrome://") ||
        tabUrl.startsWith("edge://") ||
        tabUrl.startsWith("about:") ||
        tabUrl.startsWith("chrome-extension://") ||
        tabUrl.startsWith("devtools://") ||
        tabUrl.startsWith("view-source:") ||
        tabUrl.startsWith("https://chrome.google.com/webstore") ||
        tabUrl.startsWith("https://chromewebstore.google.com");

      if (isRestricted) {
        if (typeof callback === "function") {
          callback(null, new Error("Restricted browser page"));
        }
        return;
      }

      chrome.tabs.sendMessage(activeTab.id, message, (response) => {
        // ALWAYS read lastError to prevent Chrome from logging unchecked error
        const sendErr = chrome.runtime.lastError;

        if (sendErr) {
          const errMsg = sendErr.message || "";
          if (
            (errMsg.includes("Receiving end does not exist") ||
             errMsg.includes("Could not establish connection")) &&
            chrome.scripting &&
            typeof chrome.scripting.executeScript === "function"
          ) {
            try {
              chrome.scripting.executeScript(
                {
                  target: { tabId: activeTab.id },
                  files: ["content.js"]
                },
                () => {
                  const injectErr = chrome.runtime.lastError;
                  if (!injectErr) {
                    setTimeout(() => {
                      chrome.tabs.sendMessage(activeTab.id, message, (retryRes) => {
                        const retryErr = chrome.runtime.lastError;
                        if (typeof callback === "function") {
                          callback(retryRes, retryErr || null);
                        }
                      });
                    }, 180);
                  } else {
                    if (typeof callback === "function") {
                      callback(null, sendErr);
                    }
                  }
                }
              );
              return;
            } catch (injectEx) {
              if (typeof callback === "function") {
                callback(null, sendErr);
              }
              return;
            }
          }

          if (typeof callback === "function") {
            callback(null, sendErr);
          }
          return;
        }

        if (typeof callback === "function") {
          callback(response, null);
        }
      });
    });
  } catch (ex) {
    if (typeof callback === "function") {
      callback(null, ex);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const siteEl = document.getElementById("site-name");
  const statusTag = document.getElementById("status-tag");
  const dot = document.getElementById("dot-indicator");
  const themeBtn = document.getElementById("theme-toggle-btn");
  const themeIcon = document.getElementById("theme-icon");
  const auditBtn = document.getElementById("audit-btn");
  const openDrawerLink = document.getElementById("open-drawer-link");
  const toggleHighlight = document.getElementById("toggle-highlight");
  const toggleSound = document.getElementById("toggle-sound");

  const audio = new AlethexAudioEngine();

  // ==========================================
  // 1. Theme Management (Auto / Light / Dark)
  // ==========================================
  let currentThemeMode = "auto";

  function renderThemeIcon(effectiveTheme) {
    if (effectiveTheme === "light") {
      themeIcon.innerHTML = `
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      `;
      themeBtn.title = "Current: Light Theme (Click to switch to Dark)";
    } else {
      themeIcon.innerHTML = `
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      `;
      themeBtn.title = "Current: Dark Theme (Click to switch to Light)";
    }
  }

  function applyTheme(theme) {
    let effective = theme;
    if (theme === "auto") {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      effective = prefersDark ? "dark" : "light";
    }
    document.documentElement.setAttribute("data-theme", effective);
    renderThemeIcon(effective);
  }

  // Load saved theme
  chrome.storage.local.get(["userTheme"], (data) => {
    if (data && data.userTheme) {
      currentThemeMode = data.userTheme;
    } else {
      currentThemeMode = "auto";
    }
    applyTheme(currentThemeMode);
  });

  // Cycle Theme: auto -> light -> dark -> auto
  themeBtn.addEventListener("click", () => {
    audio.playClick();
    if (currentThemeMode === "auto") {
      currentThemeMode = "light";
    } else if (currentThemeMode === "light") {
      currentThemeMode = "dark";
    } else {
      currentThemeMode = "auto";
    }
    chrome.storage.local.set({ userTheme: currentThemeMode });
    applyTheme(currentThemeMode);

    // Notify active tab to synchronize theme
    sendTabMessage({
      action: "set_theme",
      theme: currentThemeMode
    }, () => {});
  });

  // Listen for system theme changes when in auto mode
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (currentThemeMode === "auto") {
      applyTheme("auto");
    }
  });

  // ==========================================
  // 2. Active Platform Detection & Connection
  // ==========================================
  function updatePlatformUI(url, isConnected) {
    url = (url || "").toLowerCase();
    let platformName = "Universal AI Guard";
    let isAIPage = false;

    if (url.includes("chatgpt") || url.includes("openai")) {
      platformName = "ChatGPT";
      isAIPage = true;
    } else if (url.includes("claude.ai")) {
      platformName = "Claude.ai";
      isAIPage = true;
    } else if (url.includes("gemini.google")) {
      platformName = "Google Gemini";
      isAIPage = true;
    } else if (url.includes("copilot.microsoft")) {
      platformName = "Copilot";
      isAIPage = true;
    } else if (url.includes("deepseek")) {
      platformName = "DeepSeek";
      isAIPage = true;
    } else if (url.includes("perplexity")) {
      platformName = "Perplexity";
      isAIPage = true;
    } else if (url.includes("grok") || url.includes("x.com")) {
      platformName = "Grok";
      isAIPage = true;
    } else if (url.includes("poe.com")) {
      platformName = "Poe";
      isAIPage = true;
    } else if (url.includes("mistral")) {
      platformName = "Mistral";
      isAIPage = true;
    } else if (url.includes("localhost") || url.includes("127.0.0.1")) {
      platformName = "Local AI WebUI";
      isAIPage = true;
    } else if (
      url.startsWith("chrome://") ||
      url.startsWith("edge://") ||
      url.startsWith("about:") ||
      url.startsWith("chrome-extension://")
    ) {
      siteEl.innerText = "AI Platform Standby";
      statusTag.innerText = "IDLE";
      statusTag.className = "status-tag idle";
      dot.className = "dot-pulse idle";
      return;
    }

    if (isConnected) {
      siteEl.innerText = `${platformName} Active`;
      statusTag.innerText = "ACTIVE";
      statusTag.className = "status-tag";
      dot.className = "dot-pulse";
    } else if (isAIPage) {
      siteEl.innerText = `${platformName} (Standby)`;
      statusTag.innerText = "STANDBY";
      statusTag.className = "status-tag standby";
      dot.className = "dot-pulse standby";
    } else {
      siteEl.innerText = platformName;
      statusTag.innerText = "STANDBY";
      statusTag.className = "status-tag standby";
      dot.className = "dot-pulse standby";
    }
  }

  // Initial detection & live ping
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const qErr = chrome.runtime.lastError;
    if (qErr || !tabs || !tabs[0]) {
      updatePlatformUI("", false);
      return;
    }

    const currentUrl = tabs[0].url || "";
    updatePlatformUI(currentUrl, false);

    sendTabMessage({ action: "ping" }, (response, pingErr) => {
      if (!pingErr && response && response.ok) {
        updatePlatformUI(currentUrl, true);
        if (response.beliefsCount !== undefined && response.conflictsCount !== undefined) {
          document.getElementById("stat-beliefs").innerText = response.beliefsCount;
          const confEl = document.getElementById("stat-conflicts");
          confEl.innerText = response.conflictsCount;
          confEl.style.color = response.conflictsCount > 0 ? "var(--danger-text)" : "var(--success-text)";
          updateConsistencyGauge(response.beliefsCount, response.conflictsCount);
        }
      } else {
        updatePlatformUI(currentUrl, false);
      }
    });
  });

  // ==========================================
  // 3. Stats & Consistency Gauge
  // ==========================================
  function updateConsistencyGauge(beliefs, conflicts) {
    const ciPercentEl = document.getElementById("ci-percentage");
    const ciBarEl = document.getElementById("ci-bar");

    if (beliefs === 0 && conflicts === 0) {
      ciPercentEl.innerText = "100%";
      ciPercentEl.style.color = "var(--success-text)";
      ciBarEl.style.width = "100%";
      ciBarEl.style.background = "var(--success)";
      return;
    }

    const total = beliefs + conflicts;
    const ratio = Math.max(0, 1 - (conflicts / total));
    const percent = Math.round(ratio * 100);

    ciPercentEl.innerText = `${percent}%`;
    ciBarEl.style.width = `${percent}%`;

    if (percent >= 90) {
      ciPercentEl.style.color = "var(--success-text)";
      ciBarEl.style.background = "var(--success)";
    } else if (percent >= 60) {
      ciPercentEl.style.color = "var(--warning-text)";
      ciBarEl.style.background = "var(--warning)";
    } else {
      ciPercentEl.style.color = "var(--danger-text)";
      ciBarEl.style.background = "var(--danger)";
    }
  }

  chrome.storage.local.get(
    ["totalContradictionsBlocked", "activeBeliefsCount", "highlightConflicts", "soundEnabled"],
    (data) => {
      const beliefs = (data && data.activeBeliefsCount) || 0;
      const conflicts = (data && data.totalContradictionsBlocked) || 0;

      document.getElementById("stat-beliefs").innerText = beliefs;
      const confEl = document.getElementById("stat-conflicts");
      confEl.innerText = conflicts;
      confEl.style.color = conflicts > 0 ? "var(--danger-text)" : "var(--success-text)";

      updateConsistencyGauge(beliefs, conflicts);

      if (data && data.highlightConflicts !== undefined) {
        toggleHighlight.checked = data.highlightConflicts;
      }

      const isSoundOn = !data || data.soundEnabled !== false;
      if (toggleSound) {
        toggleSound.checked = isSoundOn;
      }
      audio.enabled = isSoundOn;
    }
  );

  // ==========================================
  // 4. Toggle In-Chat Highlighting
  // ==========================================
  toggleHighlight.addEventListener("change", (e) => {
    const isChecked = e.target.checked;
    audio.playClick();
    chrome.storage.local.set({ highlightConflicts: isChecked });

    sendTabMessage({
      action: "toggle_highlights",
      enabled: isChecked
    }, () => {});
  });

  // ==========================================
  // 5. Toggle Acoustic Truth Cues
  // ==========================================
  if (toggleSound) {
    toggleSound.addEventListener("change", (e) => {
      const isChecked = e.target.checked;
      audio.enabled = isChecked;
      chrome.storage.local.set({ soundEnabled: isChecked });

      if (isChecked) {
        audio.playVerify();
      }

      sendTabMessage({
        action: "toggle_sound",
        enabled: isChecked
      }, () => {});
    });
  }

  // ==========================================
  // 6. Audit Button
  // ==========================================
  let isAuditing = false;

  function resetAuditBtn() {
    auditBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
        <path d="M21 3v5h-5"/>
      </svg>
      <span>Audit Conversation Consistency</span>
    `;
    auditBtn.style.opacity = "1";
    auditBtn.disabled = false;
    isAuditing = false;
  }

  auditBtn.addEventListener("click", () => {
    if (isAuditing) return;
    isAuditing = true;
    audio.playAudit();

    auditBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="animation: alethex-spin 1s linear infinite;">
        <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
        <path d="M21 3v5h-5"/>
      </svg>
      <span>Scanning Conversation...</span>
    `;
    auditBtn.style.opacity = "0.85";
    auditBtn.disabled = true;

    sendTabMessage({ action: "audit_now" }, (response, err) => {
      if (!err && response && response.ok) {
        setTimeout(() => {
          resetAuditBtn();
        }, 700);
      } else {
        // Tab not connected or restricted
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          const qErr = chrome.runtime.lastError;
          const url = (!qErr && tabs && tabs[0] && tabs[0].url) ? tabs[0].url.toLowerCase() : "";
          const isRestricted =
            url.startsWith("chrome://") ||
            url.startsWith("edge://") ||
            url.startsWith("about:");

          if (isRestricted) {
            auditBtn.innerHTML = `<span>Active on AI chat pages</span>`;
          } else {
            auditBtn.innerHTML = `<span>Refresh tab to activate guard</span>`;
          }

          setTimeout(() => {
            resetAuditBtn();
          }, 2200);
        });
      }
    });
  });

  // ==========================================
  // 7. Open Inspector Drawer Link
  // ==========================================
  openDrawerLink.addEventListener("click", (e) => {
    e.preventDefault();
    audio.playClick();
    sendTabMessage({ action: "open_drawer" }, (response, err) => {
      if (!err && response && response.ok) {
        window.close(); // Close popup so user interacts with in-page drawer
      } else {
        const originalText = openDrawerLink.innerText;
        openDrawerLink.innerText = "Available on AI chat pages";
        setTimeout(() => {
          openDrawerLink.innerHTML = "Open Inspector Drawer &rarr;";
        }, 2200);
      }
    });
  });
});
