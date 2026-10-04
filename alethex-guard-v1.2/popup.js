/**
 * Popup Script for ALETHEX Universal Chrome Extension (v1.2.0)
 * Dual Light/Dark Theme Support & Resilient Content Script Messaging.
 */

document.addEventListener("DOMContentLoaded", () => {
  const siteEl = document.getElementById("site-name");
  const statusTag = document.getElementById("status-tag");
  const dot = document.getElementById("dot-indicator");
  const themeBtn = document.getElementById("theme-toggle-btn");
  const themeIcon = document.getElementById("theme-icon");
  const auditBtn = document.getElementById("audit-btn");
  const openDrawerLink = document.getElementById("open-drawer-link");
  const toggleHighlight = document.getElementById("toggle-highlight");

  // ==========================================
  // 1. Theme Management (Auto / Light / Dark)
  // ==========================================
  let currentThemeMode = "auto"; // "auto" | "light" | "dark"

  function renderThemeIcon(effectiveTheme) {
    if (effectiveTheme === "light") {
      // Sun Icon
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
      // Moon Icon
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
    if (data.userTheme) {
      currentThemeMode = data.userTheme;
    } else {
      currentThemeMode = "auto";
    }
    applyTheme(currentThemeMode);
  });

  // Cycle Theme: auto -> light -> dark -> auto
  themeBtn.addEventListener("click", () => {
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
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "set_theme",
          theme: currentThemeMode
        }, () => {
          if (chrome.runtime.lastError) { /* ignore tab error */ }
        });
      }
    });
  });

  // Listen for system theme changes when in auto mode
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (currentThemeMode === "auto") {
      applyTheme("auto");
    }
  });

  // ==========================================
  // 2. Active Platform Detection
  // ==========================================
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs && tabs[0] && tabs[0].url) {
      const url = tabs[0].url.toLowerCase();
      if (url.includes("chatgpt") || url.includes("openai")) {
        siteEl.innerText = "ChatGPT Active";
      } else if (url.includes("claude.ai")) {
        siteEl.innerText = "Claude.ai Active";
      } else if (url.includes("gemini.google")) {
        siteEl.innerText = "Google Gemini Active";
      } else if (url.includes("copilot.microsoft")) {
        siteEl.innerText = "Copilot Active";
      } else if (url.includes("deepseek")) {
        siteEl.innerText = "DeepSeek Active";
      } else if (url.includes("perplexity")) {
        siteEl.innerText = "Perplexity Active";
      } else if (url.includes("grok") || url.includes("x.com")) {
        siteEl.innerText = "Grok Active";
      } else if (url.includes("localhost") || url.includes("127.0.0.1")) {
        siteEl.innerText = "Local AI WebUI";
      } else if (url.startsWith("chrome://") || url.startsWith("edge://")) {
        siteEl.innerText = "AI Platform Standby";
        statusTag.innerText = "IDLE";
        statusTag.style.background = "var(--border-subtle)";
        statusTag.style.color = "var(--text-tertiary)";
        statusTag.style.borderColor = "var(--border-subtle)";
        dot.style.background = "var(--text-tertiary)";
        dot.style.boxShadow = "none";
      } else {
        siteEl.innerText = "Universal AI Guard";
      }
    }
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
    ["totalContradictionsBlocked", "activeBeliefsCount", "highlightConflicts"],
    (data) => {
      const beliefs = data.activeBeliefsCount || 0;
      const conflicts = data.totalContradictionsBlocked || 0;

      document.getElementById("stat-beliefs").innerText = beliefs;
      const confEl = document.getElementById("stat-conflicts");
      confEl.innerText = conflicts;

      if (conflicts > 0) {
        confEl.style.color = "var(--danger-text)";
      } else {
        confEl.style.color = "var(--success-text)";
      }

      updateConsistencyGauge(beliefs, conflicts);

      if (data.highlightConflicts !== undefined) {
        toggleHighlight.checked = data.highlightConflicts;
      }
    }
  );

  // ==========================================
  // 4. Toggle In-Chat Highlighting
  // ==========================================
  toggleHighlight.addEventListener("change", (e) => {
    const isChecked = e.target.checked;
    chrome.storage.local.set({ highlightConflicts: isChecked });

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "toggle_highlights",
          enabled: isChecked
        }, () => {
          if (chrome.runtime.lastError) { /* ignore */ }
        });
      }
    });
  });

  // ==========================================
  // 5. Audit Button
  // ==========================================
  auditBtn.addEventListener("click", () => {
    auditBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="animation: alethex-spin 1s linear infinite;">
        <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
        <path d="M21 3v5h-5"/>
      </svg>
      <span>Scanning Conversation...</span>
    `;
    auditBtn.style.opacity = "0.75";
    auditBtn.disabled = true;

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "audit_now" }, () => {
          setTimeout(() => {
            auditBtn.innerHTML = `
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
                <path d="M21 3v5h-5"/>
              </svg>
              <span>Audit Conversation Consistency</span>
            `;
            auditBtn.style.opacity = "1";
            auditBtn.disabled = false;
          }, 800);
        });
      } else {
        auditBtn.innerText = "Audit Complete";
        setTimeout(() => {
          auditBtn.innerText = "Audit Conversation Consistency";
          auditBtn.style.opacity = "1";
          auditBtn.disabled = false;
        }, 800);
      }
    });
  });

  // ==========================================
  // 6. Open Inspector Drawer Link
  // ==========================================
  openDrawerLink.addEventListener("click", (e) => {
    e.preventDefault();
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "open_drawer" }, () => {
          if (chrome.runtime.lastError) { /* ignore */ }
          window.close(); // Close popup so user interacts with in-page drawer
        });
      }
    });
  });
});
