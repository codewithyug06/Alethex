/**
 * Popup Script for ALETHEX Universal Chrome Extension (v1.2.0)
 * Modern, error-free communication with active tab content scripts.
 */

document.addEventListener("DOMContentLoaded", () => {
  const siteEl = document.getElementById("site-name");
  const statusTag = document.getElementById("status-tag");
  const dot = document.getElementById("dot-indicator");

  // Query active tab URL safely
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs && tabs[0] && tabs[0].url) {
      const url = tabs[0].url.toLowerCase();
      if (url.includes("chatgpt") || url.includes("openai")) {
        siteEl.innerText = "ChatGPT Guarded";
      } else if (url.includes("claude.ai")) {
        siteEl.innerText = "Claude.ai Guarded";
      } else if (url.includes("gemini.google")) {
        siteEl.innerText = "Gemini Guarded";
      } else if (url.includes("copilot.microsoft")) {
        siteEl.innerText = "Copilot Guarded";
      } else if (url.includes("deepseek")) {
        siteEl.innerText = "DeepSeek Guarded";
      } else if (url.includes("perplexity")) {
        siteEl.innerText = "Perplexity Guarded";
      } else if (url.includes("grok") || url.includes("x.com")) {
        siteEl.innerText = "Grok Guarded";
      } else if (url.includes("localhost") || url.includes("127.0.0.1")) {
        siteEl.innerText = "Local AI Guarded";
      } else if (url.startsWith("chrome://") || url.startsWith("edge://")) {
        siteEl.innerText = "Open an AI Chat Tab";
        statusTag.innerText = "STANDBY";
        statusTag.style.background = "rgba(148, 163, 184, 0.15)";
        statusTag.style.color = "#94a3b8";
        statusTag.style.borderColor = "rgba(148, 163, 184, 0.3)";
        dot.style.background = "#94a3b8";
        dot.style.boxShadow = "none";
      } else {
        siteEl.innerText = "Universal AI Guard";
      }
    }
  });

  // Load stats from chrome.storage
  chrome.storage.local.get(
    ["totalContradictionsBlocked", "activeBeliefsCount", "highlightConflicts"],
    (data) => {
      const beliefs = data.activeBeliefsCount || 0;
      const conflicts = data.totalContradictionsBlocked || 0;
      document.getElementById("stat-beliefs").innerText = beliefs;
      
      const confEl = document.getElementById("stat-conflicts");
      confEl.innerText = conflicts;
      if (conflicts > 0) {
        confEl.style.color = "#f87171"; // Red alert if conflicts caught
      } else {
        confEl.style.color = "#34d399"; // Green if zero conflicts
      }

      if (data.highlightConflicts !== undefined) {
        document.getElementById("toggle-highlight").checked = data.highlightConflicts;
      }
    }
  );

  // Toggle Highlight Switch
  document.getElementById("toggle-highlight").addEventListener("change", (e) => {
    const isChecked = e.target.checked;
    chrome.storage.local.set({ highlightConflicts: isChecked });

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "toggle_highlights",
          enabled: isChecked
        }, () => {
          if (chrome.runtime.lastError) {
            // Ignored on non-injected pages
          }
        });
      }
    });
  });

  // Audit Button (Safe Message Relay - Zero executeScript dependency)
  const auditBtn = document.getElementById("audit-btn");
  auditBtn.addEventListener("click", () => {
    auditBtn.innerText = "Auditing Conversation...";
    auditBtn.style.opacity = "0.7";

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "audit_now" }, (response) => {
          setTimeout(() => {
            auditBtn.innerHTML = `
              <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
              <span>Audit Conversation Consistency</span>
            `;
            auditBtn.style.opacity = "1";
            window.close(); // Close popup so user sees the opened drawer on the page!
          }, 350);

          if (chrome.runtime.lastError) {
            console.log("[ALETHEX] Tab not on AI page or content script loading:", chrome.runtime.lastError.message);
          }
        });
      } else {
        auditBtn.innerText = "Audit Conversation Consistency";
        auditBtn.style.opacity = "1";
      }
    });
  });
});
