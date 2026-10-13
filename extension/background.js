/**
 * ALETHEX Universal Chrome Extension Background Service Worker (Manifest V3)
 * Coordinates memory auditing, local microservice communication, and cross-tab state.
 */

console.log("[ALETHEX Background] Universal AI Memory Guard active.");

chrome.runtime.onInstalled.addListener(() => {
  console.log("[ALETHEX Background] Extension installed successfully.");
  chrome.storage.local.set({
    enabled: true,
    highlightConflicts: true,
    totalAuditedTurns: 0,
    totalContradictionsBlocked: 0,
    activeBeliefsCount: 0,
    backendStatus: "auto" // 'online', 'standalone', 'auto'
  });
});

// Periodic or on-demand health check for local ALETHEX python backend
async function checkLocalBackend() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1200);
    const resp = await fetch("http://localhost:8080/docs", { signal: controller.signal });
    clearTimeout(timeoutId);
    return resp.status === 200;
  } catch (e) {
    return false;
  }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "check_backend") {
    checkLocalBackend().then(isOnline => {
      sendResponse({ online: isOnline });
    });
    return true;
  }

  if (request.action === "audit_context") {
    const chunks = request.chunks || [];
    
    // Check if local python microservice is running
    checkLocalBackend().then(isOnline => {
      if (isOnline) {
        fetch("http://localhost:8080/v1/rag/filter", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chunks: chunks, mode: "annotate" })
        })
        .then(res => res.json())
        .then(data => {
          sendResponse({ status: "success", backend: "onnx_int8", data: data });
        })
        .catch(err => {
          sendResponse({ status: "fallback", backend: "client_engine", error: err.toString() });
        });
      } else {
        // Standalone client engine handles it locally
        sendResponse({ status: "standalone", backend: "client_engine" });
      }
    });

    return true; // Keep message channel open for async response
  }

  if (request.action === "update_stats") {
    chrome.storage.local.get(
      ["totalAuditedTurns", "totalContradictionsBlocked", "activeBeliefsCount"],
      (res) => {
        const turns = (res.totalAuditedTurns || 0) + (request.turns || 0);
        const conflicts = (res.totalContradictionsBlocked || 0) + (request.conflicts || 0);
        const beliefs = request.activeBeliefs !== undefined ? request.activeBeliefs : (res.activeBeliefsCount || 0);
        chrome.storage.local.set({
          totalAuditedTurns: turns,
          totalContradictionsBlocked: conflicts,
          activeBeliefsCount: beliefs
        });
      }
    );
    sendResponse({ ok: true });
    return false;
  }
});
