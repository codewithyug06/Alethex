/**
 * ALETHEX NLI Web Worker
 * Loads Xenova/nli-deberta-v3-small via Transformers.js (ONNX, fully in-browser).
 * First load: ~85MB downloaded and cached by browser. Subsequent loads: instant.
 *
 * Receives: { id, type: "nli_batch", data: { pairs: [{premise, hypothesis, meta}], threshold } }
 * Sends:    { id, type: "nli_result", results: [{...meta, contradictionScore, reason}] }
 */

// Vendored locally — do NOT change back to a CDN URL. This extension has access to the user's
// AI chat sessions; loading remote executable JS from a CDN is a critical supply chain risk.
importScripts(chrome.runtime.getURL("vendor/transformers.min.js"));

const { pipeline, env } = self.Transformers;
env.allowLocalModels = false;
env.useBrowserCache = true;

let clf = null;
let loadPromise = null;

function getModel() {
  if (clf) return Promise.resolve(clf);
  if (loadPromise) return loadPromise;

  self.postMessage({ type: "status", message: "Loading NLI model — first time takes ~30s, then cached forever." });

  loadPromise = pipeline("zero-shot-classification", "Xenova/nli-deberta-v3-small", { quantized: true })
    .then((model) => {
      clf = model;
      loadPromise = null;
      self.postMessage({ type: "status", message: "NLI model ready." });
      return clf;
    })
    .catch((e) => {
      loadPromise = null;
      self.postMessage({ type: "error", message: "Model load failed: " + e.message });
      return null;
    });

  return loadPromise;
}

self.onmessage = async (e) => {
  const { id, type, data } = e.data;

  if (type === "ping") {
    getModel(); // preload without blocking
    self.postMessage({ id, type: "pong" });
    return;
  }

  if (type === "nli_batch") {
    const model = await getModel();
    if (!model) {
      self.postMessage({ id, type: "nli_result", results: [], error: "Model not available" });
      return;
    }

    const threshold = data.threshold || 0.60;
    const results = [];

    for (const pair of (data.pairs || [])) {
      if (!pair.premise || !pair.hypothesis) continue;
      if (pair.premise.length < 10 || pair.hypothesis.length < 10) continue;
      try {
        // zero-shot-classification runs NLI internally: premise is the sequence,
        // candidate labels are scored as entailment hypotheses.
        const out = await model(pair.premise, ["contradiction", "same meaning", "unrelated"]);
        const idx = out.labels.indexOf("contradiction");
        const score = idx >= 0 ? out.scores[idx] : 0;
        if (score >= threshold) {
          results.push({
            ...(pair.meta || {}),
            contradictionScore: score,
            reason: `Contradicting "${pair.premise.slice(0, 55)}..." ↔ "${pair.hypothesis.slice(0, 55)}..."`
          });
        }
      } catch (_) {
        // Skip failing pairs silently
      }
    }

    self.postMessage({ id, type: "nli_result", results });
  }
};
