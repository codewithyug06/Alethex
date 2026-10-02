# Alethex

> **Universal Real-Time AI Memory & Contradiction Guard**  
> *Active Neuro-Symbolic truth verification and hallucination auditing for ChatGPT, Claude, Gemini, Copilot, DeepSeek, Perplexity, Grok, and Local LLMs.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Chrome Extension: Manifest V3](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-success.svg)](./alethex-guard-v1.2)
[![Tests](https://img.shields.io/badge/tests-48%2F48%20passed-brightgreen.svg)]()

---

## 🌟 Overview

Modern Large Language Models (LLMs) operate with **append-only context windows**. When facts evolve over time (*"User prefers Python"* ➔ *"Switched to Rust"*), standard RAG and conversational buffers retrieve both statements simultaneously, resulting in contradictory outputs, reasoning breakdown, and silent hallucinations.

**ALETHEX** solves this fundamental flaw by transforming raw conversational streams into a living, multi-directed **Temporal Belief Graph**. It monitors AI generations in real time, verifies cross-turn factual consistency using Natural Language Inference (NLI) and Temporal Interval Logic (Allen's interval algebra), and visibly flags contradictions directly on any AI interface.

---

## ⚡ Key Components

### 1. 🛡️ ALETHEX Universal Chrome Extension (`alethex-guard-v1.2`)
A lightweight, non-intrusive Manifest V3 browser extension that attaches an active truth HUD to major AI platforms:
- **Supported Platforms:** ChatGPT, Claude.ai, Google Gemini, Microsoft Copilot, DeepSeek, Perplexity AI, Grok / X, and Local WebUIs (`localhost:3000`, `127.0.0.1:8080`).
- **Live Memory HUD:** Real-time counter of verified claims, contradictions, and system status docked discreetly in the top-right corner.
- **Visual Contradiction Banners:** Red warning ribbons highlight conflicting statements inline before they derail reasoning.
- **Instant Audit Modal:** Full inspection drawer showing timestamped claims, confidence scores, and Allen temporal relations.
- **Offline / Local Fallback:** Runs self-contained rules and regex heuristic parsing locally, and seamlessly connects to the Python engine when active.

### 2. 🧠 Neuro-Symbolic Verification Engine (`chronos/alethex`)
- **OpenIE & Claim Extraction:** Deconstructs complex user prompts and model responses into atomic subject-predicate-object triples with temporal anchors.
- **Cross-Turn Coreference & Entity Linking:** Clusters entities across multi-turn dialogues with DBSCAN semantic embeddings.
- **TIMEX Interval Normalization:** Resolves relative temporal expressions into ISO intervals using Allen's Interval Algebra.
- **Staged NLI Verification:** High-throughput relation classifier scoring entailment, contradiction, and neutral drift.
- **Temporal Belief Graph (TBG):** Dynamic NetworkX/GraphML representation computing a mathematical **Consistency Index (CI)**.

### 3. 🌐 Global Landing Page (`index.html`)
A high-conversion, responsive presentation site featuring live interactive demos, architecture breakdown, platform status, and 1-click extension download.

---

## 🚀 Quick Start

### 🔧 Installing the Chrome Extension
1. Clone this repository:
   ```bash
   git clone https://github.com/codewithyug06/Alethex.git
   cd Alethex
   ```
2. Open Google Chrome and navigate to `chrome://extensions`.
3. Enable **Developer mode** (toggle in the top-right corner).
4. Click **Load unpacked** (top-left).
5. Select the `alethex-guard-v1.2` directory inside this repository.
6. Navigate to any supported AI (ChatGPT, Claude, Gemini, etc.) — the ALETHEX Truth HUD will appear automatically!

---

### 🐍 Running the Python Core Engine
```bash
# Navigate to the chronos directory
cd chronos

# Install dependencies
pip install -e .

# Run test suite
pytest -q

# Launch all background services & desktop companion
python start_all.py
```

---

## 📂 Repository Structure

```text
Alethex/
├── index.html                   # Global landing page & product showcase
├── README.md                    # Project documentation
├── launch_all.bat               # 1-Click launcher for engine & services
├── run_desktop.bat              # Desktop companion runner
├── open_extension_folder.bat    # Extension folder locator helper
├── alethex-guard-v1.2/          # Chrome Extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html
│   ├── popup.js
│   └── icons/
├── dist/
│   └── alethex-extension-v1.0.0.zip
└── chronos/                     # Core Python verification package
    ├── pyproject.toml
    ├── requirements.txt
    ├── alethex_desktop.py       # Desktop monitoring companion
    ├── start_all.py             # Service orchestrator
    ├── configs/                 # Pipeline and model configurations
    ├── tests/                   # Integration and unit tests (48 passing)
    └── alethex/                 # Engine implementation
        ├── models/              # NLI & Claim extraction models
        ├── temporal/            # Allen interval algebra & TIMEX parser
        ├── graph/               # Temporal Belief Graph & Consistency Index
        ├── memory/              # LangChain & LlamaIndex adapters
        └── api/                 # FastAPI REST endpoint
```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](chronos/LICENSE) file for details.
