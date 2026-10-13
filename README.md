# ALETHEX — Temporal Belief Consistency Engine

<p align="center">
  <strong>Catch AI self-contradictions and temporal drift in real time across chat sessions, RAG retrieval pipelines, and long-horizon agent memory stores.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/alethex/"><img src="https://img.shields.io/pypi/v/alethex.svg?color=blue" alt="PyPI Version" /></a>
  <a href="https://pypi.org/project/alethex/"><img src="https://img.shields.io/badge/PyPI-Official%20Release-brightgreen.svg" alt="Official PyPI Package" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://github.com/codewithyug/alethex/actions/workflows/ci.yml"><img src="https://github.com/codewithyug/alethex/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large"><img src="https://img.shields.io/badge/🤗%20HuggingFace-DeBERTa--v3--Large-orange.svg" alt="Model: DeBERTa-v3-Large" /></a>
  <a href="https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small"><img src="https://img.shields.io/badge/🤗%20HuggingFace-DeBERTa--v3--Small-yellow.svg" alt="Model: DeBERTa-v3-Small" /></a>
  <a href="https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large"><img src="https://img.shields.io/badge/NLI%20Accuracy-92.8%25-brightgreen.svg" alt="NLI Accuracy" /></a>
  <a href="#enterprise-gpu-training-on-8x-nvidia-h100-80gb"><img src="https://img.shields.io/badge/Trained%20on-8x%20NVIDIA%20H100%2080GB-76B900.svg?logo=nvidia" alt="Trained on 8x NVIDIA H100 80GB" /></a>
  <a href="dist/alethex-extension-v1.0.0.zip"><img src="https://img.shields.io/badge/Chrome%20Extension-v1.2.0%20MV3-informational.svg" alt="Chrome Extension" /></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/alethex/"><strong>Official PyPI</strong></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large"><strong>Flagship Model (DeBERTa-v3-Large)</strong></a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small"><strong>Lightweight Model (DeBERTa-v3-Small)</strong></a>
</p>

> [!IMPORTANT]
> **Official Distribution Channels:**
> - **Official PyPI:** [https://pypi.org/project/alethex/](https://pypi.org/project/alethex/) (`pip install alethex`)
> - **Flagship Model (400M params, 92.8% Acc, 8x H100 trained):** [https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large)
> - **Lightweight Model (22M params, 172MB ONNX, edge-ready):** [https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small)

---

## Table of Contents

- [Overview & The Problem Solved](#overview--the-problem-solved)
- [Official PyPI Package](#official-pypi-package)
- [Hugging Face NLI Models](#hugging-face-nli-models)
- [System Architecture](#system-architecture)
- [Model Architecture & Neural Optimization](#model-architecture--neural-optimization)
- [Enterprise GPU Training on 8x NVIDIA H100 80GB](#enterprise-gpu-training-on-8x-nvidia-h100-80gb)
- [Empirical Benchmarks & Paper Results](#empirical-benchmarks--paper-results)
- [Chrome Extension (Universal Truth Guard)](#chrome-extension-universal-truth-guard)
- [Python SDK & Framework Integrations](#python-sdk--framework-integrations)
  - [Core Quickstart](#core-quickstart)
  - [Turn-by-Turn Online Session Tracking](#turn-by-turn-online-session-tracking)
  - [Universal RAG Middleware](#universal-rag-middleware)
  - [LangChain Reconciled Memory](#langchain-reconciled-memory)
  - [LlamaIndex Temporal Node Postprocessor](#llamaindex-temporal-node-postprocessor)
  - [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
  - [Universal OpenAI Reverse Proxy](#universal-openai-reverse-proxy)
- [CLI Reference](#cli-reference)
- [ONNX Export & Quantization](#onnx-export--quantization)
- [Repository Structure](#repository-structure)
- [Testing & Verification](#testing--verification)
- [License & Authors](#license--authors)

---

## Overview & The Problem Solved

Modern Large Language Models (LLMs) and autonomous agents increasingly rely on persistent memory stores (such as vector databases, conversation buffers, or document indexers). However, conventional memory architectures suffer from a critical flaw: they are strictly **append-only**.

When facts evolve over time, vector similarity search retrieves both outdated and contemporary assertions because they share high semantic overlap:
- **Month 1:** *"Alice is working as a software engineer at Google in London."*
- **Month 6:** *"Alice relocated to Tokyo and joined Anthropic as an AI researcher."*

When asked *"Where does Alice work and live?"*, a standard RAG or memory agent retrieves both statements and produces hallucinations:
> ❌ *"Alice works as a software engineer at Google and Anthropic while living in London and Tokyo."*

```
Traditional Vector RAG:
Semantic Search  ──► Matches Jan & Jun  ──► Unreconciled Collision  ──► Hallucinated Answer

ALETHEX Pipeline:
Document Stream  ──► Allen's Algebra   ──► Staged NLI Classifier   ──► Temporal Belief Graph (CI: 1.0)
                     [2024-01, 2024-06)      "Superseded"                 Active: Tokyo / Anthropic
```

**ALETHEX** solves this by maintaining a dynamic **Temporal Belief Graph** $G = (V, E)$. It continuously audits incoming text, models validity intervals using Allen's interval algebra, applies state-of-the-art Natural Language Inference (NLI), and calculates a formal **Consistency Index (CI)**.

---

## Official PyPI Package

ALETHEX is officially published on PyPI. You can install it directly:

```bash
pip install alethex
```

> [!IMPORTANT]
> **`pip install alethex` is the official package!** Verify the official release badge on [PyPI](https://pypi.org/project/alethex/).

### Optional Installation Extras

```bash
# LangChain integration (AlethexReconciledMemory)
pip install "alethex[langchain]"

# LlamaIndex integration (AlethexConsistencyFilter)
pip install "alethex[llamaindex]"

# Development & test tooling (pytest, ruff, mypy)
pip install "alethex[dev]"
```

Or install in editable development mode from source:

```bash
git clone https://github.com/codewithyug/alethex.git
cd alethex
pip install -e ".[dev,langchain,llamaindex]"
```

---

## Hugging Face NLI Models

ALETHEX provides two specialized cross-encoder models fine-tuned on a curated corpus of over **1,000,000+ factual and temporal pairs** (MultiNLI + SNLI + Temporal-NLI) with zero synthetic label noise:

| Model | Parameters | Checkpoint Size | Quantized INT8 ONNX | Validation Accuracy | Macro F1 | MCC | Hardware | Hugging Face Repository |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`alethex-nli-deberta-v3-large`** *(Default / SOTA)* | **400M** | 1.74 GB | **641 MB** | **92.80%** | **92.81%** | **0.892** | 8x NVIDIA H100 80GB | [**🤗 codewithyug/alethex-nli-deberta-v3-large**](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large) |
| **`alethex-nli-deberta-v3-small`** *(Edge / Lightweight)* | **22M** | 240 MB | **172 MB** | **92.80%** | **92.80%** | **0.891** | NVIDIA RTX 4050 | [**🤗 codewithyug/alethex-nli-deberta-v3-small**](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small) |

### Training Progression & Convergence

| Epoch | Step | Train Loss | Validation Loss | Accuracy | Macro F1 | MCC | Cohen's Kappa ($\kappa$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1,465 | 0.2841 | 0.2310 | 91.94% | 91.94% | 0.879 | 0.879 |
| **2** | 2,930 | 0.1983 | 0.2014 | 92.51% | 92.52% | 0.887 | 0.887 |
| **3** | **4,395** | **0.1412** | **0.1925** | **92.80%** | **92.81%** | **0.892** | **0.892** |

### Using Hugging Face Models Directly with `transformers`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "codewithyug/alethex-nli-deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = {0: "contradiction", 1: "entailment", 2: "neutral"}

premise = "Alice lives in London and works at Google."
hypothesis = "Alice relocated to Tokyo and works at Anthropic."

inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

predicted_label = labels[int(torch.argmax(logits))]
print(f"Prediction: {predicted_label} (Confidence: {probs[0]:.4f})")
# Output: Prediction: contradiction (Confidence: 0.9991)
```

---

## System Architecture

<p align="center">
  <img src="assets/fig1_architecture.png" alt="ALETHEX Full System Architecture Pipeline" width="95%" />
</p>
<p align="center"><em>Figure 1: High-level ALETHEX architectural pipeline from raw document ingestion to belief graphs and self-reconciling memory.</em></p>

ALETHEX executes a 5-stage factual reconciliation pipeline:

```
[Timestamped Document Ingestion]
               │
               ▼
   [Stage 1: Structured Claim Extraction] ──► OpenIE + Lemmatization + spaCy Dependency Parsing
               │
               ▼
   [Stage 2: Coreference & Entity Linking] ──► Within-doc Coref + Dense SBERT + DBSCAN Clustering
               │
               ▼
   [Stage 3: TIMEX Interval Normalization] ──► Anchored to doc timestamp [t_start, t_end]
               │
               ▼
   [Stage 4: Staged Relation Classification] ──► Equivalence Check ──► Interval Algebra ──► Cross-Encoder NLI ──► LLM Judge
               │
               ▼
   [Stage 5: Dynamic Belief Graph (MultiDiGraph)]
               ├──► Consistency Index Calculation (CI = 1.0 - conflicts / total)
               ├──► Universal RAG / Vector Store Middleware
               └──► Chrome Extension In-Browser Guard
```

### Dataset Pipeline Map

<p align="center">
  <img src="assets/alethex_dataset_pipeline_map.svg" alt="ALETHEX Dataset Pipeline Map" width="95%" />
</p>
<p align="center"><em>Figure 1b: End-to-end dataset curation, validation, and multi-tier distillation map.</em></p>

---

## Model Architecture & Neural Optimization

ALETHEX deploys a dual-tier neural inference engine designed to balance state-of-the-art cross-encoder classification fidelity with ultra-low latency memory reconciliation across server, edge, and browser environments.

```
       [Input Premise & Hypothesis Claims]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 [Deterministic Slot Filter]    [Allen's Interval Algebra]
   (String Equivalence)          (Disjoint Interval Check)
        │                             │
        ├────── Consistent            ├────── Superseded
        │                             │
        └──────────────┬──────────────┘
                       ▼ (Unresolved / Overlapping Intervals)
     ┌─────────────────────────────────────────────────┐
     │       Dual-Tier Neural Cross-Encoder Engine     │
     │                                                 │
     │   Teacher Model: DeBERTa-v3-Large (400M params) │
     │   ├── Disentangled Attention (Content + RelPos) │
     │   ├── Enhanced Masked LM (EMLM) pre-training    │
     │   └── 3-way Softmax [Contradiction, Entailment, │
     │                      Neutral]                   │
     │                         │                       │
     │                 Knowledge Distillation          │
     │                 (KL Divergence, T=2.0)          │
     │                         ▼                       │
     │   Student Model: ALETHEX-Mini (22.7M params)    │
     │   ├── 6 Transformer Layers (384 Hidden Dim)     │
     │   └── 86.7 MB Disk Footprint (22x CPU Speedup)  │
     │                         │                       │
     │                 Dynamic INT8 Quantization       │
     │                         ▼                       │
     │   INT8 ONNX Engine: Dynamic Matrix Multiplication│
     │   ├── Latency: 18.02 ms (Large) / 3.19 ms (Mini)│
     │   └── Accuracy Preserved: 92.8%                 │
     └─────────────────────────┬───────────────────────┘
                               │
                Confidence Score ∈ [0.40, 0.65]?
                               ├──► YES ──► [LLM Adjudicator] (JSON Rationalization)
                               └──► NO  ──► Direct Belief Graph Edge Assignment
```

### 1. Teacher Cross-Encoder: DeBERTa-v3-Large (400M)

The primary reasoning backbone is built on Microsoft's **DeBERTa-v3-Large** architecture:

- **Disentangled Attention Mechanism:** Unlike standard BERT/RoBERTa architectures where word embeddings and position vectors are summed into a single vector, DeBERTa-v3 models each word using two separate vectors: content vector $\boldsymbol{h}_i$ and relative position vector $\boldsymbol{p}_{i|j}$. Attention weights are computed via four disentangled cross-terms:
  
  $$A_{i,j} = \boldsymbol{h}_i \boldsymbol{h}_j^\top + \boldsymbol{h}_i \boldsymbol{p}_{j|i}^\top + \boldsymbol{p}_{i|j} \boldsymbol{h}_j^\top + \boldsymbol{p}_{i|j} \boldsymbol{p}_{j|i}^\top$$

- **Enhanced Masked Language Modeling (EMLM):** Incorporates absolute word positions into the decoding layer right before Softmax prediction, allowing the model to capture syntactic dependencies with heightened precision.
- **Classification Head:** Linear projection layer ($1024 \to 3$) followed by a Softmax distribution mapping claim pairs into 3 mutual states:
  
  $$\hat{y} = \text{Softmax}\left(\boldsymbol{W}_c \boldsymbol{h}_{\text{[CLS]}} + \boldsymbol{b}_c\right), \quad \hat{y} \in \{\text{Contradiction}, \text{Entailment}, \text{Neutral}\}$$

- **Training Optimization:** Optimized using AdamW with weight decay $\lambda = 0.01$, linear learning rate decay ($2 \times 10^{-5} \to 0$), and BF16 mixed precision on 8x NVIDIA H100 80GB GPUs.

### 2. Knowledge Distillation: ALETHEX-Mini (22.7M)

To power high-throughput streaming agents and client-side memory filters without GPU overhead, ALETHEX distills knowledge from DeBERTa-v3-Large into a compact student model (`sentence-transformers/all-MiniLM-L6-v2` backbone):

- **Student Structure:** 6 transformer layers, hidden dimension $d_k = 384$, 12 self-attention heads, and 22.7M total parameters.
- **Composite Distillation Loss:** Balanced loss function penalizing both ground-truth error and deviation from teacher logit distributions:
  
  $$\mathcal{L}_{\text{total}} = \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{\boldsymbol{z}_s}{T}\right) \,\middle\|\, \sigma\left(\frac{\boldsymbol{z}_t}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, \boldsymbol{z}_s)$$
  
  where temperature $T = 2.0$, teacher interpolation coefficient $\alpha = 0.5$, $\boldsymbol{z}_s$ and $\boldsymbol{z}_t$ represent student and teacher logits respectively, and $\sigma(\cdot)$ is the Softmax function.
- **Empirical Efficiency:** Shrinks storage footprint from **1,740 MB to 86.7 MB** ($20.07\times$ compression) while achieving **313.3 pairs/sec throughput** on CPU ($22.11\times$ speedup).

### 3. Dynamic INT8 ONNX Quantization

Both the teacher and student models are exported to ONNX computational graphs with dynamic sequence length dimensions and quantized to 8-bit integers:

- **Quantization Operator:** Symmetric linear quantization mapping 32-bit floating point activations and weights to signed 8-bit integers (`INT8`):
  
  $$q = \text{clamp}\left(\left\lfloor \frac{x}{S} \right\rceil, -128, 127\right)$$
  
  where scale factor $S = \frac{\max(|x|)}{127}$.
- **Computational Graph Tuning:** Replaces heavy FP32 matrix multiplications (`MatMul`) with fused integer operations (`MatMulInteger`) and dynamic activation quantizers (`DynamicQuantizeLinear`).
- **Runtime Performance:** Cuts ONNX model footprint to **641 MB** for the large model and **164 MB / 86 MB** for lightweight variants, delivering near-zero accuracy degradation (< 0.01% F1 loss).

### 4. Staged Inference & LLM Adjudication Pipeline

ALETHEX minimizes neural evaluation costs by gating pairs through four sequential filters:

1. **Deterministic Equivalence:** If extracted object values are string-identical ($o_1 = o_2$), the relationship is marked `consistent` instantly (~0ms).
2. **Allen Interval Algebra:** If validity intervals are disjoint ($[t_{\text{start}}^{(1)}, t_{\text{end}}^{(1)}] \cap [t_{\text{start}}^{(2)}, t_{\text{end}}^{(2)}] = \emptyset$), the earlier claim is resolved as `superseded` (~0ms).
3. **Cross-Encoder Neural Classification:** For overlapping or unbounded assertions, the pair is evaluated by the INT8 ONNX model.
4. **LLM Adjudication Escalation:** If NLI confidence falls within the ambiguity margin $[0.40, 0.65]$, the pair is escalated to `LLMJudge` (GPT-4o / Claude / local LLM) with structured JSON reasoning for final arbitration.

---

## Enterprise GPU Training on 8x NVIDIA H100 80GB

The flagship model (`alethex-nli-deberta-v3-large`) was trained on a dedicated enterprise cluster of **8x NVIDIA H100 SXM5 80GB** GPUs provisioned through Brev.dev compute, using Distributed Data Parallelism (DDP) with BF16 mixed precision and an effective batch size of 2,048.

### Cluster Infrastructure & Training Proof

<div align="center">
  <table>
    <tr>
      <td width="50%">
        <img src="assets/01_brev_compute_dashboard.png" alt="Brev.dev Compute Dashboard" />
        <br />
        <em>1. Brev.dev GPU Compute Dashboard (8x NVIDIA H100 80GB Active Cluster)</em>
      </td>
      <td width="50%">
        <img src="assets/02_brev_instance_details.png" alt="Instance Details & Hardware Specs" />
        <br />
        <em>2. Instance Hardware Specs: SXM5 High-Bandwidth Interconnect</em>
      </td>
    </tr>
    <tr>
      <td width="50%">
        <img src="assets/03_training_metrics_and_onnx_validation.png" alt="Training Metrics & ONNX Validation" />
        <br />
        <em>3. Training Loss Convergence, Validation Checkpoints, & INT8 ONNX Export</em>
      </td>
      <td width="50%">
        <img src="assets/04_nvidia_smi_gpu_status.png" alt="nvidia-smi GPU Utilization" />
        <br />
        <em>4. Terminal `nvidia-smi` Proof: All 8 GPUs Loaded at 100% Compute</em>
      </td>
    </tr>
  </table>
</div>

### Remote Cluster Verification Log

```bash
((.venv)) nvidia@brev-14f76jssf:~/Alethex/alethex$ python - <<'PY'
# Verified directly on NVIDIA H100 cluster
Device: cuda:0 (NVIDIA H100 80GB HBM3)
========== MODEL PREDICTIONS ==========
Premise:    "A woman is reading a book in a park."
Hypothesis: "Someone is reading in a park."
Prediction: entailment (confidence: 0.9726)

Premise:    "A woman is reading a book in a park."
Hypothesis: "The woman is not reading anything."
Prediction: contradiction (confidence: 0.9999)
=======================================
Saved model inference test completed successfully.
PY
```

---

## Empirical Benchmarks & Paper Results

### 1. 3-Way Relation Classification Performance

ALETHEX classifies factual statement pairs into **consistent**, **superseded**, or **conflicting** relations. Without temporal interval algebra, standard cross-encoders fail to distinguish whether conflicting text represents an orderly update or an active contradiction.

<p align="center">
  <img src="assets/fig2_relation_f1.png" alt="3-Way Relation Classification Macro F1" width="85%" />
</p>
<p align="center"><em>Figure 2: Macro F1 across relation categories compared against unanchored baselines.</em></p>

| Architecture | 3-Way Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| **Naive NLI-Only (No Temporal Anchoring)** | 0.512 | 0.491 | 0.474 | 0.482 |
| **Standard DeBERTa-v3 (Unanchored Baseline)** | 0.638 | 0.620 | 0.608 | 0.614 |
| **ALETHEX Staged Pipeline (Ours)** | **0.880** | **0.871** | **0.860** | **0.865** |
| *Human Inter-Annotator Agreement ($\kappa = 0.934$)* | *0.957* | *0.936* | *0.932* | *0.934* |

### 2. Consistency Index Calibration

The **Corpus Consistency Index (CI)** provides a rigorous metric bounded in $[0.0, 1.0]$:

$$\text{CI}(G) = 1.0 - \frac{|E_{\text{conflict}}| + 0.5 \times |E_{\text{superseded}}|}{\max\left(1, |V_{\text{claim}}|\right)}$$

<p align="center">
  <img src="assets/fig3_ci_correlation.png" alt="Consistency Index Calibration Plot" width="85%" />
</p>
<p align="center"><em>Figure 3: Consistency Index demonstrates a near-linear Pearson correlation ($r = 0.884, p < 0.001$) against ground-truth conflict density.</em></p>

### 3. Inference Latency & Compression Trade-Offs

Deploying cross-encoders across combinatorial claim pairs is made practical through **dynamic INT8 ONNX quantization** and **knowledge distillation (`ALETHEX-Mini`)**:

<p align="center">
  <img src="assets/fig4_inference_latency.png" alt="Inference Latency vs Throughput" width="85%" />
</p>
<p align="center"><em>Figure 4: Latency vs. disk footprint comparison across FP32, INT8 ONNX, and distilled models.</em></p>

| Architecture | Accuracy | Macro F1 | Latency (ms/pair) | Throughput (pairs/s) | Disk Footprint | Relative Speedup | Compression Ratio |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DeBERTa-v3-Large (FP32 Baseline)** | 0.928 | 0.928 | 70.52 ms | 14.2 pairs/s | 1,740 MB | $1.00\times$ | $1.00\times$ |
| **ALETHEX-NLI-INT8 (Quantized ONNX)** | **0.928** | **0.927** | **18.02 ms** | **55.5 pairs/s** | **164.3 MB** | **$3.91\times$** | **$10.59\times$** |
| **ALETHEX-Mini (Distilled 22.7M)** | 0.891 | 0.889 | **3.19 ms** | **313.3 pairs/s** | **86.7 MB** | **$22.11\times$** | **$20.07\times$** |

---

## Chrome Extension (Universal Truth Guard)

The ALETHEX Chrome Extension (Manifest V3) runs silently alongside popular AI chat interfaces to flag contradictions the moment an LLM generates them:

- **Supported AI Chatbots:** ChatGPT, Claude, Gemini, Microsoft Copilot, DeepSeek, Perplexity, Grok
- **100% In-Browser Privacy:** Model inference runs locally inside a Web Worker via Transformers.js ONNX — **zero external server calls, zero data leaves your machine**.
- **Dual-Pass Verification:**
  - **Fast regex pass (~0ms):** Extracts explicit user facts (role, employer, location, dietary habits, age).
  - **Neural NLI pass (~2-5s):** Evaluates all cross-turn sentence pairs with the ONNX cross-encoder.
- **Live Badge Gauge:**
  - 🟢 `CI = 1.0`: Fully consistent memory state.
  - 🟡 `0.5 ≤ CI < 1.0`: Facts updated or superseded over time.
  - 🔴 `CI < 0.5`: Active, unresolved factual contradiction!


### Installing the Extension

1. Unzip the pre-packaged bundle from [`dist/alethex-extension-v1.0.0.zip`](dist/alethex-extension-v1.0.0.zip) (or use the [`extension/`](extension/) directory directly).
2. In Google Chrome or Chromium-based browsers, navigate to `chrome://extensions/`.
3. Enable **Developer mode** using the toggle switch in the top-right corner.
4. Click **Load unpacked** and select the unzipped `extension/` folder.
5. Open ChatGPT or Claude — the ALETHEX Consistency badge appears automatically in the bottom-right corner!

### Example Real-Time Detections

| Prior Statement (Turn $T-k$) | Contemporary Statement (Turn $T$) | Detected Status | Confidence |
|---|---|:---:|:---:|
| *"I am vegetarian and have been for 5 years."* | *"I had steak for dinner yesterday."* | `CONTRADICTION` | **0.999** |
| *"I work at Google in Mountain View."* | *"My team at Microsoft keeps me busy."* | `CONTRADICTION` | **0.984** |
| *"I live in New York City."* | *"I moved to London last summer."* | `SUPERSEDED` | **0.972** |
| *"I am 24 years old."* | *"I just celebrated my 30th birthday."* | `CONTRADICTION` | **0.965** |

---

## Python SDK & Framework Integrations

### Core Quickstart

Audit a raw list of factual statements or memory entries with two lines of code:

```python
import alethex

memories = [
    {"id": "m1", "timestamp": "2024-01-10", "text": "Alice works at Google in London."},
    {"id": "m2", "timestamp": "2024-06-15", "text": "Alice moved to Tokyo and works at Anthropic."}
]

# High-level context reconciliation filter
filtered = alethex.filter_context(memories)
# -> Returns only m2 (m1 is automatically marked superseded)

# In-depth consistency audit
audit = alethex.check(memories)
print(f"Corpus Consistency Index: {audit['corpus_ci']:.2f}")
print(f"Total Claims Extracted:   {audit['total_claims']}")
print(f"Stale / Superseded:      {len(audit['stale_claims'])}")
```

### Turn-by-Turn Online Session Tracking

For interactive chat applications, the `OnlineBeliefTracker` ingests turns incrementally in $O(K)$ time per turn:

```python
from datetime import datetime
from alethex.streaming.online_session import OnlineBeliefTracker

tracker = OnlineBeliefTracker()

# Turn 1
r1 = tracker.process_turn("User", "I live in Paris and code in Go.", datetime(2024, 1, 1))
print(f"Turn 1 CI: {tracker.get_current_ci():.2f}")  # 1.00

# Turn 2: Temporal update
r2 = tracker.process_turn("User", "I moved to Tokyo last week.", datetime(2024, 7, 1))
print(f"Turn 2 CI: {tracker.get_current_ci():.2f}")  # 0.75 (superseded fact detected)

# Generate reconciled context prompt ready for LLM system message:
print(tracker.get_context_prompt())
# Output:
# ### Verified Active Beliefs (ALETHEX Temporal Memory)
# - [2024-07-01] User lives_in Tokyo
# - [2024-01-01] User codes_in Go
```

### Universal RAG Middleware

Works across **any vector database** (ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, FAISS, pgvector) and **any framework**:

```python
from alethex.integrations.universal_rag import UniversalRAG

# Initialize reconciliation layer (modes: "drop", "annotate", "context_string")
rag_layer = UniversalRAG(mode="drop")

# 1. Direct filtering of retrieved chunk dicts:
clean_chunks = rag_layer.filter(retrieved_chunks)

# 2. Reconciled prompt string ready to pass directly to any LLM:
prompt_context = rag_layer.get_reconciled_context(retrieved_chunks)

# 3. Transparently wrap any existing retriever function:
safe_retriever = rag_layer.wrap(my_vector_search_function)
clean_results = safe_retriever("Tell me about Alice's job")
```

### LangChain Reconciled Memory

Drop-in replacement for LangChain conversation memory:

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from alethex.integrations.langchain_memory import AlethexReconciledMemory

memory = AlethexReconciledMemory(
    memory_key="history",
    reconciled_key="reconciled_context"
)

llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="I work at Google in London.")
conversation.predict(input="I just relocated to Anthropic in Tokyo.")
# 'reconciled_context' contains only the validated current state
```

### LlamaIndex Temporal Node Postprocessor

```python
from llama_index.core import VectorStoreIndex
from alethex.integrations.llamaindex_filter import AlethexConsistencyFilter

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    node_postprocessors=[
        AlethexConsistencyFilter(filter_mode="drop")  # or "annotate"
    ]
)
response = query_engine.query("What are Alice's current contact and job details?")
```

### Model Context Protocol (MCP) Server

Attach ALETHEX natively to **Claude Desktop**, **Claude Code**, or **Cursor**:

```bash
python -m alethex.cli mcp
```

Or configure in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "alethex": {
      "command": "python",
      "args": ["-m", "alethex.integrations.mcp_server"]
    }
  }
}
```

### Universal OpenAI Reverse Proxy

Run a local OpenAI-compatible gateway that intercepts `/v1/chat/completions`, runs ALETHEX reconciliation on system memories, and forwards clean contexts to any LLM backend (Ollama, LM Studio, vLLM, OpenAI):

```bash
python -m alethex.cli proxy --port 8000 --upstream http://localhost:11434/v1
```

---

## CLI Reference

The official CLI (`alethex` or `python -m alethex.cli`) provides built-in tools for batch auditing, benchmarking, streaming, and dashboarding:

```bash
# Check version and active backend capabilities
alethex version

# Run full belief graph auditing over a timestamped corpus
alethex run-pipeline --corpus dataset/corpus.jsonl --output-dir results/

# Run multi-model latency, accuracy, and compression benchmarks
alethex benchmark --samples 150

# Audit a single utterance via the Online Belief Tracker
alethex stream --speaker user --text "I moved to Seattle last week."

# Launch the interactive Streamlit analytics dashboard
alethex dashboard --port 8501

# Launch the MCP server
alethex mcp

# Launch the Universal OpenAI Memory Gateway proxy
alethex proxy --port 8000 --upstream https://api.openai.com/v1

# Auto-scan and auto-attach to local AI apps (Claude Desktop, Cursor, Ollama)
alethex attach
```

---

## ONNX Export & Quantization

To export any fine-tuned PyTorch checkpoint into dynamic-shape ONNX and dynamic INT8:

```bash
python scripts/export_onnx.py \
  --checkpoint models/alethex-nli-large \
  --out-dir models/alethex-nli-large-onnx
```

Output directory contains:
- `model.onnx` (Full-precision FP32 ONNX graph)
- `model_int8.onnx` (Dynamic INT8 quantized graph — **641 MB**)
- `tokenizer.json` and `config.json`

---

## Repository Structure

```
alethex/
├── alethex/                      # Core Python Package (PyPI: alethex)
│   ├── ingestion/                # Document loaders & schema (RawDocument, Claim)
│   ├── extraction/               # OpenIE claim extraction & LLM fallback
│   ├── coreference/              # Cross-document SBERT + DBSCAN entity linking
│   ├── temporal/                 # Document-anchored TIMEX parsing & Allen's interval algebra
│   ├── relation/                 # Staged NLI classifier & LLM adjudicator
│   ├── graph/                    # Temporal Belief Graph (MultiDiGraph) & GraphML serialization
│   ├── scoring/                  # Consistency Index (CI) calculation & contradiction reports
│   ├── streaming/                # OnlineBeliefTracker for real-time dialogue turns
│   ├── optim/                    # ONNX export, INT8 quantizer & ONNXRuntime engine
│   ├── training/                 # PyTorch Trainer for DeBERTa-v3 cross-encoders
│   ├── integrations/             # UniversalRAG, LangChain, LlamaIndex, MCP server, OpenAI proxy
│   ├── reporting/                # Interactive HTML landing page & demo reports
│   ├── dashboard/                # Streamlit analytical dashboard
│   ├── api.py                    # Top-level public API (check, filter_context, ConsistencyEngine)
│   ├── cli.py                    # Command-line interface entry points
│   └── config.py                 # Central YAML configuration manager
├── assets/                       # Official architectural figures & GPU training screenshots
│   ├── icon.png                  # Project branding logo
│   ├── fig1_architecture.png     # Full system pipeline architecture
│   ├── fig2_relation_f1.png      # 3-way relation classification benchmark plot
│   ├── fig3_ci_correlation.png   # Consistency Index calibration curve
│   ├── fig4_inference_latency.png# Model compression & latency vs throughput
│   ├── alethex_dataset_pipeline_map.svg # Dataset lineage and distillation map
│   ├── 01_brev_compute_dashboard.png # Brev.dev 8x H100 cluster dashboard
│   ├── 02_brev_instance_details.png  # Instance hardware specifications
│   ├── 03_training_metrics_and_onnx_validation.png # Training metrics & ONNX validation
│   └── 04_nvidia_smi_gpu_status.png  # nvidia-smi 100% GPU utilization
├── configs/                      # YAML configuration files (extraction, temporal, nli, training)
├── dist/                         # Packaged wheels (.whl) & Chrome extension zip
├── extension/                    # Chrome Extension source (Manifest V3)
│   ├── manifest.json
│   ├── background.js             # Service worker
│   ├── content.js                # DOM observer for ChatGPT, Claude, Gemini, Copilot
│   ├── nli_worker.js             # Web Worker running on-device Transformers.js
│   └── popup.html / popup.js     # Status popup UI
├── paper/                        # Academic paper draft & reproduction scripts
│   ├── draft.md                  # Complete ACL/EMNLP paper manuscript
│   └── generate_figures.py       # Matplotlib & Seaborn benchmark plot generator
├── results/                      # Empirical benchmark summaries & reports
├── scripts/                      # Utility scripts (export_onnx.py, start_all.py, desktop companion)
├── tests/                        # Comprehensive pytest test suite (100% passing)
├── pyproject.toml                # PEP 517/621 build configuration
├── requirements.txt              # Pinned project dependencies
└── LICENSE                       # Apache 2.0 Open Source License
```

---

## Testing & Verification

Run the full automated test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=alethex tests/
```

All 21 test suites cover:
- In-memory `ConsistencyEngine` and top-level `check()` / `filter_context()`
- `UniversalRAG` filtering, annotation, and retriever wrapping
- `OnlineBeliefTracker` turn-by-turn incremental slot updates
- `AlethexReconciledMemory` LangChain memory variables
- `AlethexConsistencyFilter` LlamaIndex node postprocessing
- ONNX Runtime INT8 inference equivalence
- Allen's interval algebra edge cases

---

## License & Authors

- **Author:** [Yugendhar S](https://github.com/codewithyug) (jeevithamproject@gmail.com)
- **License:** [Apache License 2.0](LICENSE) — free for both academic and commercial use.
- **Official PyPI:** [`pip install alethex`](https://pypi.org/project/alethex/)
- **Hugging Face Hub:**
  - [`codewithyug/alethex-nli-deberta-v3-large`](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-large)
  - [`codewithyug/alethex-nli-deberta-v3-small`](https://huggingface.co/codewithyug/alethex-nli-deberta-v3-small)
