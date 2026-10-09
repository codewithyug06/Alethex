# ALETHEX: Universal Temporal Belief Consistency and Memory Verification Engine for Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Chrome Extension: Manifest V3](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-success.svg)](./alethex-guard-v1.2)
[![Inference: ONNX INT8](https://img.shields.io/badge/Inference-ONNX%20INT8-orange.svg)]()
[![Model: ALETHEX--Mini](https://img.shields.io/badge/Distilled-ALETHEX--Mini%20(22.7M)-purple.svg)]()
[![Tests: Passing](https://img.shields.io/badge/Tests-48%2F48%20Passed-brightgreen.svg)]()

Corpus-scale temporal belief consistency, real-time contradiction auditing, and self-reconciling memory architectures for long-horizon AI agents, Retrieval-Augmented Generation (RAG) pipelines, and frontier web interfaces.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Foundations](#theoretical-foundations)
   - [The Append-Only Memory Failure Mode](#the-append-only-memory-failure-mode)
   - [Mathematical Formulation of Temporal Claims](#mathematical-formulation-of-temporal-claims)
   - [Allen's Interval Algebra](#allens-interval-algebra)
   - [Dynamic Temporal Belief Graph (DTBG)](#dynamic-temporal-belief-graph-dtbg)
   - [Consistency Index (CI) Formulation](#consistency-index-ci-formulation)
   - [Staleness Scoring Formulation](#staleness-scoring-formulation)
3. [System Architecture](#system-architecture)
   - [Pipeline Data Flow](#pipeline-data-flow)
   - [Stage 1: Document Ingestion and Normalization](#stage-1-document-ingestion-and-normalization)
   - [Stage 2: OpenIE and LLM Claim Extraction](#stage-2-openie-and-llm-claim-extraction)
   - [Stage 3: Coreference Resolution and Entity Linking](#stage-3-coreference-resolution-and-entity-linking)
   - [Stage 4: TIMEX3 Normalization and Temporal Resolution](#stage-4-timex3-normalization-and-temporal-resolution)
   - [Stage 5: Staged NLI Verification and Hybrid Classification](#stage-5-staged-nli-verification-and-hybrid-classification)
   - [Stage 6: Temporal Belief Graph Assembly and Pruning](#stage-6-temporal-belief-graph-assembly-and-pruning)
   - [Stage 7: Quantitative Consistency Auditing and Reporting](#stage-7-quantitative-consistency-auditing-and-reporting)
   - [Stage 8: Context-Aware Memory Reconciliation](#stage-8-context-aware-memory-reconciliation)
4. [Neural Model Architecture and Optimization Engine](#neural-model-architecture-and-optimization-engine)
   - [Teacher Cross-Encoder (DeBERTa-v3)](#teacher-cross-encoder-deberta-v3)
   - [Student Architecture (ALETHEX-Mini, 22.7M Parameters)](#student-architecture-alethex-mini-227m-parameters)
   - [Knowledge Distillation Formulation](#knowledge-distillation-formulation)
   - [Dynamic INT8 Quantization Architecture](#dynamic-int8-quantization-architecture)
   - [Dual-Layer Neural Architecture Diagram](#dual-layer-neural-architecture-diagram)
5. [End-to-End LLM Workflow Walkthrough: Production Case Study](#end-to-end-llm-workflow-walkthrough-production-case-study)
   - [Enterprise Scenario: Multi-Turn Infrastructure Evolution](#enterprise-scenario-multi-turn-infrastructure-evolution)
   - [Step 1: Raw Turn Ingestion & Atomic Claim Extraction](#step-1-raw-turn-ingestion--atomic-claim-extraction)
   - [Step 2: Canonical Entity Linking via SBERT and DBSCAN](#step-2-canonical-entity-linking-via-sbert-and-dbscan)
   - [Step 3: Temporal Interval Normalization and Allen Calculus](#step-3-temporal-interval-normalization-and-allen-calculus)
   - [Step 4: Staged NLI Verification and Logit Distribution](#step-4-staged-nli-verification-and-logit-distribution)
   - [Step 5: Dynamic Belief Graph Mutation and Supersession](#step-5-dynamic-belief-graph-mutation-and-supersession)
   - [Step 6: Memory Reconciliation and Context Injection](#step-6-memory-reconciliation-and-context-injection)
   - [Comparative Output Analysis: Standard LLM vs ALETHEX-Guarded LLM](#comparative-output-analysis-standard-llm-vs-alethex-guarded-llm)
   - [Real-Time User Interface Visualization](#real-time-user-interface-visualization)
6. [Client Applications and User Interfaces](#client-applications-and-user-interfaces)
   - [Universal Chrome Extension (Manifest V3)](#universal-chrome-extension-manifest-v3)
   - [Native Desktop Companion](#native-desktop-companion)
   - [Production Landing Page and Documentation Hub](#production-landing-page-and-documentation-hub)
7. [Enterprise Integrations and Middleware Gateways](#enterprise-integrations-and-middleware-gateways)
   - [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
   - [Universal OpenAI-Compatible Reverse Proxy](#universal-openai-compatible-reverse-proxy)
   - [Open WebUI Pipeline Filter](#open-webui-pipeline-filter)
   - [Universal RAG Middleware](#universal-rag-middleware)
   - [LangChain Memory Adapter](#langchain-memory-adapter)
   - [LlamaIndex Consistency Postprocessor](#llamaindex-consistency-postprocessor)
   - [Automated Environment Detector and Attacher](#automated-environment-detector-and-attacher)
   - [FastAPI REST Service and Visualization Dashboard](#fastapi-rest-service-and-visualization-dashboard)
8. [Model Distillation and Optimization Benchmarks](#model-distillation-and-optimization-benchmarks)
   - [Empirical Relation Classification Benchmarks](#empirical-relation-classification-benchmarks)
   - [Latency, Throughput, and Compression Profiles](#latency-throughput-and-compression-profiles)
9. [Installation and Deployment](#installation-and-deployment)
   - [Prerequisites](#prerequisites)
   - [Python Environment Setup](#python-environment-setup)
   - [Chrome Extension Deployment](#chrome-extension-deployment)
   - [Automated Service Orchestration](#automated-service-orchestration)
10. [Python API and CLI Reference](#python-api-and-cli-reference)
    - [Python API Usage Examples](#python-api-usage-examples)
    - [Online Streaming Belief Tracker](#online-streaming-belief-tracker)
    - [Command Line Interface (CLI)](#command-line-interface-cli)
11. [Repository Structure](#repository-structure)
12. [Verification and Test Suite](#verification-and-test-suite)
13. [Citation](#citation)
14. [License](#license)

---

## Executive Summary

State-of-the-art conversational AI agents and Retrieval-Augmented Generation (RAG) frameworks rely on persistent vector memory stores (Chroma, Pinecone, Weaviate, Milvus, FAISS) to maintain context across extended multi-turn interactions. However, virtually all contemporary memory architectures are fundamentally **append-only**. When facts change dynamically over time (for example, a user changing their location, job title, infrastructure credentials, or technical stack preferences), standard semantic embeddings retrieve both past and present assertions simultaneously. The underlying Large Language Model (LLM) cannot determine chronological precedence or logical supersession, resulting in conflicting generations, compromised reasoning chains, and silent hallucinations.

**ALETHEX** solves this challenge through a formal neuro-symbolic framework. Rather than storing unanchored text chunks, ALETHEX extracts atomic factual claims, resolves temporal intervals via Allen's Interval Algebra, links entities across turns using semantic DBSCAN clustering, and maintains an explicit, directed **Dynamic Temporal Belief Graph (DTBG)**. Contradictory statements are actively identified via staged Natural Language Inference (NLI) and pruned or annotated before context enters the LLM context window.

The system is deployed as an end-to-end ecosystem:
- A high-throughput Python engine providing sub-4ms belief reconciliation.
- A zero-latency Manifest V3 Chrome Extension offering real-time memory surveillance across ChatGPT, Claude.ai, Google Gemini, Microsoft Copilot, DeepSeek, Perplexity AI, Grok, and local WebUIs.
- Plug-and-play middleware adapters for LangChain, LlamaIndex, Open WebUI, Model Context Protocol (MCP), and OpenAI-compatible API reverse proxies.

---

## Theoretical Foundations

### The Append-Only Memory Failure Mode

Consider two statements ingested across an ongoing interaction:

$$\sigma_1: \text{"Alice works as a senior software engineer at Google in London."} \quad (t_1 = \text{2024-01-15})$$

$$\sigma_2: \text{"Alice relocated to Tokyo to lead research at Anthropic."} \quad (t_2 = \text{2024-07-01})$$

Under standard dense vector retrieval with cosine distance metric $d_{\cos}(q, k)$, a query $q = \text{"Where does Alice work and live?"}$ yields near-identical high similarity scores for both $\sigma_1$ and $\sigma_2$:

$$d_{\cos}(q, \sigma_1) \approx 0.89, \quad d_{\cos}(q, \sigma_2) \approx 0.91$$

Because both vectors reside above top-$k$ retrieval thresholds, both statements are concatenated into the prompt context. Standard transformer attention mechanisms fail to guarantee chronological priority without structured metadata, often generating syntheses such as:

$$\hat{y}: \text{"Alice works at Google in London while leading research at Anthropic in Tokyo."}$$

This phenomenon represents a categorical failure of temporal consistency in parametric and semi-parametric memory systems.

### Mathematical Formulation of Temporal Claims

ALETHEX represents verified knowledge as an atomic factual quadruple:

$$c = \langle s, p, o, \tau \rangle$$

Where:
- $s \in \mathcal{E}$: Subject entity identifier.
- $p \in \mathcal{P}$: Predicate relation describing the semantic action or state.
- $o \in \mathcal{E} \cup \mathcal{V}$: Object entity or scalar attribute value.
- $\tau = [t_{\text{start}}, t_{\text{end}}] \subset \mathcal{T}$: Normalized continuous or semi-bounded temporal validity interval, where $t_{\text{start}}, t_{\text{end}} \in \mathbb{R} \cup \{-\infty, +\infty\}$.

Each claim $c$ is assigned an empirical extraction confidence score $\gamma(c) \in [0, 1]$ and an anchor document identifier $d \in \mathcal{D}$.

### Allen's Interval Algebra

Temporal compatibility between two claims $c_i = \langle s_i, p_i, o_i, \tau_i \rangle$ and $c_j = \langle s_j, p_j, o_j, \tau_j \rangle$ sharing identical or coreferent subjects ($s_i \equiv s_j$) is evaluated using James Allen's 13 elementary temporal interval relations:

```text
Relation            Symbol      Condition on Intervals [s1, e1] and [s2, e2]
---------------------------------------------------------------------------------
before              <           e1 < s2
after               >           s1 > e2
meets               m           e1 = s2
met-by              mi          s1 = e2
overlaps            o           s1 < s2 < e1 < e2
overlapped-by       oi          s2 < s1 < e2 < e1
starts              s           s1 = s2 and e1 < e2
started-by          si          s1 = s2 and e1 > e2
during              d           s1 > s2 and e1 < e2
contains            di          s1 < s2 and e1 > e2
finishes            f           e1 = e2 and s1 > s2
finished-by         fi          e1 = e2 and s1 < s2
equals              =           s1 = s2 and e1 = e2
```

Using interval projection:
- **Disjoint Intervals ($\tau_i < \tau_j$ or $\tau_i > \tau_j$):** Factual divergence may indicate legitimate historical change rather than mutual contradiction.
- **Overlapping or Identical Intervals ($\tau_i \cap \tau_j \neq \emptyset$):** Incompatible predicates or objects indicate a factual contradiction.

### Dynamic Temporal Belief Graph (DTBG)

The complete conversational or corpus belief state is represented as a directed multigraph:

$$G = (V, E, \mathcal{T}_V, \mathcal{R}_E)$$

- **Vertices ($V$):** Entity nodes $v_e \in V_{\mathcal{E}}$ and Claim nodes $v_c \in V_{\mathcal{C}}$.
- **Edges ($E$):**
  - Membership edges: $e_{\text{arg}} = (v_e, v_c)$ connecting entities to their participating claims.
  - Relational edges between claims: $e_{\text{rel}} = (v_{c_i}, v_{c_j}, r)$ where relation type $r \in \{\text{consistent}, \text{supersedes}, \text{contradicts}, \text{neutral}\}$.
- **Edge Metadata:** Timestamp, NLI probability distribution, Allen relation tag, and classified confidence.

When a new claim $c_j$ supersedes an earlier claim $c_i$ ($r = \text{supersedes}$), $c_i$ is transitioned to status $\text{INACTIVE}$ ($c_i.\text{active} \leftarrow \text{False}$), while remaining in graph storage for historical auditability.

### Consistency Index (CI) Formulation

ALETHEX defines a deterministic metric to evaluate the internal factual coherence of individual entities and entire multi-document corpora.

#### Entity Consistency Index
For a canonical entity $e \in \mathcal{E}$, let $C(e)$ denote the set of active claims asserting attributes of $e$, and let $E_{\text{conflict}}(e) \subseteq C(e) \times C(e)$ represent the set of active contradiction edges incident upon claims associated with $e$:

$$CI(e) = 1.0 - \frac{2 \cdot \sum_{(c_i, c_j) \in E_{\text{conflict}}(e)} \min\big(\gamma(c_i), \gamma(c_j)\big)}{\sum_{c \in C(e)} \gamma(c) + \epsilon}$$

Where $\gamma(c)$ is the confidence of claim $c$ and $\epsilon = 10^{-6}$ prevents zero-division for sparse entities. The value is strictly bounded: $CI(e) \in [0.0, 1.0]$.

#### Corpus Consistency Index
Across the entire belief graph $G$, the global Corpus Consistency Index aggregates over all $N$ canonical entities:

$$CI(G) = \frac{1}{|V_{\mathcal{E}}|} \sum_{e \in V_{\mathcal{E}}} CI(e)$$

A value of $CI(G) = 1.000$ indicates zero detected contradictions across all entities, while lower scores quantitatively signal degree of factual decay.

### Staleness Scoring Formulation

To penalize outdated information that has not been explicitly overwritten, ALETHEX computes an exponential decay staleness score $S(c)$ for each claim $c$ relative to an audit reference timestamp $t_{\text{ref}}$:

$$S(c) = 1.0 - \exp\left(-\lambda \cdot \max\left(0, \frac{t_{\text{ref}} - t_{\text{end}}}{\Delta_{\text{threshold}}}\right)\right)$$

Where:
- $\Delta_{\text{threshold}}$ represents the configured staleness threshold in days (default: 180 days).
- $\lambda$ is a decay coefficient governing decay sensitivity.
- Claims whose validity period ended long before $t_{\text{ref}}$ approach a staleness score of $1.0$.

---

## System Architecture

### Pipeline Data Flow

The ALETHEX core engine processes unconstrained text through an eight-stage neuro-symbolic pipeline:

```text
[Input Documents / Live Chat Stream]
                |
                v
+-----------------------------------------------------------+
| Stage 1: Document Ingestion & Turn Normalization          |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 2: OpenIE & LLM Claim Extraction                    |
|   - Extract atomic claims: <Subject, Predicate, Object>   |
|   - Confidence scoring & metadata anchoring               |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 3: Cross-Document Coreference & Entity Linking      |
|   - Heuristic within-document mention resolution          |
|   - Sentence-BERT embedding generation                    |
|   - DBSCAN semantic clustering for canonical entities     |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 4: TIMEX3 Normalization & Interval Resolution       |
|   - Regex & HeidelTime temporal expression parsing        |
|   - Reference date anchoring to continuous intervals      |
|   - Allen's 13 interval relation evaluation               |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 5: Staged NLI Verification & Hybrid Classification  |
|   - Fast entity-based candidate pair pruning              |
|   - Cross-Encoder NLI (DeBERTa-v3 / ALETHEX-Mini)         |
|   - Optional LLM-as-a-Judge arbitration for ambiguities   |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 6: Dynamic Temporal Belief Graph (DTBG) Assembly    |
|   - MultiDiGraph construction                             |
|   - Active belief state maintenance & supersession        |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 7: Quantitative Consistency & Contradiction Audits  |
|   - CI calculation (Entity & Corpus levels)               |
|   - Staleness score computation & GraphML export          |
+-----------------------------------------------------------+
                |
                v
+-----------------------------------------------------------+
| Stage 8: Context-Aware Memory Reconciliation              |
|   - Drop / Annotate modes for RAG prompts                 |
|   - Direct injection into LangChain / LlamaIndex / MCP    |
+-----------------------------------------------------------+
```

### Stage 1: Document Ingestion and Normalization
Text sources (session dialogue turns, JSONL corpora, documents) are ingested through `RawDocument` schemas. Each document is validated for source identifier, speaker/author attribution, document-level creation timestamp ($t_{\text{doc}}$), and raw content text.

### Stage 2: OpenIE and LLM Claim Extraction
Unstructured sentences are deconstructed into atomic factual units. ALETHEX implements a dual extraction architecture:
- **Deterministic OpenIE (`OpenIEExtractor`):** High-speed dependency parsing and rule-based verb-argument extraction, operating at zero inference cost.
- **Parametric LLM Extractor (`LLMExtractor`):** Few-shot structured prompt extraction leveraging local or remote models to parse complex idioms, implicit relationships, and subordinate clauses.
- **Claim Merger (`ClaimMerger`):** Deduplicates and fuses overlapping extractions, prioritizing high-confidence representations.

### Stage 3: Coreference Resolution and Entity Linking
To link statements referencing the same real-world entity:
1. **Within-Document Coreference (`WithinDocCoref`):** Resolves third-person pronouns (*"he", "she", "it", "they"*) to nearest antecedent noun phrases.
2. **Cross-Document Entity Linking (`EntityLinker`):** Generates dense vector representations of entity mentions using Sentence-BERT (`all-MiniLM-L6-v2`).
3. **Density-Based Spatial Clustering (DBSCAN):** Groups mentions into canonical entity clusters based on cosine distance threshold $\epsilon = 0.35$ and minimum cluster size. Entities with high similarity are assigned a single canonical identifier $e^* \in \mathcal{E}$.

### Stage 4: TIMEX3 Normalization and Temporal Resolution
Natural language temporal expressions (such as *"last Tuesday"*, *"since August 2022"*, *"for three months"*, *"currently"*) are mapped into standard TIMEX3 representations.
- Anchored to the document creation timestamp $t_{\text{doc}}$.
- Converted into continuous interval tuples $\tau = [t_{\text{start}}, t_{\text{end}}]$.
- Evaluated against Allen's interval algebra matrix to classify temporal intersection topology before semantic comparison.

### Stage 5: Staged NLI Verification and Hybrid Classification
Pairwise comparison of all claims in an $N$-claim corpus has complexity $\mathcal{O}(N^2)$. ALETHEX applies a multi-stage filtering hierarchy:
1. **Candidate Pair Generation (`PairGenerator`):** Pairs are evaluated only if they share a canonical entity or exhibit semantic topical overlap. Disjoint entities are skipped immediately.
2. **Cross-Encoder NLI (`NLIClassifier`):** Evaluates candidate pairs $(c_1, c_2)$ through a cross-encoder scoring ternary logits:

$$\mathbf{p} = \text{Softmax}\big(\mathbf{W} \cdot \text{Encoder}(c_1 \oplus [\text{SEP}] \oplus c_2)\big) \in \mathbb{R}^3$$

Classifying relations into:
- **Entailment:** Consistent assertions ($c_1 \implies c_2$).
- **Contradiction:** Conflicting assertions ($c_1 \land c_2 \implies \bot$).
- **Neutral:** Independent or unrelated assertions.

3. **Temporal Synthesis:** If a contradiction is detected, the temporal interval resolver determines whether the contradiction is **Simultaneous** (true factual conflict) or **Sequential** (historical supersession where the later claim invalidates the former).
4. **LLM Judge Arbitration (`LLMJudge`):** Pairs with borderline NLI confidence scores ($0.45 < p < 0.70$) can be optionally escalated to an LLM evaluator with explicit prompt instructions to resolve subtle nuance.

### Stage 6: Temporal Belief Graph Assembly and Pruning
The graph builder (`BeliefGraph`) ingests canonical entities, claims, and classified relations to construct a `networkx.MultiDiGraph`.
- **Node Properties:** Text, canonical subject, predicate, object, interval, confidence, and active boolean flag.
- **Edge Properties:** Relation type, confidence, temporal classification, and provenance index.
- **Supersession Resolution:** When claim $c_2$ supersedes $c_1$, $c_1.\text{active}$ is toggled to `False`. Subsequent retrieval queries exclude $c_1$, preventing outdated context injection.

### Stage 7: Quantitative Consistency Auditing and Reporting
The scoring module computes:
- Canonical Entity Consistency Indices ($CI(e)$).
- Global Corpus Consistency Index ($CI(G)$).
- Detailed contradiction diagnostic reports including conflicting claim IDs, raw text snippets, and temporal intervals.
- Staleness audit reports identifying claims exceeding validity retention thresholds.

### Stage 8: Context-Aware Memory Reconciliation
During memory retrieval for downstream generation, the reconciliation layer filters raw memory buffers:
- **Drop Mode:** Strips out all inactive, superseded, and conflicting claims, passing only mathematically consistent facts to the LLM.
- **Annotate Mode:** Retains statements but injects explicit warnings directly into context headers, instructing the LLM to prioritize the latest assertion.

---

## Neural Model Architecture and Optimization Engine

The inference engine of ALETHEX is designed for ultra-low latency execution on resource-constrained consumer hardware while maintaining the reasoning precision of frontier cross-encoders.

### Teacher Cross-Encoder (DeBERTa-v3)

The primary teacher model is based on **DeBERTa-v3** (*Decoding-enhanced BERT with Disentangled Attention*):
- **Disentangled Attention:** Each input token is represented by two separate vectors: one for content and one for relative position. The self-attention matrix calculates attention weights across four decoupled components: content-to-content, content-to-position, position-to-content, and position-to-position.
- **Joint Premise-Hypothesis Encoding:** Unlike dual-encoder bi-encoders that process claims independently, the cross-encoder ingests the concatenated sequence:
  $$\mathbf{x} = \text{[CLS]} \circ c_i \circ \text{[SEP]} \circ c_j \circ \text{[SEP]}$$
  Allowing all tokens of claim $c_i$ to attend directly to all tokens of claim $c_j$ across all 24 Transformer layers.

### Student Architecture (ALETHEX-Mini, 22.7M Parameters)

To serve high-concurrency client streams, ALETHEX distills knowledge from the 435M-parameter DeBERTa-v3 teacher into a highly compacted student architecture:
- **Base Architecture:** MiniLM-6L-384H (6 Transformer encoder layers, hidden dimension $d_h = 384$, 12 self-attention heads, intermediate feed-forward size $d_{ff} = 1536$).
- **Parameter Footprint:** 22,713,219 parameters (representing a $19.15\times$ reduction in parameter volume).
- **Classification Head:** Linear projection layer $\mathbf{W}_{\text{head}} \in \mathbb{R}^{3 \times 384}$ mapping the pooled $\text{[CLS]}$ embedding directly into the ternary relation logits $\mathbf{z}_s \in \{\text{Contradiction}, \text{Entailment}, \text{Neutral}\}$.

### Knowledge Distillation Formulation

Training minimizes a combined loss function that balances soft dark-knowledge transfer from the teacher logits $\mathbf{z}_t$ with ground-truth supervised classification loss $y \in \{0, 1, 2\}$:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{\mathbf{z}_s}{T}\right), \sigma\left(\frac{\mathbf{z}_t}{T}\right)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(\mathbf{z}_s, y)$$

Where:
- $\sigma(\cdot)$ is the softmax function.
- $T = 2.0$ is the distillation temperature parameter, smoothing probability distributions over subtle relation boundaries.
- $\alpha = 0.5$ balances matching teacher distribution softness against ground-truth categorical cross-entropy.
- $\mathcal{L}_{\text{KL}}(P, Q) = \sum_k P(k) \log\left(\frac{P(k)}{Q(k)}\right)$ is the Kullback-Leibler divergence.

Optimization is governed by AdamW ($\eta = 3 \times 10^{-5}$, weight decay $0.01$) with a cosine learning rate scheduler and $10\%$ linear warmup steps over 3 epochs.

### Dynamic INT8 Quantization Architecture

For real-time CPU deployment, ALETHEX applies dynamic 8-bit integer quantization (`onnxruntime.quantization.quantize_dynamic`):
- **Weights:** Quantized offline from 32-bit floating-point ($\text{FP32}$) to signed 8-bit integers ($\text{INT8}$) via symmetric per-channel quantization:
  $$\mathbf{W}_{\text{int8}} = \text{clip}\left(\left\lfloor \frac{\mathbf{W}_{\text{fp32}}}{S_w} \right\rceil, -128, 127\right)$$
- **Activations:** Dynamically quantized at runtime using asymmetric per-tensor scaling factors $S_a$ and zero-points $Z_a$:
  $$A_{\text{int8}} = \text{clip}\left(\left\lfloor \frac{A_{\text{fp32}}}{S_a} \right\rceil + Z_a, 0, 255\right)$$
- **Execution:** Heavy matrix multiplications are dispatched directly to hardware-accelerated integer instruction sets (`AVX-512 VNNI`, `AVX2`, `ARM Neon`), bypassing floating-point arithmetic bottlenecks entirely.

### Dual-Layer Neural Architecture Diagram

```text
               +-------------------------------------------------------------+
               |  Concatenated Claim Pair: [CLS] c1 [SEP] c2 [SEP]           |
               +-------------------------------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
    +----------------------------------+   +----------------------------------+
    | Teacher Model: DeBERTa-v3-Large  |   | Student Model: ALETHEX-Mini      |
    | - 24 Layers, 1024 Dim, 16 Heads  |   | - 6 Layers, 384 Dim, 12 Heads    |
    | - 435M Parameters (FP32)         |   | - 22.7M Parameters (INT8 ONNX)   |
    | - Disentangled Attention         |   | - Dynamic Quantized MatMul       |
    +----------------------------------+   +----------------------------------+
                     |                                 |
                     | Soft Logits (zt)                | Student Logits (zs)
                     v                                 v
        +---------------------------------------------------------------+
        | Loss Function:                                                |
        | L = alpha * T^2 * KL(zs/T || zt/T) + (1 - alpha) * CE(zs, y)  |
        +---------------------------------------------------------------+
                                      |
                                      v
        +---------------------------------------------------------------+
        | Output Distribution:                                          |
        | [ Contradiction: 0.982 | Entailment: 0.011 | Neutral: 0.007 ] |
        +---------------------------------------------------------------+
```

---

## End-to-End LLM Workflow Walkthrough: Production Case Study

To illustrate how ALETHEX operates in live environments, the following walkthrough demonstrates the complete resolution lifecycle across a multi-turn software architecture dialogue.

### Enterprise Scenario: Multi-Turn Infrastructure Evolution

A software engineer collaborates with an AI assistant over three months to plan a microservice infrastructure.

```text
Turn 1 (Timestamp: 2024-03-01):
User: "Our core API backend is deployed on AWS RDS PostgreSQL in us-east-1 with Redis caching."
Assistant: "Understood. I will remember that your backend is PostgreSQL on AWS RDS in us-east-1."

Turn 2 (Timestamp: 2024-04-15):
User: "We decided to migrate completely away from AWS RDS to a self-managed CockroachDB cluster
       on GCP because we need multi-region active-active distributed transactions."
Assistant: "Noted. CockroachDB on GCP is now your primary distributed database."

Turn 3 (Timestamp: 2024-05-30):
User: "Write the production database connection initialization module and health check script."
```

---

### Step 1: Raw Turn Ingestion & Atomic Claim Extraction

When Turn 1 and Turn 2 are ingested, the extraction subsystem deconstructs each sentence into formal claims:

```json
[
  {
    "claim_id": "claim_001",
    "turn_id": "turn_1",
    "subject": "core API backend",
    "predicate": "uses database engine",
    "object": "PostgreSQL on AWS RDS",
    "confidence": 0.94,
    "source_doc_timestamp": "2024-03-01T00:00:00Z",
    "temporal_expression": "currently",
    "validity_interval": ["2024-03-01T00:00:00Z", "infinity"]
  },
  {
    "claim_id": "claim_002",
    "turn_id": "turn_2",
    "subject": "core API backend",
    "predicate": "uses database engine",
    "object": "CockroachDB on GCP",
    "confidence": 0.97,
    "source_doc_timestamp": "2024-04-15T00:00:00Z",
    "temporal_expression": "migrated completely away",
    "validity_interval": ["2024-04-15T00:00:00Z", "infinity"]
  }
]
```

---

### Step 2: Canonical Entity Linking via SBERT and DBSCAN

1. **Mention Vectors:** Mentions `"core API backend"` (Turn 1) and `"we"` / implicit subject (Turn 2) are mapped into 384-dimensional dense vectors using Sentence-BERT.
2. **Clustering:** DBSCAN groups both subject mentions into a unified canonical entity:
   $$e^*_{\text{backend\_db}} = \text{"ent\_cluster\_42"}$$
3. **Candidate Pairing:** Because `claim_001` and `claim_002` share canonical subject $e^*_{\text{backend\_db}}$ and compatible predicates (`uses database engine`), they are scheduled for pairwise relational evaluation.

---

### Step 3: Temporal Interval Normalization and Allen Calculus

The Interval Resolver evaluates the temporal relationship between:
- $\tau_1 = [2024\text{-}03\text{-}01, +\infty)$
- $\tau_2 = [2024\text{-}04\text{-}15, +\infty)$

1. **Topology:** Both intervals have open upper bounds, creating an overlapping region $[2024\text{-}04\text{-}15, +\infty)$.
2. **Chronological Anchor:**
   $$t_{\text{start}}(\tau_1) < t_{\text{start}}(\tau_2)$$
3. **Allen Relation Output:** Evaluated as `overlaps` ($o$) with sequential precedence ($c_1 \text{ preceded } c_2$).

---

### Step 4: Staged NLI Verification and Logit Distribution

The candidate pair is tokenized and fed into the `ALETHEX-Mini` INT8 cross-encoder:

$$\text{Premise: "We migrated completely away from AWS RDS to CockroachDB on GCP on 2024-04-15."}$$
$$\text{Hypothesis: "Our core API backend uses PostgreSQL on AWS RDS."}$$

**Cross-Encoder Logit Output:**
- $z_0 \text{ (Contradiction)} = +4.82 \implies \mathbf{p}_{\text{contradiction}} = \mathbf{0.982}$
- $z_1 \text{ (Entailment)} = -1.14 \implies p_{\text{entailment}} = 0.011$
- $z_2 \text{ (Neutral)} = -1.58 \implies p_{\text{neutral}} = 0.007$

**Result:** High-confidence mutual exclusivity detected. Combined with sequential temporal ordering ($t_1 < t_2$), the relationship is classified as **`supersedes`**.

---

### Step 5: Dynamic Belief Graph Mutation and Supersession

The belief graph executes the state transition:

```text
[Entity: core API backend]
        |
        +---> (Claim 001) [PostgreSQL on AWS RDS] (STATUS: INACTIVE / SUPERSEDED)
        |       ^
        |       |-- [EDGE: SUPERSEDES | Conf: 0.982 | Allen: overlaps]
        |       |
        +---> (Claim 002) [CockroachDB on GCP]    (STATUS: ACTIVE)
```

1. Edge `(claim_002, claim_001, type="supersedes")` is written to `G`.
2. Node `claim_001.active` is flipped to `False`.
3. Entity Consistency Index remains clean: $CI(e^*_{\text{backend\_db}}) = 1.000$ (conflict successfully reconciled).

---

### Step 6: Memory Reconciliation and Context Injection

When Turn 3 arrives:
*"Write the production database connection initialization module and health check script."*

The standard vector store retrieves both Turn 1 and Turn 2. The ALETHEX reconciliation layer intercepts the retrieved entries:

```python
# Before ALETHEX: Both chunks passed to prompt
[Retrieved Chunk 1] (Turn 1): "Our core API backend is deployed on AWS RDS PostgreSQL..."
[Retrieved Chunk 2] (Turn 2): "We decided to migrate completely away to CockroachDB on GCP..."

# After ALETHEX Filtering (Drop Mode):
[Filtered Prompt Context]:
"[2024-04-15] The core API backend uses CockroachDB on GCP (Active Production Database)."
# Chunk 1 is suppressed because its graph node is marked INACTIVE.
```

---

### Comparative Output Analysis: Standard LLM vs ALETHEX-Guarded LLM

#### Without ALETHEX (Standard Append-Only Memory Failure)
The LLM receives both chunks in context, becomes confused by chronological ambiguity, and hallucinates a conflicting hybrid implementation:

```python
# HALLUCINATED OUTPUT FROM UNGUARDED LLM:
import psycopg2
import os

def get_db_connection():
    # Connecting to AWS RDS PostgreSQL
    conn = psycopg2.connect(
        host="rds.us-east-1.amazonaws.com",
        database="production",
        user=os.environ.get("RDS_USER"),
        password=os.environ.get("RDS_PASSWORD")
    )
    return conn

# Wait, also initialize CockroachDB cluster endpoint on GCP...
def get_cockroach_client():
    ...
```
*Result: Production failure. Code attempts to establish connections to a decommissioned AWS database.*

#### With ALETHEX (Reconciled Context)
The LLM receives only verified active assertions and generates a clean, correct, single-target implementation:

```python
# RECONCILED OUTPUT FROM ALETHEX-GUARDED LLM:
import psycopg2
import ssl
import os

def init_cockroach_pool():
    """Initializes connection pool to self-managed CockroachDB cluster on GCP."""
    return psycopg2.connect(
        host=os.environ.get("COCKROACH_GCP_HOST", "cockroach-cluster.gcp.internal"),
        port=26257,
        database="production",
        user=os.environ.get("COCKROACH_USER"),
        sslmode="verify-full",
        sslrootcert="/etc/certs/ca.crt"
    )

def health_check():
    conn = init_cockroach_pool()
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        return {"status": "healthy", "database": "CockroachDB-GCP"}
```
*Result: Flawless execution with zero obsolete references.*

---

### Real-Time User Interface Visualization

In the user's browser, the ALETHEX Chrome Extension renders the live state directly over ChatGPT or Claude:

```text
+-------------------------------------------------------------------------------+
| ChatGPT - Architecture Planning                                               |
+-------------------------------------------------------------------------------+
|                                                                               |
| User: Write the production database connection initialization module...      |
|                                                                               |
| Assistant:                                                                    |
| Here is the production CockroachDB connection pool configuration on GCP...    |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | ALETHEX TRUTH HUD (Active)                                            |   |
|   | Verified Claims: 14 | Contradictions Resolved: 1 | CI Score: 1.000    |   |
|   | [Open Audit Drawer]  [Force Re-scan]                                  |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
+-------------------------------------------------------------------------------+
```

When the user clicks **[Open Audit Drawer]**, the inspection modal displays:

```text
=================================================================================
ALETHEX AUDIT LOG: ENTITY [core API backend]
=================================================================================
[2024-03-01] [SUPERSEDED] Uses PostgreSQL on AWS RDS (Conf: 0.94)
      |
      +---> [Superseded by turn_2 at 2024-04-15 | NLI Conf: 0.982 | Allen: overlaps]
      |
[2024-04-15] [ACTIVE]     Uses CockroachDB on GCP (Conf: 0.97)
=================================================================================
Consistency Index: 1.0000 | Active Conflicts: 0 | Staleness: 0.00
```

---

## Client Applications and User Interfaces

### Universal Chrome Extension (Manifest V3)

The ALETHEX Chrome Extension (`alethex-guard-v1.2`) provides zero-latency client-side truth monitoring across major web AI interfaces.

#### Key Characteristics
- **Manifest V3 Compliant:** Uses lightweight service workers (`background.js`) and high-efficiency DOM mutation observers (`content.js`).
- **Target Platform Auto-Detection:** Automatically activates specialized observer rules on:
  - OpenAI ChatGPT (`chatgpt.com`, `chat.openai.com`)
  - Anthropic Claude (`claude.ai`)
  - Google Gemini (`gemini.google.com`)
  - Microsoft Copilot (`copilot.microsoft.com`)
  - DeepSeek (`chat.deepseek.com`)
  - Perplexity AI (`perplexity.ai`)
  - Grok / X (`x.com/i/grok`, `grok.com`)
  - Local AI Interfaces (`localhost:3000`, `localhost:8080`, `127.0.0.1`)
- **Real-Time Memory HUD:** A floating glassmorphic heads-up display in the upper right corner displaying:
  - Live count of extracted factual beliefs.
  - Number of flagged contradictions.
  - Verification engine connection status (Local Engine vs Offline Heuristic).
  - Quick action buttons (Audit Memory Drawer, Force Rescan, Platform Mode).
- **Inline Contradiction Banners:** Automatically attaches warning tags with detailed conflict rationales directly below inconsistent generation blocks in the chat thread.
- **Inspection Drawer Modal:** An interactive slide-out panel allowing users to inspect extracted subject-predicate-object triples, temporal timestamps, and contradiction graphs.
- **Dual-Engine Execution:**
  - **Local Heuristic Fallback:** Executes client-side regex parsing and contradiction heuristics with zero network overhead.
  - **Live Backend Synchronization:** Seamlessly pairs with the local Python engine (`http://localhost:8000`) for full deep neural NLI inference.

### Native Desktop Companion

`alethex_desktop.py` provides a cross-platform desktop monitoring utility built with Tkinter:
- Displays continuous system-wide status, memory usage, and background service states.
- 1-Click launcher buttons for the Chrome extension folder, web dashboard, and test suites.
- Live system tray integration and service logs.

### Production Landing Page and Documentation Hub

A fully responsive, production-ready landing page is provided at `index.html`:
- Interactive memory conflict simulator demonstrating append-only RAG failures versus ALETHEX reconciled memory.
- Platform feature matrix and architecture breakdown.
- Direct 1-click ZIP download for the Chrome extension (`dist/alethex-extension-v1.0.0.zip`).
- Installation and configuration instructions.

---

## Enterprise Integrations and Middleware Gateways

ALETHEX is designed as a modular middleware layer that can be integrated at any stage of an enterprise AI stack:

### Model Context Protocol (MCP) Server
- **Module:** `alethex.integrations.mcp_server`
- **Protocol:** Standard JSON-RPC 2.0 via standard input/output (`stdio`).
- **Compatible Clients:** Claude Desktop, Claude Code, Cursor IDE, Windsurf.
- **Exposed Tools:**
  1. `audit_memory_consistency`: Accepts an array of timestamped memory statements and returns full contradiction and staleness reports with calculated CI scores.
  2. `filter_superseded_facts`: Accepts memory entries and returns an ordered list of only valid, non-superseded assertions.
  3. `query_belief_graph`: Performs graph-traversal queries across extracted entity nodes and claims.

#### MCP Configuration (`claude_desktop_config.json` / `mcp.json`)
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

### Universal OpenAI-Compatible Reverse Proxy
- **Module:** `alethex.integrations.openai_proxy`
- **Description:** A high-concurrency FastAPI gateway exposing standard OpenAI endpoints (`POST /v1/chat/completions`).
- **Functionality:** Intercepts incoming chat completions requests, extracts user and system messages, reconciles memory statements using the ALETHEX engine, rewrites prompt context to eliminate contradictions, and forwards the cleaned request to any upstream provider (OpenAI, Anthropic, Ollama, vLLM, LM Studio).

```bash
# Start proxy pointing to local Ollama instance
python -m alethex.integrations.openai_proxy --port 8000 --upstream http://localhost:11434/v1

# Start proxy pointing to OpenAI API
python -m alethex.integrations.openai_proxy --port 8000 --upstream https://api.openai.com/v1
```

### Open WebUI Pipeline Filter
- **Module:** `alethex.integrations.openwebui_filter`
- **Description:** A native Open WebUI Python Function filter.
- **Usage:** In Open WebUI, navigate to `Admin Settings -> Functions`, create a new filter, and paste the module code. Contradictory previous turns are resolved before local models (e.g. Llama-3, Qwen) synthesize responses.

### Universal RAG Middleware
- **Module:** `alethex.integrations.universal_rag`
- **Description:** Framework-agnostic middleware that wraps any vector store retriever function or post-processes retrieved document chunks.
- **Modes:**
  - `mode="drop"`: Suppresses invalid and stale chunks entirely.
  - `mode="annotate"`: Appends validity warning banners to chunk headers.
  - `mode="context_string"`: Concatenates valid chunks into a unified string.

### LangChain Memory Adapter
- **Module:** `alethex.integrations.langchain_memory`
- **Class:** `AlethexReconciledMemory`
- **Description:** Subclasses LangChain's `BaseMemory` interface. Ingests conversation turns, tracks entity belief states, and supplies reconciled memory variables into standard LangChain execution chains.

### LlamaIndex Consistency Postprocessor
- **Module:** `alethex.integrations.llamaindex_filter`
- **Class:** `AlethexConsistencyFilter`
- **Description:** A LlamaIndex `BaseNodePostprocessor` that filters retrieved `NodeWithScore` objects, guaranteeing that context injected into synthesized query engines is free of temporal contradictions.

### Automated Environment Detector and Attacher
- **Module:** `alethex.integrations.auto_attach`
- **Description:** Inspects the local operating system (Windows, macOS, Linux) to automatically discover installed AI applications (Claude Desktop, Cursor, Windsurf, Ollama, LM Studio, Open WebUI) and injects required MCP server configurations and network proxies with a single command:

```bash
python -m alethex.integrations.auto_attach --apply
```

### FastAPI REST Service and Visualization Dashboard
- **Module:** `alethex.dashboard.app` / `alethex.api`
- Exposes REST endpoints for external microservices:
  - `POST /api/audit`: Comprehensive document consistency evaluation.
  - `POST /api/filter`: RAG context reconciliation endpoint.
  - `GET /api/status`: Engine health and active model metadata.
  - `GET /`: Interactive web dashboard rendering GraphML belief networks.

---

## Model Distillation and Optimization Benchmarks

To achieve real-time throughput during live conversation streaming, the ALETHEX inference pipeline was optimized through knowledge distillation and dynamic quantization.

### Empirical Relation Classification Benchmarks

Evaluation performed on a verified test corpus of temporal belief pairs categorized into three classes: **Consistent**, **Superseded**, and **Conflicting**:

| Architecture | 3-Way Accuracy | Macro Precision | Macro Recall | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| Naive Semantic NLI (Unanchored, No Temporal Logic) | 0.512 | 0.491 | 0.474 | 0.482 |
| Standard DeBERTa-v3-Large (Without Temporal Interval Calculus) | 0.638 | 0.620 | 0.608 | 0.614 |
| **ALETHEX Staged Pipeline (Neuro-Symbolic + Allen Interval Logic)** | **0.880** | **0.871** | **0.860** | **0.865** |
| *Human Inter-Annotator Agreement Upper Bound* | *0.957* | *0.936* | *0.932* | *0.934* |

### Latency, Throughput, and Compression Profiles

Inference benchmarks evaluated on a single consumer CPU core (Intel Core i7 / AMD Ryzen equivalent):

| Model Backend | Model Size | Latency per Pair | Throughput | Acceleration | Compression |
|:---|:---:|:---:|:---:|:---:|:---:|
| **DeBERTa-v3-Large (FP32 PyTorch)** | 1,740.0 MB | 70.52 ms | 14.2 pairs/sec | 1.00x | 1.00x |
| **ALETHEX-NLI-INT8 (Quantized ONNX)** | 164.3 MB | 18.02 ms | 55.5 pairs/sec | **3.91x** | **10.59x** |
| **ALETHEX-Mini (Distilled Student, 22.7M)** | 86.7 MB | **3.19 ms** | **313.3 pairs/sec** | **22.11x** | **20.07x** |

The distilled `ALETHEX-Mini` student model reduces per-pair latency to **3.19 milliseconds**, enabling live evaluation of conversational turns within interactive streaming budgets.

---

## Installation and Deployment

### Prerequisites

- **Python:** Version 3.10, 3.11, or 3.12.
- **Operating System:** Windows 10/11, macOS (Apple Silicon / Intel), or Linux (Ubuntu 20.04+).
- **Web Browser:** Google Chrome, Brave, Microsoft Edge, or any Chromium-based browser supporting Manifest V3.

### Python Environment Setup

Clone the repository and install the package in editable mode:

```bash
# Clone the repository
git clone https://github.com/codewithyug06/Alethex.git
cd Alethex

# Navigate to the Python core engine
cd alethex

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install package dependencies
pip install -e .
```

### Chrome Extension Deployment

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable the **Developer mode** toggle in the top-right corner.
3. Click the **Load unpacked** button in the top-left menu.
4. Select the `alethex-guard-v1.2` directory inside this repository.
5. The **ALETHEX AI Guard** card will appear in your extensions list.
6. Open any supported platform (e.g., [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai)). The floating ALETHEX Truth HUD will appear automatically.

### Automated Service Orchestration

On Windows environments, pre-configured batch scripts allow 1-click execution:

- `launch_all.bat`: Orchestrates the FastAPI verification service, OpenAI proxy gateway, and desktop companion concurrently.
- `run_desktop.bat`: Launches the native Tkinter desktop monitoring companion.
- `open_extension_folder.bat`: Automatically copies the extension folder path to clipboard and reveals it in Windows Explorer for fast installation.

---

## Python API and CLI Reference

### Python API Usage Examples

#### Reconciling Retrieved Context in a Custom RAG Pipeline

```python
import alethex

# Raw chunks retrieved from any vector database
retrieved_memories = [
    {
        "id": "mem_001",
        "timestamp": "2024-01-10T10:00:00",
        "text": "The primary production database is PostgreSQL hosted on AWS RDS us-east-1."
    },
    {
        "id": "mem_002",
        "timestamp": "2024-08-20T14:30:00",
        "text": "Production database migrated from AWS RDS to a self-managed CockroachDB cluster."
    },
    {
        "id": "mem_003",
        "timestamp": "2024-03-05T09:00:00",
        "text": "The development team uses Python 3.11 with FastAPI."
    }
]

# Filter out superseded assertions
clean_memories = alethex.filter_context(retrieved_memories)

for entry in clean_memories:
    print(f"[{entry['status'].upper()}] {entry['id']}: {entry['text']}")
# Output:
# [SUPERSEDED] mem_001: The primary production database is PostgreSQL...
# [VALID]      mem_002: Production database migrated from AWS RDS to a self-managed CockroachDB...
# [VALID]      mem_003: The development team uses Python 3.11 with FastAPI.
```

#### Performing an End-to-End Consistency Audit

```python
from alethex.api import ConsistencyEngine
from alethex.ingestion.schema import RawDocument
from datetime import datetime

engine = ConsistencyEngine(device="cpu")

documents = [
    RawDocument(
        source_id="turn_1",
        timestamp=datetime(2024, 2, 1),
        text="The client server protocol is strictly WebSockets over TLS."
    ),
    RawDocument(
        source_id="turn_2",
        timestamp=datetime(2024, 2, 1),
        text="The client server protocol is strictly REST over HTTP/2."
    )
]

audit_results = engine.check_documents(documents)

print(f"Corpus Consistency Index: {audit_results['corpus_ci']:.4f}")
print(f"Contradictions Flagged: {len(audit_results['contradictions'])}")
for conflict in audit_results["contradictions"]:
    print(f"Conflict: '{conflict['claim_A_text']}' vs '{conflict['claim_B_text']}'")
```

### Online Streaming Belief Tracker

For real-time multi-turn agent sessions where context is updated incrementally turn-by-turn:

```python
from datetime import datetime
from alethex.streaming.online_session import OnlineBeliefTracker

tracker = OnlineBeliefTracker()

# Turn 1
report1 = tracker.process_turn(
    speaker="user",
    text="Our API rate limit is set to 100 requests per minute.",
    timestamp=datetime(2024, 1, 15)
)

# Turn 2
report2 = tracker.process_turn(
    speaker="user",
    text="We upgraded the infrastructure tier; API rate limit is now 5000 requests per minute.",
    timestamp=datetime(2024, 5, 20)
)

print(f"Active Beliefs: {tracker.get_active_beliefs(speaker='user')}")
print(f"Current Consistency Index: {tracker.get_current_ci():.3f}")
```

### Command Line Interface (CLI)

The package provides a comprehensive CLI for headless pipeline operations:

```bash
# Execute end-to-end reconciliation on a JSONL document corpus
python -m alethex.cli run-pipeline \
    --corpus path/to/corpus.jsonl \
    --output-dir output/ \
    --device cpu

# Launch the interactive local REST API and Web Dashboard
python -m alethex.cli serve --host 127.0.0.1 --port 8000

# Execute benchmark evaluation across synthetic and real datasets
python -m alethex.evaluation.eval_pipeline --samples 150 --output-dir results/
```

---

## Repository Structure

```text
Alethex/
|-- README.md                             # Comprehensive technical documentation
|-- index.html                            # Production presentation site & interactive demo
|-- launch_all.bat                        # Automated multi-service orchestration launcher
|-- run_desktop.bat                       # Native desktop companion runner
|-- open_extension_folder.bat             # Extension directory location helper
|
|-- alethex-guard-v1.2/                   # Universal Chrome Extension (Manifest V3)
|   |-- manifest.json                     # Extension manifest configuration
|   |-- background.js                     # Service worker managing state and background events
|   |-- content.js                        # High-performance DOM mutation observer & UI HUD
|   |-- popup.html                        # Quick status and control panel
|   |-- popup.js                          # Control panel interaction script
|   |-- icon16.png                        # 16x16 status icon
|   |-- icon48.png                        # 48x48 browser action icon
|   `-- icon128.png                       # 128x128 store asset
|
|-- dist/
|   `-- alethex-extension-v1.0.0.zip      # Packaged extension archive for direct distribution
|
`-- alethex/                              # Python Verification Engine Package
    |-- pyproject.toml                    # Package metadata, dependencies, and build specs
    |-- requirements.txt                  # Pinned runtime dependencies
    |-- alethex_desktop.py                # Native Tkinter desktop monitoring companion
    |-- start_all.py                      # Master process orchestrator
    |-- LICENSE                           # MIT License specification
    |
    |-- configs/                          # Declarative system configurations
    |   |-- extraction.yaml               # OpenIE and LLM extraction parameters
    |   |-- entity_linking.yaml           # DBSCAN epsilon and min_samples settings
    |   |-- temporal.yaml                 # Temporal parsing and interval thresholds
    |   |-- nli.yaml                      # Cross-encoder confidence thresholds and model paths
    |   |-- training.yaml                 # Knowledge distillation hyperparameters
    |   `-- training_h100.yaml            # High-throughput cluster training configuration
    |
    |-- alethex/                          # Core Package Modules
    |   |-- api.py                        # Public high-level ConsistencyEngine & helper functions
    |   |-- cli.py                        # Headless command-line interface entry points
    |   |-- pipeline.py                   # Canonical eight-stage execution pipeline
    |   |-- config.py                     # Configuration loader and environment manager
    |   |
    |   |-- ingestion/                    # Ingestion Schemas and Loaders
    |   |   |-- schema.py                 # Claim, EntityNode, RawDocument dataclass definitions
    |   |   `-- loaders.py                # Multi-format document parsers (JSONL, CSV, Text)
    |   |
    |   |-- extraction/                   # Factual Claim Extraction Subsystem
    |   |   |-- openie_extractor.py       # High-speed rule-based OpenIE parser
    |   |   |-- llm_extractor.py          # Few-shot LLM structured triple extractor
    |   |   `-- claim_merger.py           # Deduplication and confidence aggregation
    |   |
    |   |-- coreference/                  # Cross-Document Mention Resolution
    |   |   |-- within_doc_coref.py       # Heuristic antecedent pronoun resolver
    |   |   `-- entity_linker.py          # Sentence-BERT embedding generator & DBSCAN clusterer
    |   |
    |   |-- temporal/                     # Temporal Resolution Subsystem
    |   |   |-- timex_tagger.py           # Relative and absolute TIMEX3 normalization
    |   |   `-- interval_resolver.py      # Allen's 13 interval algebra comparative resolver
    |   |
    |   |-- relation/                     # Relational Classification Subsystem
    |   |   |-- pair_generator.py         # Entity-constrained candidate pair generator
    |   |   |-- nli_classifier.py         # Cross-encoder NLI classifier (PyTorch / ONNX)
    |   |   `-- llm_judge.py              # LLM-as-a-Judge arbitrator for ambiguous candidates
    |   |
    |   |-- graph/                        # Dynamic Temporal Belief Graph Subsystem
    |   |   |-- belief_graph.py           # NetworkX MultiDiGraph manager & active belief state
    |   |   `-- query.py                  # Structured subgraph querying & path validation
    |   |
    |   |-- scoring/                      # Mathematical Consistency Auditing
    |   |   |-- consistency_index.py      # Entity and Corpus Consistency Index implementations
    |   |   |-- contradiction_report.py   # Comprehensive conflict reporting generator
    |   |   `-- staleness_report.py       # Exponential decay staleness scoring
    |   |
    |   |-- streaming/                    # Online Real-Time Ingestion
    |   |   `-- online_session.py         # Incremental turn-by-turn belief tracking
    |   |
    |   |-- context_memory/               # RAG Context Memory Adapters
    |   |   |-- compressor.py             # Context compression and redundancy reduction
    |   |   |-- injector.py               # Reconciled prompt formatting and context injection
    |   |   `-- retriever.py              # Consistency-aware retrieval wrapper
    |   |
    |   |-- integrations/                 # Universal Middleware & Gateway Adapters
    |   |   |-- mcp_server.py             # Model Context Protocol server (Claude / Cursor)
    |   |   |-- openai_proxy.py           # Reverse proxy for OpenAI-compatible endpoints
    |   |   |-- openwebui_filter.py       # Pipeline filter for Open WebUI
    |   |   |-- universal_rag.py          # Universal RAG middleware for all vector stores
    |   |   |-- langchain_memory.py       # Drop-in BaseMemory implementation for LangChain
    |   |   |-- llamaindex_filter.py      # BaseNodePostprocessor for LlamaIndex
    |   |   |-- auto_attach.py            # OS-level auto-detector and auto-configuration
    |   |   `-- rag_service.py            # Background microservice integration
    |   |
    |   |-- optim/                        # Production Inference Optimization
    |   |   |-- onnx_exporter.py          # Dynamic INT8 ONNX conversion pipeline
    |   |   `-- onnx_pipeline.py          # High-performance ONNX Runtime inference engine
    |   |
    |   |-- data/                         # NLI Dataset Downloader & Preprocessor Subsystem
    |   |   |-- preprocess_nli.py         # Unifies MultiNLI, SNLI, Temporal-NLI & All-NLI into canonical 3-way format
    |   |   |-- download_datasets.py      # HuggingFace automated multi-corpus downloader
    |   |   |-- export_datasets.py        # Raw JSONL dataset split exporter
    |   |   `-- synthetic_benchmark/      # Algorithmic benchmark synthesis generator
    |   |
    |   |-- training/                     # Distillation and Model Fine-Tuning
    |   |   |-- distill.py                # Student distillation trainer (ALETHEX-Mini)
    |   |   `-- train_nli.py              # Supervised fine-tuning for cross-encoder NLI
    |   |
    |   `-- reporting/                    # Visualization and Export Subsystem
    |       |-- export.py                 # GraphML, JSON, and HTML report exporters
    |       `-- dashboard/
    |           `-- app.py                # FastAPI visualization server
    |
    `-- tests/                            # Test Suite (Unit & Integration Tests)
        |-- test_api.py                   # High-level Python API tests
        |-- test_pipeline.py              # Full 8-stage canonical pipeline tests
        |-- test_graph.py                 # Belief graph and revision semantics tests
        |-- test_scoring.py               # Consistency Index and staleness calculation tests
        |-- test_temporal.py              # Allen's interval algebra and TIMEX tests
        |-- test_relation.py              # NLI and pair classification tests
        |-- test_extraction.py            # OpenIE and claim extraction tests
        |-- test_coref.py                 # Mention extraction and entity linking tests
        |-- test_streaming.py             # Incremental online tracking tests
        |-- test_onnx_inference.py        # INT8 ONNX runtime equivalence tests
        |-- test_integrations.py          # LangChain and LlamaIndex integration tests
        |-- test_universal_rag.py         # Universal RAG middleware tests
        |-- test_context_memory.py        # Memory compression and injection tests
        |-- test_distill.py               # Model distillation harness tests
        |-- test_reporting_export.py      # Report export format tests
        |-- test_preprocess_nli.py        # MultiNLI / SNLI preprocessing and data integrity tests
        `-- test_run_real_dataset.py      # Real-world benchmark dataset evaluation tests
```

---

## Verification and Test Suite

ALETHEX includes an automated test suite comprising 48 unit and integration tests covering every subsystem from raw string extraction to ONNX runtime quantization.

To execute the test suite:

```bash
# Navigate to the alethex directory
cd alethex

# Run the complete test suite
python -m pytest tests/ -v
```

### Test Coverage Highlights

- **`test_api.py`**: Verifies high-level `ConsistencyEngine` construction, `filter_context`, and multi-document consistency auditing.
- **`test_temporal.py`**: Validates all 13 Allen interval relations, unbounded intervals, and edge cases in TIMEX normalization.
- **`test_scoring.py`**: Confirms strict metric bounds ($0.0 \le CI \le 1.0$) across perfectly consistent, partially conflicting, and completely disjoint graphs.
- **`test_graph.py`**: Validates graph mutation integrity, node deactivation on supersession, and GraphML export formatting.
- **`test_onnx_inference.py`**: Verifies that the INT8 quantized ONNX runtime produces relation probabilities within numerical tolerance of the FP32 PyTorch baseline.
- **`test_integrations.py`**: Verifies compatibility with LangChain memory interfaces and LlamaIndex node postprocessors.
- **`test_universal_rag.py`**: Tests chunk suppression and annotation across simulated vector database outputs.
- **`test_preprocess_nli.py`**: Verifies MultiNLI / SNLI / Temporal-NLI canonical label remapping, schema transformations, and on-disk file integrity.

---

## Dataset Preprocessing & NLI Fine-Tuning Corpus (MultiNLI / SNLI / Temporal-NLI)

ALETHEX includes an end-to-end multi-corpus ingestion and preprocessing subsystem located in [`alethex/alethex/data/preprocess_nli.py`](file:///D:/Projects/Notes/Text%20Analytics%20Project/alethex/alethex/data/preprocess_nli.py) (with top-level runner [`alethex/preprocess_nli.py`](file:///D:/Projects/Notes/Text%20Analytics%20Project/alethex/preprocess_nli.py)) that unifies four leading NLI benchmarks:
- **MultiNLI (`nyu-mll/multi_nli`)**: Complex premise-hypothesis pairs spanning 10 distinct dialogue, spoken, and written genres.
- **SNLI (`stanfordnlp/snli`)**: High-agreement crowd-sourced image caption entailments and contradictions.
- **Temporal-NLI (`tasksource/temporal-nli`)**: Strict time-interval and temporal-drift logic pairs.
- **All-NLI (`sentence-transformers/all-nli`)**: Extended positive entailment grounding pairs.

### Canonical Label Alignment
Raw MultiNLI and SNLI define labels as `0: entailment, 1: neutral, 2: contradiction`. ALETHEX explicitly normalizes all sources into the pretrained `cross-encoder/nli-deberta-v3-large` canonical head format:
- **`0`**: Contradiction
- **`1`**: Entailment
- **`2`**: Neutral

Unlabeled examples (e.g., SNLI consensus label `-1`) are automatically filtered out.

### Verifying and Running Preprocessing
```bash
# Verify existing preprocessed dataset integrity, pair counts, and class splits:
python -m alethex.cli verify-dataset

# Run automated unit tests verifying preprocessing transformations:
python -m pytest tests/test_preprocess_nli.py -v

# Execute preprocessing from raw sources:
python alethex/preprocess_nli.py
# or via package module:
python -m alethex.data.preprocess_nli
```

## Citation

If you utilize ALETHEX in your research or production systems, please cite the following work:

```bibtex
@article{alethex2026,
  title   = {ALETHEX: Corpus-Scale Temporal Belief Consistency and Contradiction Detection for Persistent LLM Memory Systems},
  author  = {Yugendhar S},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](alethex/LICENSE) file for the complete license terms.
