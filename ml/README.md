---
library_name: transformers
tags:
- legal
- dpdp
- qwen
- slm
- unsloth
- simpo
- rslora
- rag
- bge
license: apache-2.0
language:
- en
datasets:
- custom
metrics:
- f1
- accuracy
- ndcg
pipeline_tag: text-generation
---

# 🛡️ DPDP SSense: Legal SLM Subsystem (`ml/`)

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11_cu130-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.5.0-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026.8-brightgreen.svg)](https://github.com/unslothai/unsloth)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_DGX_Spark_GB10-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**Industrial-grade Data Engineering, Synthetic GAN Forge, Supervised Fine-Tuning (SFT), Reference-Free Preference Optimization (SimPO), SOTA Hybrid RAG, and Adversarial Certification Harness for the Indian Digital Personal Data Protection (DPDP) Act 2023 & Rules 2025.**

</div>

---

## 📑 Table of Contents

1. [Executive Summary & Architectural Paradigm](#1-executive-summary--architectural-paradigm)
2. [Directory & Subsystem Structure](#2-directory--subsystem-structure)
3. [Hardware & Provisioning Pipeline](#3-hardware--provisioning-pipeline)
4. [Stage 1: SOTA Data Engineering & Synthesis](#4-stage-1-sota-data-engineering--synthesis)
   - [4.1 Legal-Aware Hybrid RAG Engine (`build_vector_db.py`)](#41-legal-aware-hybrid-rag-engine)
   - [4.2 Teacher-Student Synthetic GAN Forge (`gan_forge.py`)](#42-teacher-student-synthetic-gan-forge)
   - [4.3 Deterministic Statutory Decision Tree (`build_dpdp_tree.py`)](#43-deterministic-statutory-decision-tree)
   - [4.4 Unsloth Dataset Formatting & Leaky-Split Firewall (`prepare_unsloth_data.py`)](#44-unsloth-dataset-formatting--leaky-split-firewall)
5. [Stage 2: The Training Loop & Alignment Architecture](#5-stage-2-the-training-loop--alignment-architecture)
   - [5.1 VRAM Airlock & Hardware Isolation](#51-vram-airlock--hardware-isolation)
   - [5.2 Phase 1: Supervised Fine-Tuning (SFT) with rsLoRA](#52-phase-1-supervised-fine-tuning-sft-with-rslora)
   - [5.3 Phase 2: Simple Preference Optimization (SimPO)](#53-phase-2-simple-preference-optimization-simpo)
   - [5.4 Model Specialization Matrix](#54-model-specialization-matrix)
   - [5.5 Triple-Format Export Engine](#55-triple-format-export-engine)
6. [Stage 3: Functional, Adversarial & SOTA Certification](#6-stage-3-functional-adversarial--sota-certification)
   - [6.1 Polymorphic Backend Loader (`backend_loader.py`)](#61-polymorphic-backend-loader)
   - [6.2 Statistical Power & Wilson 95% Confidence Interval Gating](#62-statistical-power--wilson-95-confidence-interval-gating)
   - [6.3 The 18-Axis Master Certification Matrix](#63-the-18-axis-master-certification-matrix)
   - [6.4 MinHash 7-Gram Data Contamination Firewall (`check_data_leakage.py`)](#64-minhash-7-gram-data-contamination-firewall)
7. [Stage 4: Unified Hugging Face Mono-Repo Deployment](#7-stage-4-unified-hugging-face-mono-repo-deployment)
8. [Quickstart & Inference Guide](#8-quickstart--inference-guide)
9. [Operational Runbook (Step-by-Step DGX Execution)](#9-operational-runbook-step-by-step-dgx-execution)

---

## 1. Executive Summary & Architectural Paradigm

General-purpose foundation models frequently hallucinate legal sections, invent arbitrary penalties, stretch statutory provisions, or bleed conversational preambles into structured JSON APIs. 

To eliminate these vulnerabilities, **Ssense** decouples legal analysis into two domain-specialized **Small Language Models (SLMs)** fine-tuned on top of **`Qwen2.5-7B-Instruct`** in native BF16 precision, backed by a deterministic Hybrid RAG retriever:

```text
                                 ┌────────────────────────────────────────────────────────┐
                                 │                DPDP Act 2023 & Rules 2025              │
                                 │             Raw Legal Knowledge & Case Law             │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │       STAGE 1: DATA ENGINEERING & SYNTHESIS            │
                                 │  • Hybrid BM25 + BGE Embeddings + Cross-Encoder        │
                                 │  • Teacher 72B GAN Forge (Qwen2-72B-Instruct-FP8)      │
                                 │  • Deterministic Rust Decision Tree Compilation        │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │ (SFT + SimPO Datasets)
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │      STAGE 2: UNSLOTH TRAINING & SIMPO ALIGNMENT       │
                                 │  Base Model: Qwen2.5-7B-Instruct (Native BF16)         │
                                 │  Architecture: rsLoRA + FP32 AdamW + Fused Triton      │
                                 └─────────────────────┬───────────────────┬──────────────┘
                                                       │                   │
                            ┌──────────────────────────┘                   └──────────────────────────┐
                            ▼                                                                         ▼
┌────────────────────────────────────────────────────────┐         ┌────────────────────────────────────────────────────────┐
│             FORENSIC LEGAL AUDITOR SLM                 │         │             CONVERSATIONAL CHATBOT SLM                 │
│                 (audit-model-final)                    │         │                 (chatbot-model-final)                  │
├────────────────────────────────────────────────────────┤         ├────────────────────────────────────────────────────────┤
│ • Task: Comprehensive Privacy Policy Forensic Auditing │         │ • Task: Citizen Dialogue & Data Rights Guidance        │
│ • Context Window: 8,192 tokens                         │         │ • Context Window: 4,096 tokens                         │
│ • Output: Strict dpdp_schema.json validated JSON       │         │ • Output: High-fluidity Markdown Natural Language      │
│ • LoRA: rsLoRA r=128, α=32, target: all-linear         │         │ • LoRA: rsLoRA r=64, α=16, target: all-linear          │
│ • Alignment: SimPO β=2.0 (Zero-hallucination mandate)  │         │ • Alignment: SimPO β=1.0 (Conversational RAFT)         │
└───────────────────────────┬────────────────────────────┘         └───────────────────────────┬────────────────────────────┘
                            │                                                                  │
                            └──────────────────────────┐   ┌───────────────────────────────────┘
                                                       ▼   ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │   STAGE 3: FUNCTIONAL & ADVERSARIAL CERTIFICATION      │
                                 │  • 18 Threshold Gates across 9 Eval Sub-Suites         │
                                 │  • Wilson 95% CI Lower Bound Statistical Gating        │
                                 │  • MinHash 7-Gram Contamination Firewall               │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                                             ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │   STAGE 4: UNIFIED HUGGING FACE LFS DEPLOYMENT         │
                                 │  • Push to PRiyanshu0-1/DPDP-SSense Mono-Repo          │
                                 │  • vLLM Production Server / Edge GGUF Delivery         │
                                 └────────────────────────────────────────────────────────┘
```

---

## 2. Directory & Subsystem Structure

The `ml/` repository is self-contained and structured to manage data synthesis, model training, evaluation, and Git LFS deployment:

```text
ml/
├── requirements.txt             # Pinned dependencies (PyTorch 2.11, Transformers 5.5, TRL 0.24, Unsloth)
├── README.md                    # Hugging Face Model Card & Technical Specification
├── .gitattributes               # Git LFS tracking rules (safetensors, gguf, pkl, pdf)
├── .gitignore                   # Firewall for optimizer states, raw dumps, and caches
│
├── data-forge/                  # STAGE 1: Data Engineering, Vector DB & Synthetic Forge
│   ├── DPDP_Act_2023.pdf        # Official Gazette of India: DPDP Act 2023
│   ├── DPDP_Rules_2025.pdf      # Official Draft: DPDP Rules 2025
│   ├── dpdp_act_and_rules_2025.txt # Curated statutory reference corpus
│   ├── build_vector_db.py       # SOTA Hybrid Search Engine Builder (BM25 + Dense + Reranker)
│   ├── gan_forge.py             # 72B Teacher Synthetic Generator & Loophole Injection
│   ├── build_dpdp_tree.py       # Mathematical Rust Decision Tree Generator
│   ├── prepare_unsloth_data.py  # ChatML formatting, Response-Only Masking, & Leaky-Split Isolation
│   ├── dpdp_hybrid_index.pkl    # Serialized hybrid search index artifact
│   └── training-pairs/          # Raw and formatted SFT / SimPO training data
│
├── slm-training/                # STAGE 2: High-Throughput SFT & SimPO Training Loops
│   ├── train_audit.py           # Industrial SFT + SimPO pipeline for Forensic Auditor (8k context)
│   ├── train_chatbot.py         # Industrial SFT + SimPO pipeline for Conversational Chatbot (4k context)
│   └── data/                    # Ingestion-ready JSONL datasets for Unsloth
│
├── evals/                       # STAGE 3: Industrial Functional & Adversarial Certification Harness
│   ├── backend_loader.py        # Polymorphic Model Loader (Unsloth / vLLM / llama.cpp / 72B Judge)
│   ├── stats.py                 # Wilson Score 95% Confidence Interval & Lexical Fluidity Metrics
│   ├── metrics.py               # Shared Deduplicated Evaluators & Parametric Citation Registry
│   ├── path_resolver.py         # Centralized immutable path resolution
│   ├── run_grammar_evals.py     # Pillar 1 & 5: Schema Compliance, Delimiter Integrity, Latency
│   ├── run_accuracy_evals.py    # Pillar 2-4: Severity-Weighted F1, Trust MAE, Zero-Hallucination
│   ├── evaluate_rag.py          # Pillar 1 RAG: BM25 + Dense + Reranker Recall@3 & NDCG@3
│   ├── run_chatbot_evals.py     # Chatbot Statutory Accuracy, Schema Bleed, & MTLD Fluidity
│   ├── evaluate_chatbot.py      # SOTA Chatbot 5-Axis Certification (SCP, Context Faithfulness, JCR)
│   ├── run_hallucination_benchmark.py # Red-Team Statutory Traps & Silent Refusal (N=50)
│   ├── run_security_evals.py    # Adversarial Security: 2D NIAH (20k), Prompt Injection, Sycophancy
│   ├── benchmark_latency.py     # Concurrency (1, 4, 8, 16) & 32k-Token Memory Stress Benchmark
│   ├── compare_sota_models.py   # Baseline vs SLM Win-Condition Comparator
│   ├── verify.py                # Master Automated Verification Orchestrator & Markdown Generator
│   ├── verify.sh                # Full-Precision BF16 Verification Runner
│   ├── verify_edge.sh           # Quantized GGUF Q4_K_M Edge Verification Runner
│   ├── holdout_policies/        # Statistically balanced holdout evaluation datasets
│   ├── benchmarks/              # Expanded adversarial & security benchmark vectors
│   └── reports/                 # JSON and Markdown certification scorecards
│
├── scripts/                     # Operational Lifecycle & Automation Scripts
│   ├── install.sh               # DGX Spark GB10 Hardware Provisioning & Source Compilation
│   ├── 01_prepare_data.sh       # Stage 1: Vector DB, GAN Forge & Dataset Formatting
│   ├── 02_train_models.sh       # Stage 2: VRAM Airlock & Unsloth Training Loops
│   ├── 03_evaluate_models.sh    # Stage 3: Master Certification Pipeline Runner
│   ├── evaluate.sh              # Direct shortcut alias for Stage 3 evaluation
│   └── check_data_leakage.py    # MinHash 7-gram Jaccard Contamination Firewall
│
└── models/                      # Checkpoints, LoRA Adapters, Merged Safetensors & GGUF Artifacts
```

---

## 3. Hardware & Provisioning Pipeline

Engineered natively for **NVIDIA DGX Spark** (Grace-Blackwell GB10, aarch64, CUDA 13.0, compute capability `sm_121` targeting `sm_120` forward compatibility).

### 3.1 Strict Version Lock Matrix (`ml/requirements.txt`)
Because Unsloth enforces strict compatibility bounds, dependencies are pinned:
* **PyTorch:** `torch==2.11.0` (`cu130`)
* **Transformers:** `transformers==5.5.0` (Hard cap for Unsloth)
* **TRL:** `trl==0.24.0` (Stable `CPOConfig` with native SimPO loss)
* **Accelerate:** `accelerate==1.10.0`
* **PEFT:** `peft==0.18.1` (Patched for TorchAO quantization dispatch)
* **TorchAO:** `torchao>=0.13.0`
* **Unsloth:** `unsloth==2026.8.15`, `unsloth_zoo==2026.8.10`

### 3.2 Deterministic Compilation (`ml/scripts/install.sh`)
Standard `pip install` fails on aarch64 CUDA 13.0 environments. `ml/scripts/install.sh` provisions the system in 5 phases:
1. **Phase 1: Build Essentials & Base Packages:** Builds `ninja`, `cmake`, and foundation tools.
2. **Phase 2: Xformers Source Build:** Compiles `v0.0.30` with `MAX_JOBS=8` without build isolation.
3. **Phase 3: Flash-Attention Source Build:** Compiles `v2.6.3` with `MAX_JOBS=4` to avoid host RAM exhaustion.
4. **Phase 4: Llama-CPP-Python:** Builds CUDA GGML kernels (`-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120`).
5. **Phase 5: Unsloth & BitsAndBytes (`--no-deps`):** Ingests pre-built aarch64 wheels with strict `--no-deps` to preserve compiled PyTorch bindings.

---

## 4. Stage 1: SOTA Data Engineering & Synthesis

### 4.1 Legal-Aware Hybrid RAG Engine (`build_vector_db.py`)

Legal statutory retrieval requires exact keyword matching for specific statutory clauses (e.g., "Section 8(1)(a)", "Rule 13(4)") and dense semantic understanding for conceptual queries (e.g., "consent mechanics for minor data processing").

```text
User Legal Query
      │
      ├──► BM25 Legal Tokenizer (Statutory Entity Preservation) ───────┐
      │                                                                 ├─► RRF Score Fusion ─► BGE-Reranker-v2-m3 ─► Top-3 Grounding Chunks
      └──► Dense Semantic Embedding (BAAI/bge-small-en-v1.5 + L2 Norm) ─┘
```

1. **Custom Statutory Tokenizer:** Collapses multi-word statutory terms (`"data fiduciary" -> "data_fiduciary"`, `"Section 8(1)" -> "section_8_1"`) to prevent Inverse Document Frequency (IDF) dilution in BM25.
2. **Dense Vector Embeddings:** Encodes 800-character chunks with 80-character sliding overlaps using `BAAI/bge-small-en-v1.5`. Unpolluted chunk vectors with L2 unit-norm normalization prevent dense space drift.
3. **Cross-Encoder Re-Ranking:** RRF candidate pools (Top 50 Lexical + Top 50 Dense) are fused and re-ranked using `BAAI/bge-reranker-v2-m3` across the top 30 candidates, achieving **$\ge 98\%$ Recall@3** and **$\ge 0.95$ NDCG@3**.
4. **Serialized Persistence:** The complete hybrid index is serialized into `ml/data-forge/dpdp_hybrid_index.pkl`, eliminating database concurrency deadlocks during distributed training.

### 4.2 Teacher-Student Synthetic GAN Forge (`gan_forge.py`)
Generates 10,000+ synthetic corporate privacy policies with labeled ground-truth violations across 8 statutory categories:
* **Teacher Model:** `Qwen2-72B-Instruct-FP8` running on vLLM.
* **Chosen ($y_w$):** Perfect forensic audits strictly adhering to `dpdp_schema.json`, citing exact statutory sections, with 0% hallucination and verbatim quotes.
* **Rejected ($y_l$):** Plausible but legally flawed audits exhibiting subtle legal hallucinations, non-existent sections, or loose quote attribution.

### 4.3 Deterministic Statutory Decision Tree (`build_dpdp_tree.py`)
Extracts the DPDP Act 2023 & Rules 2025 text into an Abstract Syntax Tree (AST). The decision tree compiles into `dpdp_act_tree.json` and is utilized by the upstream Rust core engine (`libs/rust-utils`) for sub-millisecond heuristic screening.

### 4.4 Unsloth Dataset Formatting & Leaky-Split Firewall (`prepare_unsloth_data.py`)
* **ChatML Normalization:** Formats conversations strictly using native tokens (`<|im_start|>system...`).
* **Leaky-Split Protection:** Groups train/eval splits by `source_document_id` rather than individual rows, guaranteeing that the evaluation split tests true generalization.

---

## 5. Stage 2: The Training Loop & Alignment Architecture

### 5.1 VRAM Airlock & Hardware Isolation
Before initializing the PyTorch context, `ml/scripts/02_train_models.sh` executes the **VRAM Airlock**:
1. Terminates lingering Ray background workers and zombie vLLM processes (`pkill -9 -f vllm`).
2. Kills orphaned PyTorch multiprocessing workers (`multiprocessing.spawn`).
3. Flushes `/dev/shm/*` POSIX shared memory allocations.
4. Executes OS cache synchronization and memory dropping (`echo 3 > /proc/sys/vm/drop_caches`).
5. Enforces `multiprocessing.set_start_method("spawn")` and `TOKENIZERS_PARALLELISM=false`.

### 5.2 Phase 1: Supervised Fine-Tuning (SFT) with rsLoRA
Trains the base student model (`Qwen2.5-7B-Instruct`) to master statutory reasoning and JSON schema adherence.

* **Rank-Stabilized LoRA (`rsLoRA`):** Traditional LoRA scales adapters by $\frac{\alpha}{r}$, collapsing optimization stability at high ranks ($r \ge 64$). `rsLoRA` scales adapters by $\gamma = \frac{\alpha}{\sqrt{r}}$, mathematically ensuring stable gradient updates regardless of rank:
  ```python
  model = FastLanguageModel.get_peft_model(
      model,
      r=128,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      lora_alpha=32,
      lora_dropout=0.0,  # Fused Triton kernel optimization
      bias="none",
      use_gradient_checkpointing="unsloth",
      use_rslora=True,
  )
  ```
* **Response-Only Loss Masking (`train_on_responses_only`):** Computes cross-entropy loss exclusively on the assistant's completion tokens. Setting prompt tokens to `label = -100` prevents wasting parameter capacity on memorizing long statutory prompts.
* **Optimizer Configuration:** 32-bit FP32 `adamw_torch` states (`weight_decay=0.05`, linear warmup for 10% of steps, cosine learning rate decay).

### 5.3 Phase 2: Simple Preference Optimization (SimPO)
Standard Direct Preference Optimization (DPO) requires maintaining a frozen reference model in VRAM, doubling memory consumption and causing token length bias.

Ssense implements **SimPO (Simple Preference Optimization)** via `TRL`'s `CPOTrainer`, which optimizes preferences without a reference model using length-normalized implicit rewards:

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w | x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l | x) - \gamma \right) \right]$$

Where:
* $\pi_\theta(y | x)$ is the policy model's likelihood.
* $|y|$ is sequence length (enforcing length penalty).
* $\gamma > 0$ is the target reward margin enforcing strict separation between chosen ($y_w$) and rejected ($y_l$) completions.
* $\beta$ is the reward scaling temperature.

```python
cpo_config = CPOConfig(
    loss_type="simpo",
    beta=2.0,        # Aggressive penalty for legal hallucinations
    simpo_gamma=0.5, # Target margin
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=True,
    max_length=8192,
)
```

### 5.4 Model Specialization Matrix

| Architectural Parameter | Forensic Legal Auditor (`train_audit.py`) | Conversational Chatbot (`train_chatbot.py`) |
| :--- | :--- | :--- |
| **Base Architecture** | `Qwen2.5-7B-Instruct` | `Qwen2.5-7B-Instruct` |
| **Context Window ($N_{\text{ctx}}$)** | **8,192 tokens** | **4,096 tokens** |
| **Max Prompt Budget** | 7,000 tokens (Pre-filtered) | 3,000 tokens (Pre-filtered) |
| **LoRA Rank ($r$) / Alpha ($\alpha$)** | $r=128, \alpha=32$ (`rsLoRA`) | $r=64, \alpha=16$ (`rsLoRA`) |
| **Target Modules** | All 7 Linear Projection Layers | All 7 Linear Projection Layers |
| **SFT Learning Rate / Epochs** | $2\times 10^{-4}$ (3 Epochs) | $2\times 10^{-4}$ (3 Epochs) |
| **SimPO Learning Rate / Epochs** | $5\times 10^{-6}$ (1 Epoch) | $5\times 10^{-6}$ (1 Epoch) |
| **SimPO Hyperparameters** | $\beta = 2.0, \gamma = 0.5$ (Strict Hallucination Block) | $\beta = 1.0, \gamma = 0.3$ (Fluidity) |
| **Output Contract** | Strict `dpdp_schema.json` compliant JSON | Natural Language Markdown + Citation Tags |
| **Primary Deployment Role** | Automated Corporate Audit Engine | Interactive Citizen Privacy Assistant |

### 5.5 Triple-Format Export Engine
Upon convergence, `train_audit.py` and `train_chatbot.py` automatically export models into three distinct formats:
1. **LoRA Adapters (`adapter_model.safetensors`):** For dynamic multi-LoRA multiplexing under `vLLM` server deployments (~250 MB).
2. **Merged 16-Bit Safetensors (`model.safetensors`):** Standalone full-precision BF16 weights for offline batch evaluation and zero-latency serving (~15 GB).
3. **GGUF Q4_K_M Edge Weights (`model.Q4_K_M.gguf`):** Quantized weights for local edge execution via `llama.cpp` and our Rust Native Daemon (`apps/native-daemon`) (~4.5 GB).

---

## 6. Stage 3: Functional, Adversarial & SOTA Certification

The evaluation suite (`ml/evals/`) operates as the automated gating authority. No model advances to production without navigating the 18 threshold gates orchestrated by `verify.py`.

### 6.1 Polymorphic Backend Loader (`backend_loader.py`)
Dynamically wraps target models behind a unified inference interface:
* **Unsloth In-Memory:** Direct CUDA inference with Triton fused attention kernels.
* **vLLM Multi-LoRA:** High-throughput async REST client.
* **Llama.cpp Edge:** GGUF quantized execution for local edge benchmarking.
* **CUDA Memory Guard:** Explicitly strips arbitrary `max_length` generation kwargs, preventing C++ KV Cache out-of-bound writes (`CUDA error: an illegal memory access`).
* **VRAM Fragmentation Airlock:** Automatically unloads the `audit_engine` from VRAM and flushes PyTorch allocators before initializing `chat_engine` during security evals.

### 6.2 Statistical Power & Wilson 95% Confidence Interval Gating
To eliminate the "Small-N Fallacy", rate-based metrics are strictly gated against the **Wilson Score 95% Confidence Interval Lower Bound**:

$$\hat{p}_{\text{lower}} = \frac{\hat{p} + \frac{z^2}{2n} - z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Where:
* $\hat{p} = \frac{k}{n}$ (Sample success proportion)
* $n$ = Total test vectors
* $z = 1.95996$ (Critical value for 95% two-sided confidence)

### 6.3 The 18-Axis Master Certification Matrix

| # | Benchmark Suite | Evaluation Metric Label | Extraction Key | Target Gate | Statistical Gate Type |
|---|---|---|---|:---:|---|
| 1 | **Grammar Evals** | Pillar 1: Schema Compliance Rate (%) | `grammar.schema_compliance_rate` | **$\ge 98.0\%$** | Point Estimate |
| 2 | **Accuracy Evals** | Pillar 2: Severity-Weighted Violation F1 | `accuracy.severity_weighted_violation_f1` | **$\ge 0.88$** | Point Estimate |
| 3 | **Accuracy Evals** | Pillar 3: Trust Score MAE (pts) | `accuracy.trust_score_mae` | **$\le 8.5$ pts** | Point Estimate |
| 4 | **Accuracy Evals** | Pillar 4: Evidence Quote Hallucination (%) | `accuracy.evidence_quote_hallucination_rate` | **$\le 0.0\%$** | Wilson Upper Bound |
| 5 | **Grammar Evals** | Pillar 5: Average Inference Latency (ms) | `grammar.average_latency_ms` | **$\le 1200$ ms** | Telemetry P95 |
| 6 | **Chatbot Evals** | Chatbot: Statutory Accuracy Rate (%) | `chatbot.statutory_accuracy_rate` | **$\ge 95.0\%$** | Wilson Lower Bound |
| 7 | **Chatbot Evals** | Chatbot: Vocabulary Diversity (TTR/MTLD) | `chatbot.mtld_lexical_diversity` | **$\ge 0.45$** | Point Estimate |
| 8 | **Chatbot Evals** | Chatbot: Schema & Preamble Bleed Rate (%) | `chatbot.schema_preamble_bleed_rate` | **$\le 0.0\%$** | Wilson Upper Bound |
| 9 | **Red-Team Evals** | Red-Team: Statutory Trap Resistance (%) | `hallucination.statutory_trap_resistance_rate` | **$\ge 95.0\%$** | Wilson Lower Bound |
| 10 | **Security Evals** | Adversarial: NIAH 20k-Token Middle Recall (%) | `security.niah_context_recall_rate` | **$= 100.0\%$** | Wilson Lower Bound |
| 11 | **Security Evals** | Adversarial: Prompt Injection Refusal (%) | `security.prompt_injection_refusal_rate` | **$\ge 95.0\%$** | Wilson Lower Bound |
| 12 | **Security Evals** | Adversarial: Anti-Sycophancy Correction (%) | `security.sycophancy_correction_rate` | **$\ge 95.0\%$** | Wilson Lower Bound |
| 13 | **Security Evals** | Adversarial: JSON Fuzzing Resilience (%) | `security.json_fuzzing_resilience_rate` | **$\ge 95.0\%$** | Wilson Lower Bound |
| 14 | **Hybrid RAG** | SOTA RAG: Recall@3 Rate (%) | `sota_hybrid_rag.recall_at_3` | **$\ge 95.0\%$** | Point Estimate |
| 15 | **Hybrid RAG** | SOTA RAG: NDCG@3 Ranking Quality | `sota_hybrid_rag.ndcg_at_3` | **$\ge 0.90$** | Point Estimate |
| 16 | **Chatbot SOTA** | SOTA Chatbot: Statute Citation Precision (%) | `chatbot_sota.statute_citation_precision` | **$\ge 90.0\%$** | Point Estimate |
| 17 | **Chatbot SOTA** | SOTA Chatbot: Context Faithfulness Score (1-5) | `chatbot_sota.context_faithfulness_score` | **$\ge 4.5$** | 72B Judge Score |
| 18 | **Chatbot SOTA** | SOTA Chatbot: Jurisdictional Contamination (%) | `chatbot_sota.jurisdictional_contamination_rate` | **$\le 0.0\%$** | Wilson Upper Bound |

### 6.4 MinHash 7-Gram Data Contamination Firewall (`check_data_leakage.py`)
To prevent test set memorization from contaminating training checkpoints, `check_data_leakage.py` executes a MinHash Jaccard similarity audit across all 320 evaluation test vectors against the entire `ml/slm-training/data/` corpus:
* **7-Gram Shingling:** Tokenizes texts into sliding 7-word windows to ignore generic legal boilerplate while capturing verbatim sentence duplication.
* **128-Permutation MinHash Signatures:** Approximates pairwise Jaccard similarity in $O(N)$ time.
* **Hard-Fail Threshold:** Hard-fails (exit code 1) if any test vector exhibits **Jaccard Similarity $> 0.85$** against any training sample.

---

## 7. Stage 4: Unified Hugging Face Mono-Repo Deployment

Rather than isolating code and models in fractured repositories, the entire `ml/` subsystem serves as a centralized Git LFS Mono-Repo on Hugging Face (`PRiyanshu0-1/DPDP-SSense`).

* **LFS Interception:** Strict `.gitattributes` routing forces all `*.safetensors`, `*.gguf`, `*.pkl`, and `*.pdf` files through Git LFS, entirely circumventing standard Git push size limits.
* **Artifact Distillation:** Checkpoint optimizer states (`*.pt`) and raw GAN generation outputs are firewalled via `.gitignore`.
* **Triple-Export Delivery:** Hosts the PEFT Adapters (~250 MB), the merged native BF16 Safetensors (~15 GB), and the high-efficiency `Q4_K_M` GGUF binaries for edge serving.

---

## 8. Quickstart & Inference Guide

### 8.1 In-Memory Inference via Transformers / Unsloth

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PRiyanshu0-1/DPDP-SSense"
# Subfolder: "models/audit-model-final" or "models/chatbot-model-final"

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="models/chatbot-model-final")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder="models/chatbot-model-final",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """<|im_start|>system
You are an expert Indian DPDP Legal Assistant.<|im_end|>
<|im_start|>user
What are the statutory duties of a Data Principal under Section 15 of the DPDP Act 2023?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### 8.2 Edge Inference via `llama-cpp-python` (GGUF)

```python
from llama_cpp import Llama

# Load quantized GGUF model directly
llm = Llama(
    model_path="ml/models/chatbot-model-final/model.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1 # Offload all layers to GPU
)

prompt = """<|im_start|>system
You are an expert Indian DPDP Legal Assistant.<|im_end|>
<|im_start|>user
Can a Data Fiduciary transfer personal data outside India under the DPDP Act 2023?<|im_end|>
<|im_start|>assistant
"""

response = llm(prompt, max_tokens=512, temperature=0.1, stop=["<|im_end|>"])
print(response["choices"][0]["text"])
```

---

## 9. Operational Runbook (Step-by-Step DGX Execution)

Follow these step-by-step commands to provision, synthesize, train, and certify the models:

### Step 1: Provision the GB10 / CUDA 13.0 Environment
```bash
# Execute from the project root
bash ml/scripts/install.sh
```

### Step 2: Ingest Law, Build Hybrid DB & Synthesize Training Data
```bash
bash ml/scripts/01_prepare_data.sh
```

### Step 3: Run the Contamination Firewall Check
```bash
python ml/scripts/check_data_leakage.py --threshold 0.85
```

### Step 4: Execute SFT + SimPO Model Training
```bash
# Train both Forensic Auditor and Conversational Chatbot sequentially (default)
bash ml/scripts/02_train_models.sh

# Or train individually:
bash ml/scripts/02_train_models.sh --model audit
bash ml/scripts/02_train_models.sh --model chatbot
```

### Step 5: Execute the Master 18-Axis Certification Suite
```bash
# Certify merged 16-bit safetensors in-memory via Unsloth:
bash ml/scripts/03_evaluate_models.sh --backend unsloth

# Or certify via high-throughput production vLLM endpoint:
bash ml/scripts/03_evaluate_models.sh --backend vllm --vllm-url http://localhost:8000/v1/completions

# Or evaluate quantized GGUF Q4_K_M edge models via llama.cpp:
cd ml/evals && bash verify_edge.sh
```

### Step 6: Deploy Unified Mono-Repo to Hugging Face
```bash
cd ml
git add .
git commit -m "feat(ml): release production DPDP-SSense SLMs and certified hybrid RAG"
git push -u hf main
```

---

<div align="center">
<b>Built with ⚖️ by Ssense Engineering</b><br>
<i>Certified against the Digital Personal Data Protection Act 2023 & DPDP Rules 2025</i>
</div>
