# 🛡️ Ssense Machine Learning Pipeline (`ml/`)

This directory houses the end-to-end **Data Engineering, Synthetic GAN Forge, Supervised Fine-Tuning (SFT), Preference Optimization (SimPO), and Certification Architecture** for the Ssense **Digital Personal Data Protection (DPDP) Act 2023 & Rules 2025 Edge AI Platform**.

Unlike general-purpose LLMs or brittle quantization wrappers, our pipeline builds two specialized **9-Billion Parameter Small Language Models (SLMs)** fine-tuned from **[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)** and optimized to execute with zero hallucination, exact statutory quote fidelity, and mathematical variance precision on **NVIDIA DGX Spark (128 GB VRAM) infrastructure**.

---

## ⚡ DGX Spark Prerequisites & Specialized Pipeline Notice

> [!IMPORTANT]
> **SPECIALIZED DOMAIN BOUNDARY:** This entire repository (`ml/` and `scripts/`) is **strictly specialized** for the Indian Digital Personal Data Protection (DPDP) Act 2023 & Rules 2025 domain. It is **not** a generic fine-tuning wrapper. It synthesizes statutory legal policies, enforces `dpdp_schema.json` structural contracts, and certifies against 13 strict regulatory thresholds.

### Prerequisites for DGX Spark Execution (`128 GB VRAM`)
To execute the pipeline seamlessly on our **NVIDIA DGX Spark** environment without OOM crashes or worker deadlocks, ensure:
1. **GPU Infrastructure**: NVIDIA DGX / Spark cluster with at least **128 GB unified VRAM** (`multiprocessing.set_start_method("spawn")` and `TOKENIZERS_PARALLELISM=false` enforced).
2. **Runtime Environment**: Python 3.10+ / 3.12 with CUDA 12.x drivers and PyTorch (`torch.bfloat16` and Flash Attention 2 enabled).
3. **Dependencies**: Install via `pip install -r requirements.txt`:
   - `unsloth` (with custom Triton kernel support)
   - `trl` (with `PatchDPOTrainer` memory optimizations)
   - `transformers`, `datasets`, `vllm`, `huggingface_hub`, `readability-lxml`, `curl_cffi`, `playwright`, `jsonschema`
4. **Local Checkpoint Directory (`ml/models/`)**: Stage 1 (`scripts/01_prepare_data.sh`) automatically checks if our Teacher Model (`Qwen2-72B-Instruct-FP8`) and Student Model (`Qwen/Qwen3.5-9B`) are present locally inside `ml/models/`. If found locally, downloading is skipped instantly; otherwise, it pulls directly from Hugging Face via `huggingface-cli` or `huggingface_hub.snapshot_download`.

---

## 🏗️ The Teacher-Student Dual Model Paradigm

Our architecture splits synthetic data generation and production edge inference across two distinct model tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TEACHER GENERATION (OFFLINE DATA FORGE)                                       │
│ Model: Qwen2-72B-Instruct-FP8 (vLLM Engine, Tensor-Parallel = 1)                       │
│ Script: ml/data-forge/gan_forge.py & build_dpdp_tree.py                                │
│ Task: Synthesize 10,000+ complex corporate privacy policies, hard negative legal       │
│       justifications, multi-turn dialogue pairs, and verify quote boundaries.          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: STUDENT SUPERVISED & PREFERENCE ALIGNMENT (DGX TRAINING)                      │
│ Base Model: Qwen/Qwen3.5-9B (https://huggingface.co/Qwen/Qwen3.5-9B)                   │
│ Scripts: ml/slm-training/train_audit.py & train_chatbot.py                             │
│ Models Created:                                                                        │
│  1. Forensic Legal Auditor (`audit-model-final`): rsLoRA r=128, SimPO beta=2.0         │
│  2. Conversational Legal Assistant (`chatbot-model-final`): rsLoRA r=64, SimPO beta=1.0│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: CERTIFICATION & EDGE DEPLOYMENT (vLLM MULTI-LORA / UNSLOTH)                 │
│ Scripts: ml/evals/verify.py across universal backend_loader.py                         │
│ Outcome: Evaluated against 13 strict functional & adversarial certification thresholds.│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Breakdown & Module Architecture

```
ml/
├── data-forge/           # Stage 1: Synthetic Legal GAN Forge & Data Schema Alignment
│   ├── gan_forge.py               # 72B vLLM GAN forge synthesizing hard negative/subtle legal policies
│   ├── build_dpdp_tree.py         # Compiles statutory text into deterministic Rust network interceptor tree
│   ├── prepare_unsloth_data.py    # Aligns SFT + DPO pairs with exact right-side boundaries & contrastive formatting
│   ├── fill_policies.py           # Stealth web harvester for real-world benchmark Indian corporate policies
│   └── prompts/                   # Atomic statutory definitions, edge-case templates, and synthesizer rules
│       ├── edge_case_templates.py # Vendor liability shields & legitimate use overreach templates
│       └── target_violations.py   # Atomic DPDP statutory violation boundary definitions
│
├── slm-training/         # Stage 2: Supervised Fine-Tuning (SFT) + Preference Optimization (SimPO)
│   ├── train_audit.py             # Forensic Auditor SFT + SimPO pipeline (r=128, beta=2.0, max_prompt=23500)
│   └── train_chatbot.py           # Conversational Chatbot SFT + SimPO pipeline (r=64, beta=1.0, max_prompt=23500)
│
├── evals/                # Stage 3: Functional & Adversarial Certification Suite
│   ├── backend_loader.py          # Universal inference engine (Unsloth in-memory, vLLM multi-LoRA, llama.cpp GGUF)
│   ├── run_grammar_evals.py       # Pillar 1 & 5: JSON schema well-formedness and inference latency telemetry
│   ├── run_accuracy_evals.py      # Pillar 2-4: Severity-weighted F1, trust score MAE, and zero-hallucination check
│   ├── run_chatbot_evals.py       # Conversational accuracy, TTR vocabulary diversity, and schema containment
│   ├── run_hallucination_benchmark.py # Red-team statutory trap rejection verification
│   ├── run_security_evals.py      # 4-Vector vulnerability harness (NIAH 20k context, prompt injection, sycophancy, JSON fuzzing)
│   ├── verify.py                  # Master scorecard aggregator checking all 13 certification thresholds
│   └── verify.sh                  # POSIX execution harness passing inference flags to verify.py
│
└── models/               # Checkpoint storage (Intermediate SFT, LoRA adapters, standalone 16-bit safetensors)
```

---

## 🔬 Deep Technical Review: The GAN Forge (`ml/data-forge/gan_forge.py`)

The **GAN Forge** (`gan_forge.py`) is an autonomous, self-healing data synthesis engine powered by `vLLM` running a 72B teacher model (`Qwen2-72B-Instruct-FP8` configurable via `TEACHER_MODEL_PATH`). It resolves the critical bottleneck of legal AI training: *obtaining thousands of structurally sound, highly complex privacy policies that contain subtle statutory violations without human bias or formatting drift.*

### Key Architectural Mechanisms of `gan_forge.py`:

1. **Dual-Track Generator Architecture:**
   - **Track 1 (`run_audit_forge`)**: Synthesizes corporate privacy policies paired with exact JSON schemas conforming to `dpdp_schema.json`. Generates SFT training data (`audit_sft_data.jsonl`) and DPO preference data (`audit_dpo_data.jsonl`) where the chosen response reflects exhaustive statutory identification and the rejected response models common corporate justifications or omission blindness.
   - **Track 2 (`run_chatbot_forge`)**: Synthesizes multi-turn dialogue trees (`chatbot_sft_data.jsonl`, `chatbot_dpo_data.jsonl`) covering user privacy rights, data breach notices, and consent withdrawal procedures.

2. **Self-Healing Reflexion Loops (`MAX_REFLEXION_STEPS = 3`):**
   - When the Synthesizer generates a policy and audit JSON, the **Judge** (`judge_prompt.txt`) inspects the output against `dpdp_schema.json`.
   - If the output contains JSON syntax anomalies, unescaped inner quotes, missing required schema fields, or fails to properly identify the injected statutory violation (`edge_case_templates.py`), the forge automatically enters a **Reflexion Loop** (`reflexion_explicit_prompt.txt` / `reflexion_subtle_prompt.txt`), feeding the critique back to the Synthesizer until the output passes all quality gates.

3. **Verbatim Quote & Evidence Verification Firewall:**
   - **Exact Substring Validation (`is_quote_in_policy`)**: Every single `evidence_quote` extracted by the teacher in the audit JSON is programmatically verified against the raw `policy_text`. If the quote is not an exact character-for-character substring, it is rejected and re-generated.
   - **Abbreviation-Aware Period Purge & Multi-Sentence Checks**: Prevents the model from chopping sentences mid-way or treating `e.g.`, `i.e.`, or `Pvt. Ltd.` as sentence breaks.
   - **Devanagari Language Gate (`filter_english`)**: Filters out foreign characters and non-English text artifacts (`threshold=0.05`) to ensure clean UTF-8 training tensors.
   - **Foreign Law Bleed Protection**: Scans and blocks accidental mentions of `GDPR`, `CCPA`, `HIPAA`, or `LGPD` in statutory justifications, enforcing strict Indian `DPDP Act 2023 & Rules 2025` jurisdiction.

---

## 🎯 Contrastive DPO Strategy (`prepare_unsloth_data.py`)

A unique innovation in our data engineering is the **Contrastive Preference Optimization alignment** implemented in `prepare_unsloth_data.py`. To prevent our 9B model from suffering from "compliance paranoia" (hallucinating violations on benign boilerplate clauses), every DPO training pair (`dpo_000_*.json`) is rigorously formatted:

* **`chosen` (The Gold Standard)**: Contains the true, high-precision audit report extracting the exact character-for-character statutory violation quote (`evidence_quote`), citing the precise DPDP Act section (`statute_reference`), and assigning an accurate, severity-calibrated `dpdp_trust_score`.
* **`rejected` (The Hard Negative Trap)**: Targets benign, standard Indian corporate legal disclaimers (such as *“We retain personal data solely for the duration required to achieve the purposes for which it was collected or as mandated by applicable law”* or *“You have the right to request erasure under Section 12”*) and falsely argues that they constitute overbroad discretionary overreach under Section 8(7). 

By optimizing our student model against this contrastive margin using **SimPO (Simple Preference Optimization)**, we teach `Qwen 3.5 9B` to maintain absolute precision: rejecting false alarms on standard corporate boilerplate while staying ultra-vigilant for genuine structural breaches.

---

## ⚙️ Hardware & MLOps Specifications (`Qwen 3.5 9B` on `128 GB DGX`)

Our student model fine-tuning (`train_audit.py` & `train_chatbot.py`) directly targets **[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)**. To exploit our **128 GB unified VRAM DGX cluster** without the quantization noise of 8-bit approximations, we enforce 18 strict industrial MLOps parameters:

| MLOps Design Parameter | Hardcoded Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Base Student Model** | **`Qwen/Qwen3.5-9B`** | SOTA 9B architecture (`https://huggingface.co/Qwen/Qwen3.5-9B`) providing enterprise-grade reasoning. |
| **Optimizer Precision** | `adamw_torch` (32-bit FP32) | Retains exact 8-byte variance tracking ($v_t$) to guarantee zero statutory drift across legal statutes. |
| **Rank-Stabilized LoRA (`rsLoRA`)** | `r=128` (Auditor), `r=64` (Chatbot) | Scales adaptation factor via $1/\sqrt{r}$ to stabilize high-rank attention projections across linear layers. |
| **Triton Kernel Activation** | `lora_dropout = 0` | Fuses LoRA weights directly into attention matrices via custom Triton kernels, achieving ~30% speedups. |
| **Length Normalization (`SimPO`)** | `gamma=0.5`, `beta=2.0`/`1.0` | Eliminates target length explosion (`ref_model=None`) while calibrating conversational reward margins (`beta=1.0` for Chatbot to prevent reward hacking). |
| **Context Protection Window** | `max_prompt_length = 23500` | Preserves extensive 23k-token corporate privacy policies and dialogue histories without left-side truncation. |
| **CUDA Process Isolation** | `multiprocessing.set_start_method("spawn")` | Isolates SFT and DPO stages into clean OS worker processes, preventing PyTorch IPC memory leaks. |
| **Unified Adapter Continuity** | Unmerged LoRA Export -> `is_trainable=True` | Bridges Phase 1 SFT and Phase 2 SimPO by preserving unmerged adapters across multi-stage execution. |

---

## 🚀 Execution Pipeline (`scripts/`)

The end-to-end pipeline is controlled via three modular shell scripts located in the repository root `scripts/` directory:

### Stage 1: Data Preparation & GAN Forge
```bash
# Execute full synthetic generation, DPO pair alignment, and Unsloth formatting
bash scripts/01_prepare_data.sh

# Skip GAN generation if raw JSONL pairs are already generated in ml/slm-training/data/
bash scripts/01_prepare_data.sh --skip-gan
```

### Stage 2: VRAM Airlock & Model Training (`Qwen 3.5 9B`)
Automatically executes our **VRAM Airlock** (purging background Ray daemons, zombie vLLM processes, shared memory leaks `/dev/shm/*`, and OS caches) before initializing `adamw_torch`:
```bash
# Train both Forensic Auditor and Conversational Chatbot sequentially (default)
bash scripts/02_train_models.sh

# Target individual models or override base path via environment
BASE_MODEL_PATH="Qwen/Qwen3.5-9B" bash scripts/02_train_models.sh --model audit
BASE_MODEL_PATH="Qwen/Qwen3.5-9B" bash scripts/02_train_models.sh --model chatbot
```

### Stage 3: Functional & Adversarial Certification
Orchestrates all 5 evaluation modules across our universal backend loader (`backend_loader.py`):
```bash
bash scripts/03_evaluate_models.sh --backend unsloth  # Evaluate trained Unsloth LoRA adapters in memory
bash scripts/03_evaluate_models.sh --backend vllm     # Evaluate high-throughput production vLLM server
bash scripts/03_evaluate_models.sh --skip-run         # Aggregate existing reports into final scorecard
```

---

## 🏆 Master Certification Scorecard & Threshold Matrix

When `verify.py` (`bash scripts/03_evaluate_models.sh`) runs against our trained **`Qwen 3.5 9B`** models, it evaluates 13 strict certification gates across 5 dedicated verification harnesses. A model is only certified (`✅ PASS`) if every threshold is satisfied:

| Benchmark Module | Evaluation Metric Label | Target Threshold | Status Gate |
| :--- | :--- | :---: | :---: |
| **Grammar Evals** (`run_grammar_evals.py`) | Pillar 1: Schema Compliance Rate (%) | `>= 98.0%` | Zero JSON formatting errors |
| **Accuracy Evals** (`run_accuracy_evals.py`) | Pillar 2: Severity-Weighted Violation F1 | `>= 0.88` | High legal precision/recall |
| **Accuracy Evals** (`run_accuracy_evals.py`) | Pillar 3: Trust Score MAE (pts) | `<= 8.5 pts` | Accurate risk calibration |
| **Accuracy Evals** (`run_accuracy_evals.py`) | Pillar 4: Evidence Quote Hallucination (%) | `== 0.0%` | Verbatim quote mandate |
| **Grammar Evals** (`run_grammar_evals.py`) | Pillar 5: Average Inference Latency (ms) | `<= 1200.0 ms` | High-speed edge/vLLM delivery |
| **Chatbot Evals** (`run_chatbot_evals.py`) | Chatbot: Statutory Accuracy Rate (%) | `>= 95.0%` | Factual dialogue correctness |
| **Chatbot Evals** (`run_chatbot_evals.py`) | Chatbot: Vocabulary Diversity (TTR) | `>= 0.45` | Anti-reward-hacking fluidity |
| **Chatbot Evals** (`run_chatbot_evals.py`) | Chatbot: Schema & Preamble Bleed Rate (%) | `== 0.0%` | Zero internal JSON bleed |
| **Red-Team Evals** (`run_hallucination_benchmark.py`) | Red-Team: Statutory Trap Resistance (%) | `>= 98.0%` | Explicit trap rejection |
| **Security Evals** (`run_security_evals.py`) | Adversarial: NIAH 20k-Token Middle Recall (%) | `== 100.0%` | Zero context degradation |
| **Security Evals** (`run_security_evals.py`) | Adversarial: Prompt Injection Refusal (%) | `>= 98.0%` | Jailbreak / DAN protection |
| **Security Evals** (`run_security_evals.py`) | Adversarial: Anti-Sycophancy Correction (%) | `>= 95.0%` | Stands firm on legal premises |
| **Security Evals** (`run_security_evals.py`) | Adversarial: JSON Fuzzing Resilience (%) | `>= 95.0%` | Chaotic input stability |

---

## 🌐 Production Serving (`vLLM` Multi-LoRA on `Qwen 3.5 9B`)

Once certified, both `audit-model-final-adapter` (`r=128`) and `chatbot-model-final-adapter` (`r=64`) are served concurrently from a single base `Qwen/Qwen3.5-9B` instance using vLLM dynamic LoRA multiplexing:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 128 \
  --lora-modules audit=../models/audit-model-final-adapter chatbot=../models/chatbot-model-final-adapter
```
* Queries specifying `"model": "audit"` dynamically trigger the `r=128` Forensic Auditor adapter for rigorous JSON inspection.
* Queries specifying `"model": "chatbot"` dynamically trigger the `r=64` Conversational Assistant adapter for natural dialogue.
