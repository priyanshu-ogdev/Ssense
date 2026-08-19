# 🏛️ Ssense Virtual SLM Server (Zero-Hop Architecture)

An enterprise-grade, high-throughput Small Language Model (SLM) server engineered for **Digital Personal Data Protection (DPDP) Act 2023** compliance. 

Built on a **Zero-Hop In-Memory Architecture**, this server eliminates container network latency, dynamically enforces strict VRAM caps, and scales to 1,000+ concurrent users via FP8 KV Cache compression and continuous batching. It operates as an ultra-secure, zero-exfiltration endpoint for the Ssense Edge Daemon.

---

## 🏗️ System Architecture: The "Zero-Hop" Paradigm

Legacy AI microservices rely on multiple containers (API Gateway → Redis Queue → Vector DB → LLM Engine) communicating over TCP/IP, introducing massive serialization taxes and latency. 

The Ssense SLM Server merges these components into a single, unified FastAPI memory space backed by an integrated `AsyncLLMEngine`.

1. **API Gateway (`main.py`):** Handles SSE streaming, payload validation, and request routing natively.
2. **State & Queue Orchestrator (`memory_orchestrator.py`):** Replaces Redis. Manages cryptographic nonces, sliding-window rate limits, LRU-TTL caching, and **Request Coalescing** (if 50 users ask the same question, 1 leader executes on GPU; 49 followers subscribe to the memory stream).
3. **Hybrid RAG Engine (`rag_engine.py`):** Replaces Qdrant. Memory-maps `.safetensors` directly from the NVMe SSD. Executes BM25 Lexical + BGE Semantic math in background C-threads.
4. **LLM Engine (`engine.py`):** Direct in-process integration with vLLM. No HTTP hops to the GPU.
5. **Nginx Reverse Proxy:** Terminates TLS and disables proxy buffering for instantaneous Server-Sent Events (SSE) token delivery.

---

## 🧠 Model & Inference Stack

* **Base Model:** `Qwen/Qwen2.5-7B-Instruct` (Served natively in `bfloat16`).
* **Multi-LoRA Multiplexing:** Dynamically routes requests through two highly-specialized adapters residing in the same VRAM footprint:
  * `audit_lora`: Forensic JSON generation for automated policy auditing.
  * `chatbot_lora`: Conversational legal guidance via the DPDP Co-Pilot.
* **Dynamic VRAM Hardcap:** Automatically queries host GPU properties (`torch.cuda.get_device_properties`) and scales `gpu_memory_utilization` to enforce a strict **<= 32GB VRAM limit**, ensuring host OS stability on 48GB or 80GB DGX nodes.
* **1k Concurrency Boosters:** Employs **FP8 KV Caching** to halve memory-per-token, and **Prefix Caching** to share system prompts and RAG contexts across thousands of simultaneous users in $O(1)$ memory.

---

## 🔍 Zero-Copy Hybrid RAG Pipeline

Instead of naive vector retrieval, the server implements mathematically rigorous **Reciprocal Rank Fusion (RRF)**:
1. **Dense Semantic Search:** Uses BGE-Small (L2-normalized) via PyTorch/NumPy matrix dot-products.
2. **Sparse Lexical Search:** Uses `BM25Okapi` with strict alphanumeric tokenization and stopword removal to guarantee precise statutory section hits.
3. **XML-Structured Injection:** Retrieved chunks are injected into the prompt using `<document>`, `<metadata>`, and `<text>` tags, geometrically forcing the LLM's attention mechanism to cite accurate legal sections and preventing hallucination.

---

## 🛡️ Defense-in-Depth Security

Operating as a Zero-Exfiltration endpoint, the server features military-grade API shielding (`security.py`):

* **Cryptographic HMAC-SHA256:** Validates payloads against spoofing, Man-in-the-Middle (MITM), and origin-faking attacks. Enforces strict 30-second temporal windows and nonce-replay blocking.
* **Shannon Entropy Analysis:** ML heuristic that calculates character entropy. Instantly drops connections containing Base64/Hex obfuscated payloads or adversarial fuzzing.
* **ChatML Delimiter Guard:** Strips `<|im_start|>` and `<|im_end|>` sequences from user input to prevent role-hijacking attacks.
* **Anti-Distillation Shield:** Tracks user query frequency and detects probing keywords (e.g., "dump chain of thought") to block model extraction attempts.
* **Hallucination Logic Gates:** Validates all Audit outputs against `dpdp_schema.json`. Dynamically penalizes `dpdp_trust_score` if the model hallucinates a perfect score alongside critical statutory violations.

---

## 🚀 Setup & Deployment

The entire server, including cryptographic generation and Docker state management, is handled by a single orchestration script. 

### Prerequisites
* NVIDIA GPU with $\ge$ 32GB VRAM (Grace-Blackwell, Hopper, Ada, or Ampere).
* Docker v24+ with NVIDIA Container Toolkit.
* Ubuntu 22.04+ or Windows WSL2.

### Booting the Server

Simply execute the orchestrator from the root of the `slm-server` directory:

```bash
./scripts/run.sh

```

**What the script does automatically:**

1. Generates cryptographically secure `API_KEY` and `HMAC_SECRET` values into a `.env` file (if one doesn't exist). *Note: You will need this API key for the frontend/daemon.*
2. Verifies Docker daemon health.
3. Compiles the SOTA `Dockerfile` (based on highly-optimized `vllm/vllm-openai:v0.27.1`).
4. Downloads the Base Model and Ssense artifacts (LoRAs/RAG matrices) directly from Hugging Face if they are not found in the local `models/` directory.
5. Deploys the unified `slm-server` and `nginx` proxy via Docker Compose.

---

## 📡 API Reference

All inference endpoints require the following headers for authentication:

* `X-Ssense-API-Key`
* `X-Ssense-Signature` (HMAC-SHA256)
* `X-Ssense-Timestamp` (UTC Epoch ms)
* `X-Ssense-Nonce` (UUID)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Real-time diagnostic telemetry, queue depth, and VRAM orchestration status. |
| `/v1/audit` | `POST` | Deterministic, JSON-schema constrained DPDP policy auditing. |
| `/v1/chat/stream` | `POST` | RAG-augmented Conversational Co-Pilot via Server-Sent Events (SSE). |
```
