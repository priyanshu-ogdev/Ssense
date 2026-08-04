# Ssense

> **Zero-Knowledge Edge AI for Privacy Policy Analysis and Browser-Side Enforcement.**

Ssense is a production-grade, edge-native privacy platform that combines local AI analysis with browser-side privacy controls to help users understand data collection practices and reduce exposure to enterprise-grade tracking techniques. 

Unlike traditional cloud-based privacy tools, Ssense operates entirely on the edge. It utilizes a local 9-billion parameter Large Language Model running via a bare-metal Rust Native Daemon to audit privacy policies in milliseconds, and injects stealth rootkits into the browser's `MAIN` world to blind hardware fingerprinting scripts before they execute.

**Core Value Proposition:**
* **For Users:** Absolute privacy, zero latency, and an invisible shield against tracking and dark patterns.
* **For Enterprise/Regulators:** Deterministic, mathematically guaranteed enforcement of the DPDP Act 2023 at the network layer.
* **For Engineering:** A modular, crash-resilient architecture bridging V8 JavaScript, Rust, and C++ tensor math.

---

## 🛡️ Key Features

### Edge & Cloud Dual-Mode AI Architecture
* **On-Device & Dual-Tier Inference:** Audits privacy policies locally using a specialized **9-Billion Parameter Small Language Model (`Qwen/Qwen3.5-9B`)** fine-tuned via Unsloth (`rsLoRA` + `SimPO`) with GGUF / vLLM multi-LoRA deployment options.
* **Dual-Mode Failover (`AUTO` / `LOCAL_DAEMON` / `CLOUD_SERVER`):** Automatically routes inference requests to the bare-metal Rust Native Daemon (`localhost`) for zero-latency local execution, with seamless exponential-backoff failover to the hardened FastAPI Virtual SLM Server (`apps/slm-server`).
* **Deterministic Output:** Uses GBNF (GPT-BNF) grammar to physically constrain the C++ tensor engine, guaranteeing 100% valid JSON schema compliance (`dpdp_schema.json`) with zero parsing failures.
* **Persistent LRU Policy Cache:** Features a SHA-256 digest-keyed inference cache (`completedAuditsCache` with 30-minute TTL) across open browser tabs, serving deduplicated DPDP audit reports in `<5ms`.
* **4-Byte LE Binary Framing:** Employs ultra-reliable Little-Endian binary IPC framing between Chrome MV3 and the Rust Daemon, completely eliminating UI lockups and JSON parse errors.

### Enterprise Cryptographic Anti-Theft & Model Protection
* **HMAC-SHA256 Challenge-Response Authentication:** Every request to the Virtual SLM Server is cryptographically signed using Web Crypto (`X-Ssense-Signature`, `X-Ssense-Timestamp`, `X-Ssense-Nonce`). The server enforces a strict 30-second timestamp window and nonce replay prevention cache to thwart Origin spoofing and API key theft.
* **Statutory Model Extraction Shield (`AntiExtractionGuard`):** Actively monitors prompt tokens and domain payloads for systematic distillation or chain-of-thought extraction probing. Offending requests are throttled with `HTTP 429 Too Many Requests` and injected with verifiable statutory watermarks (`[Ssense-DPDP-Act-2026-Certified-Provenance]`).

### SOTA Preemptive Privacy Enforcement & DOM Shielding
* **MAIN World API Spoofing:** Injects stealth Proxies at `document_start` to override `HTMLCanvasElement`, `WebGLRenderingContext`, `AudioContext`, and `Navigator.hardwareConcurrency`.
* **Anti-Detection & Getter Bypass:** Uses a Singleton `WeakSet` registry to mask Proxies as `[native code]` and hooks `HTMLIFrameElement.prototype.contentWindow` to recursively sanitize clean-room iframes against FingerprintJS v4.
* **Active DOM Node Removal (`dark-pattern-blocker.ts`):** Goes beyond visual hiding by actively stripping (`el.remove()`) offending third-party `<script>` and `<iframe>` tracking nodes from the live DOM and scrubbing image tracking URLs (`el.removeAttribute('src')`), backed by instant CSS rules (`.ssense-blocked-element`).
* **Network Telemetry Interception (`api-spoof.ts`):** Hooks `window.fetch` and `XMLHttpRequest` in the MAIN world to abort requests targeting blocked tracker domains (`__ssenseBlockedDomains`) and scrub invasive telemetry headers (`X-Telemetry`, `X-Tracker`, `X-Analytics`, `X-Mixpanel`).

### Agentic UX & Forensic Reporting
* **Granular Shield Controls Modal (`🛡️ Shield`):** Allows users to toggle active third-party tracker blocking, hardware API spoofing, and Global Privacy Control (`GPC`) signals on the fly.
* **Dual-Scorecard & Obfuscation Telemetry:** Displays both the calibrated `DPDP Trust Score` (0-100) and an `Obfuscation Subtlety Rating` (0-100) highlighting complex corporate legalese intentionally designed to obscure statutory violations.
* **Forensic Audit Report Export (`📥 Export Report`):** One-click generation and instant download of structured Markdown forensic compliance reports (`ssense_audit_*.md`) complete with statute citations, evidence quotes, and semantic justifications.

---

## 🏗️ Architecture Workflow

Ssense bridges the sandboxed Chrome V8 engine with bare-metal OS execution via Native Messaging and Cloud orchestration.

```mermaid
graph TB
    subgraph Browser["Chrome Extension (MV3)"]
        direction TB
        A1[Preemptive Strike<br>Spoof Canvas/WebGL APIs] --> A2[DOM Extraction<br>Policy Truncation to 16k]
        A2 --> A3{Cache Check}
        A3 -- Hit --> A4[DOM Enforcement<br>el.remove() Trackers]
        A3 -- Miss --> A5[Service Worker IPC Dispatch]
    end

    subgraph DualModeInference["Dual-Mode Orchestrator"]
        direction TB
        A5 -- Local Memory > 7GB --> B1[Rust Native Daemon<br>Local SQLite WAL]
        B1 --> B2[llama-cpp-rs Inference<br>n_gpu_layers or CPU threads]
        B2 --> B3[4-Byte LE Binary Frame Response]
        
        A5 -- Mobile / Fallback --> C1[Virtual SLM Server<br>HMAC-SHA256 Auth]
        C1 --> C2[O 1 Redis Queue<br>SSE Streaming]
        C2 --> C3[Multi-LoRA vLLM Engine]
    end

    B3 --> A4
    C3 --> A4
```

### The Agentic Workflow
1. **Preemptive Strike (`document_start`):** `api-spoof.ts` runs in the MAIN world, blinding Canvas/WebGL fingerprinters before the page's JavaScript loads.
2. **Extraction (`document_idle`):** `extractor.ts` fetches the privacy policy via a CORS-bypassing Service Worker proxy and truncates it to 16,000 characters.
3. **Edge Inference:** The Rust Daemon checks the SQLite cache. On a miss, it acquires a Global Mutex, runs the LLM constrained by the GBNF grammar, and saves the result.
4. **DOM Enforcement:** The audit report is broadcast back to Chrome, where `dark-pattern-blocker.ts` physically collapses offending trackers.

---

## 📂 Repository Structure

```text
ssense/
├── .github/workflows/        # CI/CD pipelines
├── apps/
│   ├── extension/            # Chrome MV3 Extension (React + TypeScript)
│   │   ├── background/       # Service Worker & Native Messaging Bridge
│   │   ├── content/          # DOM Enforcer, Extractor, MAIN world API Spoofer
│   │   └── sidebar/          # Glassmorphic Co-Pilot UI
│   │
│   ├── native-daemon/        # Rust Edge AI Engine
│   │   └── src/
│   │       ├── cache/        # SQLite WAL Memory Layer
│   │       ├── inference/    # llama-cpp-rs, GBNF Grammar, Hardware Profiler
│   │       └── messaging/    # 4-Byte LE Binary Framing
│   │
│   └── slm-server/           # FastAPI Cloud Backend
│       ├── main.py           # HMAC Auth & SSE Streaming
│       ├── engine.py         # Multi-LoRA vLLM PagedAttention
│       └── redis_queue.py    # O(1) Queue & SHA-256 Coalescing
│
├── docs/
│   ├── ARCHITECTURE.md       # ML Data Forge & Training Pipeline (vLLM/Unsloth)
│   ├── BUILD.md              # Deployment, UX, and Threat Model specifications
│   ├── DESIGN.md             # Technical design specs: GAN Forge & Process Isolation
│   ├── SLM_Server_Architecture.md # Docker 4-tier orchestrator details
│   └── project_report.md     # Comprehensive system overview
│
├── libs/
│   ├── contracts/            # Single Source of Truth (JSON Schemas, Prompts)
│   └── rust-utils/           # Shared Workspace Utilities (Hashing, Normalization)
│
├── ml/                       # Python Training Forge & Certification Engine
│   ├── data-forge/           # 72B Teacher Synthesizer & Contrastive DPO Alignment
│   ├── evals/                # Universal backend_loader.py & 13-Pillar Certification Suite
│   ├── models/               # Local model checkpoint directories (72B Teacher & 9B Base)
│   └── slm-training/         # Unsloth SFT & SimPO scripts with OS-level `spawn`
│
├── scripts/
│   ├── 01_prepare_data.sh    # Stage 1: Data preparation, GAN forge & Unsloth formatting
│   ├── 02_train_models.sh    # Stage 2: VRAM airlock & dual Qwen 3.5 9B SFT/SimPO training
│   ├── 03_evaluate_models.sh # Stage 3: Functional & adversarial 13-pillar certification
│   └── register-nmh.js       # Cross-platform Native Messaging Host registrar
│
└── Makefile                  # POSIX-compliant Build & Test Orchestrator
```

---

## 🚀 Prerequisites & Installation

### Prerequisites
* **Rust Toolchain:** `rustc 1.75+` (with `cargo`)
* **Node.js:** `v18+` (with `npm`)
* **Google Chrome:** Latest stable build
* **OS:** Windows 10/11, macOS, or Linux (64-bit required for `mmap`)

### Recommended Hardware
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | None (CPU fallback) | 6GB VRAM (for Q4_K_M quantization) |
| **RAM** | 8GB System RAM | 16GB+ System RAM |
| **Storage** | 10GB | 20GB (for model weights + cache) |

### Build & Install Workflow

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/ssense.git
   cd ssense
   ```

2. **Compile the entire stack (Extension + Rust Daemon):**
   ```bash
   make build-ext
   make build-daemon
   ```

3. **Load the Extension in Chrome:**
   * Navigate to `chrome://extensions`
   * Enable **Developer Mode**
   * Click **Load Unpacked** and select the `apps/extension/dist` directory
   * Copy the generated **Extension ID** (e.g., `abcdefghijklmnop`)

4. **Register the Native Messaging Host:**
   ```bash
   # The script handles Windows Registry, macOS, and Linux paths automatically
   node scripts/register-nmh.js <YOUR_EXTENSION_ID>
   ```

5. **Run the test suite:**
   ```bash
   make test
   ```

---

## 🔐 Security Model & Threat Mitigation

Ssense is designed around three principles:

1. **Local Processing (Zero-Knowledge):** Privacy analysis and inference happen entirely on-device. No external API calls or telemetry.
2. **Least Privilege IPC:** Browser and native components communicate through a constrained, timeout-hardened (60s) binary IPC interface with strict payload size limits (10MB max).
3. **Defense in Depth:** Tracking mitigation combines API-level controls (MAIN world spoofing), DOM analysis (MutationObserver), and policy auditing (Edge AI).

### Threat Model
| Threat Vector | Mitigation Strategy |
| :--- | :--- |
| **Elite Fingerprinting** | `api-spoof.ts` masks Proxies as `[native code]` via a Singleton `WeakSet`, defeating `.toString()` detection. |
| **Clean Room Iframes** | Hooks `Node.prototype.appendChild` and `contentWindow` getters to recursively poison iframe environments. |
| **LLM Hallucinations** | GBNF grammar compiles the schema into an FSM. The C++ backend physically masks invalid logits *during sampling*. |
| **OS OOM Kills** | `HardwareProfiler` reads Linux cgroup limits and dynamically routes CPU threads based solely on physical cores. |
| **Model Distillation & Extraction** | `AntiExtractionGuard` regex engine screens inputs for chain-of-thought probes (`HTTP 429`) and injects statutory watermarks (`[Ssense-DPDP-Act-2026-Certified-Provenance]`). |
| **Origin Spoofing & API Replay** | Web Crypto computes `HMAC-SHA256` signatures (`X-Ssense-Signature`) over request payloads. Server checks `X-Ssense-Timestamp` (30s window) and caches `X-Ssense-Nonce` to block replay attacks. |
| **Telemetry Exfiltration** | Intercepts `window.fetch` and `XMLHttpRequest` in the MAIN world, aborting requests to blocked domains (`__ssenseBlockedDomains`) and scrubbing tracking headers. |

---

## 📚 Documentation

* **[Architecture & ML Pipeline](docs/ARCHITECTURE.md):** Deep dive into the GAN Forge, vLLM data generation, Unsloth training, and DGX hardware orchestration.
* **[Build & Deployment Blueprint](docs/BUILD.md):** Detailed UX flows, agentic workflow diagrams, and edge vs. cloud orchestration logic.
* **[Design & Technical Specifications](docs/DESIGN.md):** Comprehensive technical specifications of the GAN Forge loop, Unsloth `multiprocessing spawn`, `max_prompt_length`, and process isolation.
* **[Virtual SLM Server](docs/SLM_Server_Architecture.md):** Detailed breakdown of the 48GB VRAM cloud orchestrator, SSE streaming, and Redis queueing.
* **[Project Report](docs/project_report.md):** The comprehensive overview of the entire stack.

---

## ⚖️ Disclaimer

Privacy policy analysis generated by AI should be treated as informational assistance and not as legal advice. Users should consult qualified legal professionals for regulatory or compliance decisions regarding the DPDP Act 2023 or other privacy regulations.

---

## 📄 License

Licensed under the [Apache-2.0 License](LICENSE).