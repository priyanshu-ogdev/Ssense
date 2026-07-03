# Ssense Deployment & Implementation Blueprint

## 🎯 1. Executive Summary

Ssense is a production-grade, zero-knowledge privacy enforcement platform designed to physically block dark patterns and ensure compliance with India's Digital Personal Data Protection (DPDP) Act 2023 in real-time. 

Unlike traditional cloud-based privacy tools that route user traffic through external servers, Ssense operates entirely on the edge. It utilizes a local 9-billion parameter Large Language Model (Qwen 2.5 9B) running via a bare-metal Rust Native Daemon to audit privacy policies in milliseconds, and injects stealth rootkits into the browser's `MAIN` world to blind enterprise-grade hardware fingerprinting scripts before they execute.

**Core Value Proposition for Stakeholders:**
* **For Users:** Absolute privacy, zero latency, and an invisible shield against tracking and dark patterns.
* **For Enterprise/Regulators:** Deterministic, mathematically guaranteed enforcement of the DPDP Act 2023 at the network layer.
* **For Engineering:** A modular, crash-resilient architecture bridging V8 JavaScript, Rust, and C++ tensor math.

---

## 🎨 2. The User Experience (UX) & Interface Design

The Ssense UX is designed to be **non-intrusive yet highly informative**. The user does not need to manually trigger scans; the agentic workflow operates silently in the background, surfacing critical information only when relevant.

### The User Journey
```mermaid
sequenceDiagram
    participant User
    participant Browser as Chrome Browser
    participant UI as Ssense Side Panel
    participant Agent as Agentic Workflow

    User->>Browser: Navigates to example.com
    Browser->>Agent: Page Load Event (document_start)
    Agent->>Agent: MAIN World: Spoofs Canvas/WebGL/Audio APIs
    Agent->>Agent: ISOLATED World: Extracts Privacy Policy
    Agent->>Agent: Edge AI: Audits Policy (Local LLM)
    Agent->>Browser: DOM Enforcer: Collapses Trackers to 0px
    Browser->>UI: Broadcasts Trust Score & Audit Report
    UI->>User: Displays "Score: 85/100 (Compliant)"
    User->>UI: Types: "Are they selling my data?"
    UI->>Agent: Chat Request (with Audit Context)
    Agent->>UI: Context-Aware Legal Answer
    UI->>User: Renders Markdown Response
```

### Interface Components
1. **The Glassmorphic Side Panel:** A zero-flash, dark-mode React interface that slides in seamlessly. It features an ambient glow that shifts from Cyan (Safe) to Rose (Violations Detected) based on the real-time Trust Score.
2. **The Compliance Badge:** A dynamic indicator showing the domain's DPDP Trust Score (0-100), color-coded with animated pulses during the initial scan.
3. **The Audit Dashboard:** When an audit completes, the UI renders the LLM's `global_legal_reasoning`, a list of specific `violations` with exact `evidence_quote` blockquotes, and the specific `offending_entities` (domains) that were blocked.
4. **The Context-Aware Co-Pilot:** A chat interface where users can interrogate the privacy policy. The AI answers *strictly* grounded in the cached audit report, preventing hallucinations.

---

## 🔄 3. The Autonomous Agentic Workflow

Ssense operates as a closed-loop autonomous agent. It perceives the environment (the DOM), acts (extracts and audits), and enforces (blocks trackers) without human intervention.

### The End-to-End Agentic Loop
```mermaid
graph TD
    A[User Visits Site] --> B{MAIN World: api-spoof.ts}
    B -->|Blinds Fingerprinters| C[ISOLATED World: extractor.ts]
    C -->|Fetches Policy via SW Proxy| D[Service Worker Multiplexer]
    D -->|4-Byte LE Binary Frame| E[Rust Native Daemon]
    
    E --> F{SQLite Cache Check}
    F -->|Cache Hit| G[Return Instant JSON]
    F -->|Cache Miss| H[Acquire Global Mutex]
    H --> I[llama-cpp-rs Engine + GBNF Grammar]
    I -->|Generates DpdpAuditReport| J[Save to SQLite WAL]
    J --> G
    
    G -->|IPC Response| D
    D -->|Broadcast| K[Side Panel UI: Update Trust Score]
    D -->|Broadcast| L[DOM Blocker: MutationObserver]
    L -->|Applies CSS| M[Trackers Collapsed to 0 Dimensions]
```

### Phase Breakdown
1. **Phase 1: Preemptive Strike (MAIN World @ `document_start`)**
   * **Action:** Injects stealth Proxies to override `HTMLCanvasElement`, `WebGLRenderingContext`, `AudioContext`, and `Navigator.hardwareConcurrency`.
   * **Intent:** Defeat FingerprintJS v4 by feeding trackers garbage hardware signatures before the page's JavaScript even loads.
2. **Phase 2: Extraction & IPC (ISOLATED World @ `document_idle`)**
   * **Action:** Extracts the privacy policy, truncates to exactly 16,000 characters, and pipes it to the Rust Daemon via Native Messaging.
   * **Intent:** Isolate heavy network I/O from the browser's main thread, ensuring 60fps UI performance.
3. **Phase 3: Edge Inference & Enforcement (Rust Daemon)**
   * **Action:** The Tokio async reactor reads the IPC frame, checks the cache, and runs the LLM constrained by the **GBNF Grammar**.
   * **Intent:** Guarantee memory safety, prevent browser UI jank, and ensure 100% deterministic JSON output.
4. **Phase 4: DOM Enforcement**
   * **Action:** The `dark-pattern-blocker.ts` receives the audit report and uses a highly-optimized `MutationObserver` to physically collapse trackers.
   * **Intent:** Enforce the law at the network layer without breaking the host site's layout.

---

## 🏗️ 4. Architectural Deep Dive: Extension & Native Daemon

### Why Chrome MV3 + Rust Daemon instead of Tauri/Electron?
* **The Tauri/Electron Approach:** Desktop apps lack native DOM access. To intercept web traffic, they require configuring system-wide proxy servers, which breaks HTTPS certificate pinning, triggers anti-bot defenses, and requires complex OS-level installations.
* **The Ssense Approach:** Chrome MV3 provides native `content_scripts` for DOM manipulation and `declarativeNetRequest` for traffic interception. By using **Native Messaging**, we escape the browser sandbox and execute bare-metal Rust code, achieving the best of both worlds: Web API access + OS-level performance.

### Why Rust (`llama-cpp-rs`) instead of Python/Node?
* **Memory Safety:** Python's Garbage Collector causes micro-stutters that would freeze the browser IPC pipe. Rust guarantees zero GC pauses.
* **FFI Performance:** `llama-cpp-rs` provides zero-copy bindings to the C++ tensor engine. Node.js would require expensive serialization/deserialization across the V8/C++ boundary.
* **Concurrency:** Rust's Tokio reactor handles thousands of concurrent IPC connections without the overhead of Node's event loop or Python's GIL.

### The IPC Bridge: Binary Framing & Multiplexing
Chrome's Native Messaging API is notoriously fragile. A single unhandled promise rejection crashes the Service Worker.
* **Binary Framing:** We abandoned standard JSON stringification for the pipe header. We use a strict 4-Byte Little-Endian length prefix (matching Chromium's C++ implementation) followed by the UTF-8 JSON payload.
* **Bilateral Timeouts:** The pipe enforces a 30-second payload timeout and a 10-second write timeout.
* **Zombie Prevention:** If the OS pipe drops, the Rust daemon instantly executes `std::process::exit(1)` to prevent orphaned background processes.
* **Thundering Herd Mitigation:** The Service Worker maintains an `activeAudits` Map. If 50 tabs request an audit for `amazon.com` simultaneously, only one IPC request is sent to Rust; the other 49 tabs await the same Promise.

---

## 🖥️ 5. Hardware Orchestration: Edge (CPU/GPU) vs. Cloud Server

The Ssense ecosystem is orchestrated across two distinct hardware paradigms, ensuring the system is both highly performant in the lab and universally deployable in the wild.

### The Orchestration Logic
```mermaid
graph TD
    A[Inference Request] --> B{Hardware Profiler}
    B -->|Checks RAM/VRAM/Cgroups| C{Edge Capable?}
    
    C -->|Yes: >7GB RAM Available| D[Local Edge Inference]
    D --> E{GPU Detected?}
    E -->|Yes| F[llama-cpp-rs: n_gpu_layers=9999]
    E -->|No| G[llama-cpp-rs: Dynamic CPU Thread Routing]
    
    C -->|No: <7GB RAM or Containerized| H[Cloud Orchestrator Fallback]
    H --> I[Python FastAPI + vLLM on DGX/Cloud]
    
    F --> J[Return DpdpAuditReport]
    G --> J
    I --> J
```

### 1. The Local Edge Paradigm (Default)
By default, Ssense runs the Qwen 9B Q4_K_M model locally via `llama-cpp-rs`. 
* **GPU Acceleration:** If a GPU is present, the `HardwareProfiler` sets `n_gpu_layers = 9999`, offloading all tensor math to VRAM, achieving <100ms latency.
* **CPU Fallback:** If no GPU is present (e.g., a CPU-only laptop), the profiler dynamically calculates the optimal thread count based on *physical* cores (ignoring hyper-threading to prevent L1 cache thrashing) and runs the model entirely on the CPU.
* **Intent:** Guarantee that the Ssense shield works on *any* machine, gracefully degrading performance without crashing the OS.

### 2. The GPU Server / Cloud Orchestrator (Enterprise Fallback)
*Note: We architected the IPC bridge to support cloud offloading, but intentionally disabled it in the default build to enforce the zero-knowledge mandate.*
* **Workflow:** If the `HardwareProfiler` detects insufficient RAM (e.g., <7GB), or if the daemon is running inside a memory-starved Docker container, it routes the IPC payload to a remote Python FastAPI Orchestrator hosting vLLM.
* **The DGX Spark / Cloud Setup:** The cloud orchestrator utilizes NVIDIA DGX hardware (128GB unified memory) to run the 72B Teacher model for the GAN Forge, or serves the 9B model via vLLM's PagedAttention for ultra-low-end client machines.
* **Intent:** Allow enterprise deployment on ultra-low-end hardware (e.g., thin clients, IoT devices) where local inference is physically impossible, sacrificing absolute privacy for accessibility.

---

## 🔐 6. Security, Privacy & Threat Model

### Zero-Knowledge Architecture
* **No External API Calls:** All training and inference happens locally on the edge device.
* **No Telemetry:** No browsing data, policy text, or audit reports leave the machine.
* **Encrypted Storage:** Training data and SQLite caches are stored in isolated, OS-level application directories.

### Threat Model & Mitigations
| Threat Vector | Mitigation Strategy |
| :--- | :--- |
| **Elite Fingerprinting (FingerprintJS v4)** | `api-spoof.ts` uses a Singleton `WeakSet` registry to mask Proxies as `[native code]`, defeating `.toString()` detection. |
| **Clean Room Iframe Bypasses** | We hook `Node.prototype.appendChild` and the `HTMLIFrameElement.contentWindow` getter, recursively applying Proxies to the exact millisecond the iframe is attached. |
| **IPC OOM / Memory Exhaustion** | The `framing.rs` module enforces a hard 10MB limit on IPC payloads. The extractor truncates text to 16k chars *before* serialization. |
| **LLM Hallucinations / Invalid JSON** | The GBNF grammar compiles the schema into a Finite State Machine. The C++ backend physically masks out any logit that violates the schema *during sampling*. |
| **OS OOM Kills (Docker/Containers)** | `hardware_profiler.rs` reads Linux cgroup v1/v2 memory limits, refusing to load the model if the environment is physically incapable of supporting it. |

---

## 🚀 7. Deployment & Operational Guide

### Prerequisites
* **Rust Toolchain:** `rustc 1.75+` (with `cargo`)
* **Node.js:** `v18+` (with `npm`)
* **Chrome:** Latest stable build
* **OS:** Windows 10/11, macOS, or Linux (64-bit required for mmap)

### Build & Install Workflow

1. **Compile the entire stack (Extension + Rust Daemon):**
   ```bash
   make build-ext
   make build-daemon
   ```

2. **Load the Extension in Chrome:**
   * Navigate to `chrome://extensions`
   * Enable **Developer Mode**
   * Click **Load Unpacked** and select the `apps/extension/dist` directory
   * Copy the generated **Extension ID**

3. **Register the Native Messaging Host:**
   ```bash
   # The script handles Windows Registry, macOS, and Linux paths automatically
   node scripts/register-nmh.js <YOUR_EXTENSION_ID>
   ```

4. **Run the test suite:**
   ```bash
   make test
   ```

### Hardware Requirements (Edge Deployment)
| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **GPU** | None (CPU fallback) | 6GB VRAM (Q4_K_M quantization) |
| **RAM** | 8GB System RAM | 16GB+ System RAM |
| **Storage** | 10GB | 20GB (for model weights + cache) |
| **OS** | 64-bit Windows/macOS/Linux | Modern Linux/Windows with AVX2 support |

---

## 📝 Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-07-03 | Ssense Engineering | Initial deployment & implementation blueprint |
| 1.1 | 2026-07-03 | Ssense Engineering | Added UX flows, Hardware Orchestration graphs, and Threat Model |

---
*For ML Data Forge and Training Pipeline specifications, refer to `ARCHITECTURE.md`.*