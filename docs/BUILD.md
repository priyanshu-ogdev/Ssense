# Ssense Deployment & Implementation Blueprint

## 🎯 1. Executive Summary

Ssense is a production-grade, zero-knowledge privacy enforcement platform designed to physically block dark patterns and ensure compliance with India's Digital Personal Data Protection (DPDP) Act 2023 in real-time. 

Unlike traditional cloud-based privacy tools that route sensitive user traffic through external proxy servers—often introducing new privacy risks and breaking end-to-end encryption—Ssense operates entirely on the edge via a bare-metal Rust Native Daemon, falling back to a hardened FastAPI Virtual SLM Server only when mathematically necessary (e.g., on mobile constraints).

---

## 🔄 2. The Autonomous Agentic Workflow

Ssense operates as a closed-loop autonomous agent. It perceives the environment, acts (extracts and audits), and enforces (blocks trackers) without human intervention.

```mermaid
graph TB
    subgraph BrowserLayer["1. Browser Sandbox"]
        direction TB
        A1[MAIN World<br>Preemptive API Spoofing] --> A2[ISOLATED World<br>Extractor Truncates 16k]
    end

    subgraph NativeLayer["2. OS Level IPC"]
        direction TB
        A2 --> B1[4-Byte LE Binary Frame]
        B1 --> B2[Rust Daemon<br>60s Timeout Watchdog]
    end

    subgraph EnforcementLayer["3. Active DOM Removal"]
        direction TB
        B2 --> C1[MutationObserver Trigger]
        C1 --> C2[el.remove() Trackers]
        C2 --> C3[React UI Badge Update]
    end
```

### Phase Breakdown
1. **Preemptive Strike (MAIN World @ `document_start`):** `api-spoof.ts` injects stealth Proxies to override `HTMLCanvasElement`, `WebGLRenderingContext`, `AudioContext`, and `Navigator.hardwareConcurrency` to blind enterprise fingerprinters before they execute.
2. **Extraction & IPC (ISOLATED World @ `document_idle`):** Extracts the privacy policy, truncates to exactly 16,000 characters to prevent VRAM explosion, and pipes it to the Rust Daemon via Native Messaging.
3. **Edge Inference & Enforcement:** The Rust Daemon checks the SQLite cache. On a cache miss, it acquires a Global Mutex, runs the `Qwen3.5-9B` LLM constrained by GBNF grammar, and saves the result.
4. **DOM Enforcement:** `dark-pattern-blocker.ts` uses a `MutationObserver` to actively remove (`el.remove()`) offending trackers from the DOM rather than just hiding them with CSS.

---

## 🏗️ 3. Architectural Deep Dive: Extension & Native Daemon

### Why Chrome MV3 + Rust Daemon?
Tauri/Electron requires system-wide proxy servers, breaking HTTPS pinning and introducing massive latency. Chrome MV3 provides native `content_scripts` for direct DOM manipulation. By using **Native Messaging**, we securely escape the browser sandbox and execute bare-metal Rust code, achieving Web API access + OS-level performance without compromising system integrity.

### The IPC Bridge: Binary Framing & Multiplexing
Chrome's Native Messaging API is notoriously fragile; a single unhandled stringification error will crash the pipe and spawn zombie processes. Ssense implements a highly resilient IPC pipe:
* **Binary Framing (`framing.rs`):** We use a strict 4-Byte Little-Endian length prefix matching Chromium's internal C++ implementation precisely.
* **Bilateral Timeouts:** The pipe enforces a 60-second inference timeout (`INFERENCE_TIMEOUT_SECS = 60`). If the LLM thread hangs, the daemon gracefully kills the thread instead of locking up the OS.
* **Zombie Prevention:** If the OS pipe drops or fails to send to the writer task, the Rust daemon executes `std::process::exit(1)` to prevent orphaned background processes.
* **RawEnvelope Rescue:** If a JSON payload from the LLM fails to parse, a `RawEnvelope` rescue mechanism intercepts the failure and securely communicates a standardized error back to the browser without crashing the daemon.

---

## 🖥️ 4. Hardware Orchestration: Edge vs. Cloud Server

> **⚠️ `LOCAL_DAEMON` mode and the dual-mode engine selector described in
> this section have been removed.** The extension now talks to the SLM
> server exclusively — see `docs/DEPLOYMENT.md`. The subsections below are
> kept for historical design context only.

### The Local Edge Paradigm (`LOCAL_DAEMON` Mode) *(removed)*
Ssense runs the Qwen 9B model locally via `llama-cpp-rs`.
* **Hardware Profiler (`hardware_profiler.rs`):** Before booting the engine, the profiler interrogates the host operating system. 
* **GPU Acceleration:** If a compatible GPU is detected, it sets `n_gpu_layers = 9999`, offloading all tensor math to VRAM and achieving <100ms latency.
* **CPU Fallback:** If no GPU is present, the profiler dynamically calculates the optimal thread count based strictly on *physical* cores (ignoring hyper-threading to prevent L1 cache thrashing).

### The Cloud Virtual SLM Server (now the only mode)
Ssense runs on a Virtual SLM Server (`apps/slm-server`) behind Nginx.
* ~~**Dynamic Engine Selector:** Users can toggle between `AUTO` (local first, cloud failover), `LOCAL_DAEMON`, and `CLOUD_SERVER` directly from the Extension UI.~~ *(removed — no selector, server-only)*
* ~~**Exponential Backoff:** If the local daemon disconnects (or the host runs out of RAM), the service worker intelligently fails over to the Virtual SLM Server using an exponential backoff retry schedule.~~ *(removed — nothing to fail over from)*

---

## 🔐 5. Security, Privacy & Threat Model

| Threat Vector | Ssense Mitigation Strategy |
| :--- | :--- |
| **Elite Fingerprinting** | `api-spoof.ts` uses a Singleton `WeakSet` registry to mask Proxies as `[native code]`, defeating `.toString()` detection by FingerprintJS v4. |
| **Clean Room Iframe Bypasses** | Hooks `Node.prototype.appendChild` and `contentWindow` getters to recursively poison iframe environments. |
| **IPC OOM / Memory Exhaustion** | `framing.rs` enforces a hard 10MB payload limit. Extractor strictly truncates to 16,000 characters before serialization. |
| **OS OOM Kills** | `HardwareProfiler` reads Linux cgroup limits and refuses to load models if the host environment is incapable. |
| **Origin Spoofing & API Replay** | Web Crypto computes `HMAC-SHA256` signatures (`X-Ssense-Signature`). The Cloud Server checks a strict 30s window and caches `X-Ssense-Nonce` to completely block replay attacks. |
| **Model Extraction Probes** | `AntiExtractionGuard` regex engine screens inputs for chain-of-thought probes (`HTTP 429`) and injects statutory watermarks (`[Ssense-DPDP-Act-2026-Certified-Provenance]`). |
| **Network Telemetry Exfiltration** | Intercepts `window.fetch` and `XMLHttpRequest` in the MAIN world to forcefully scrub tracking headers (`X-Telemetry`, `X-Mixpanel`). |