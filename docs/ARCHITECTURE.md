# Ssense System Architecture & Design Document

**Version:** 1.0.0  
**Status:** Production-Ready / SOTA  
**Last Updated:** 2026-06-24

---

## 1. Executive Summary

Ssense is an endpoint privacy enforcement platform designed to operate in hostile, heavily tracked web environments. The core architectural mandate was **Zero-Knowledge Compliance**: the ability to read, analyze, and enforce the Indian DPDP Act 2023 without transmitting a single byte of user browsing data to a remote server.

To achieve this, Ssense bridges the sandboxed V8 JavaScript engine of the Chrome browser directly to a bare-metal Rust daemon via OS-level Native Messaging. The daemon hosts a quantized 9-billion parameter Large Language Model (Qwen 2.5 9B) constrained by a GBNF (GPT-BNF) grammar, ensuring deterministic, mathematically valid JSON outputs.

---

## 2. The Four Pillars of Architecture

### 2.1 The Preemptive Shield (MAIN World Injection)
Standard content scripts run in an `ISOLATED` world, meaning they can manipulate the DOM but cannot alter the JavaScript execution environment of the host page. Elite trackers exploit this by reading hardware APIs directly.

**The SOTA Solution:**
We split the extension's content scripts into two execution phases:
1. **`api-spoof.ts` (MAIN World @ `document_start`):** Executes before any host JavaScript loads. It utilizes a Singleton `WeakSet` registry to override `Function.prototype.toString`, masking our Proxies as `[native code]`. It physically intercepts `HTMLCanvasElement`, `WebGLRenderingContext`, `AudioContext`, and `Navigator.hardwareConcurrency`.
2. **`dark-pattern-blocker.ts` (ISOLATED World @ `document_idle`):** Receives the audit report from the Rust daemon and uses a highly-optimized `MutationObserver` to collapse trackers to 0 dimensions without triggering Cumulative Layout Shift (CLS).

### 2.2 The IPC Bridge (Binary Framing & Multiplexing)
Chrome's Native Messaging API is notoriously fragile. A single unhandled promise rejection or stalled pipe will crash the Service Worker.

**The SOTA Solution:**
* **Binary Framing:** We abandoned standard JSON stringification for the pipe header. We use a strict 4-Byte Little-Endian length prefix (matching Chromium's C++ implementation) followed by the UTF-8 JSON payload.
* **Bilateral Timeouts:** The pipe enforces a 30-second payload timeout and a 10-second write timeout.
* **Zombie Prevention:** If the OS pipe drops, the Rust daemon instantly executes `std::process::exit(1)` to prevent orphaned background processes.
* **Thundering Herd Mitigation:** The Service Worker maintains an `activeAudits` Map. If 50 tabs request an audit for `amazon.com` simultaneously, only one IPC request is sent to Rust; the other 49 tabs await the same Promise and receive the DOM enforcement payload concurrently.

### 2.3 The Edge AI Engine (Rust & llama-cpp-rs)
Running a 5.5GB neural network in a background process requires extreme memory and thread management to avoid freezing the user's OS.

**The SOTA Solution:**
* **Async/Sync Boundary:** The Rust daemon uses a Tokio async reactor for IPC and Network I/O. However, `llama-cpp-rs` is CPU-bound and synchronous. We use `tokio::task::spawn_blocking` to offload tensor math to the OS thread pool, keeping the IPC reactor at 0% utilization.
* **Global Double-Checked Lock:** Because `llama.cpp` is not thread-safe for concurrent inferences, we wrap the engine in a `std::sync::Mutex`. The double-checked locking pattern ensures that if a second request arrives while the LLM is thinking, it waits for the lock, checks the SQLite cache, and instantly returns the result without re-running the LLM.
* **GBNF Grammar Enforcement:** To prevent the LLM from outputting conversational filler (e.g., "Here is your JSON:"), we compile the `dpdp_schema.json` into a GBNF grammar. The C++ backend physically masks out any logit that violates the schema during sampling.

### 2.4 The Memory Layer (SQLite WAL)
Querying the LLM for every page load would destroy battery life and CPU thermals. 

**The SOTA Solution:**
* **Connection Pooling:** We use `r2d2` to maintain a pool of 5 SQLite connections, allowing concurrent reads without Mutex bottlenecks.
* **Active Pruning:** On daemon boot, a sweeping `DELETE` query purges all audits older than 24 hours, preventing silent disk bloat.
* **Domain Normalization:** Domains are stripped of `www.` and lowercased before being hashed via SHA-256. This ensures `swiggy.com` and `www.swiggy.com` share the exact same cache key.

---

## 3. Threat Model & Security Boundaries

### 3.1 Fingerprinting Evasion
* **Threat:** Trackers call `.toString()` on overridden APIs to detect extensions.
* **Mitigation:** The `WeakSet` registry in `api-spoof.ts` intercepts `Function.prototype.toString` globally, returning `function name() { [native code] }` for any registered Proxy.
* **Threat:** Trackers spawn a blank `<iframe>` to access a "Clean Room" `contentWindow` with unmodified prototypes.
* **Mitigation:** We hook `Node.prototype.appendChild` and the `HTMLIFrameElement.contentWindow` getter, recursively applying our Proxies to the exact millisecond the iframe is attached to the DOM.

### 3.2 IPC & Memory Safety
* **Threat:** A malicious webpage sends a 10GB payload via `chrome.runtime.sendMessage` to trigger an Out-Of-Memory (OOM) crash in the Rust daemon.
* **Mitigation:** The `framing.rs` module enforces a hard `MAX_MESSAGE_SIZE_BYTES` limit of 10MB. Furthermore, the `extractor.ts` truncates the policy text to exactly 16,000 characters *before* serialization, dropping the IPC payload from ~900KB to ~64KB.
* **Threat:** The LLM outputs invalid JSON, crashing the TypeScript UI.
* **Mitigation:** The GBNF grammar makes invalid JSON mathematically impossible. As a fallback, the Rust daemon uses a Brace-Matching State Machine to extract the JSON object, and the TypeScript UI utilizes a Zero-Dependency Markdown parser to safely render the output.

### 3.3 Hardware & OS Constraints
* **Threat:** The daemon is installed on a 32-bit OS or a Docker container with a 2GB RAM limit, causing a fatal `mmap` or OOM crash.
* **Mitigation:** `hardware_profiler.rs` asserts `std::mem::size_of::<usize>() >= 8` to block 32-bit architectures. It also reads Linux cgroup v1/v2 memory limits to detect container constraints, refusing to load the model if the environment is physically incapable of supporting it.

---

## 4. Data Flow: The Audit Lifecycle

1. **Initialization:** User visits `example.com`. `api-spoof.ts` runs in the MAIN world, blinding Canvas and WebGL fingerprinters.
2. **Extraction:** `extractor.ts` runs at `document_idle`. It checks `<link rel="privacy-policy">`, falls back to DOM scraping, fetches the HTML via the Service Worker's CORS-bypassing proxy, and extracts the text.
3. **IPC Transmission:** The text is truncated to 16,000 chars, serialized, and framed with a 4-byte Little-Endian header.
4. **Rust Reactor:** The Tokio reactor reads the frame, parses the JSON, and checks the SQLite cache.
5. **Cache Miss:** The daemon acquires the global inference lock, loads the Qwen 9B model (if not already in RAM), and runs inference constrained by the GBNF grammar.
6. **Enforcement:** The resulting `DpdpAuditReport` is sent back to Chrome. The Service Worker broadcasts it to the Side Panel UI (updating the Trust Score) and the DOM Blocker (collapsing offending `<iframe>` and `<script>` tags).
7. **Caching:** The report is saved to SQLite with a 24-hour TTL. Subsequent navigations to `example.com` resolve in <1ms from the cache.

---

## 5. Maintenance & Future Roadmap

While the architecture is immutable, the web ecosystem is dynamic. Maintenance focuses on two areas:
1. **Prompt Engineering:** Updating `libs/contracts/constants/system_prompt.txt` as the DPDP Rules are amended or clarified by the Indian government.
2. **Tracker Evolution:** Monitoring elite fingerprinting libraries (e.g., FingerprintJS v4+) for new API probes (e.g., Battery API, Gamepad API) and adding corresponding stealth getters to `api-spoof.ts`.