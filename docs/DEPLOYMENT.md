# Ssense — Deployment & Operations Guide

This is the real, in-order sequence to stand up the SLM server and connect
the extension to it. Written for whoever runs the server (your "friend" /
ops person), not end users — see `USER_MANUAL.md` for that.

---

## 1. Prerequisites

- A machine with an NVIDIA GPU, **32GB VRAM minimum** (current default
  profile — see §5 for switching to 48GB later).
- Docker + Docker Compose, with the NVIDIA Container Toolkit installed
  (`runtime: nvidia` in `docker-compose.yml` depends on this).
- A domain name pointed at this machine, if deploying to production
  (self-signed certs work for local/dev only — see §4).

---

## 2. First-time setup

```bash
cd apps/slm-server
cp .env.example .env
```

Generate every required secret — **do not skip this, the server will
refuse to boot in production without real values**:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # → SSENSE_API_KEYS
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # → SSENSE_HMAC_SECRET
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # → REDIS_PASSWORD
```

Edit `.env`:
```ini
SSENSE_ENV=production
SSENSE_API_KEYS=<generated value>
SSENSE_ENTERPRISE_API_KEYS=          # optional — subset of the above for priority queuing
SSENSE_HMAC_SECRET=<generated value>
SSENSE_ALLOWED_ORIGINS=              # your extension's chrome-extension://<id> origin
REDIS_PASSWORD=<generated value>
```

You'll give the `SSENSE_API_KEYS` and `SSENSE_HMAC_SECRET` values to
whoever configures the extension (see `USER_MANUAL.md` §2.2) — they go in
the extension's Settings page exactly as generated here.

---

## 3. TLS certificates

See `apps/slm-server/nginx/certs/README.md` for the full instructions.
Short version:

**Local/dev (self-signed):**
```bash
cd apps/slm-server/nginx/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssense.key -out ssense.crt -subj "/CN=localhost"
```

**Production:** use Let's Encrypt or your org's CA — place the issued
files at `ssense.crt` / `ssense.key` in that same directory, and set up
automatic renewal (Let's Encrypt certs expire every 90 days).

---

## 4. Bring the stack up

```bash
cd apps/slm-server
docker compose up -d
docker compose ps
```

Everything should reach `healthy` — not just `running`. Startup order is
enforced automatically (`nginx` won't start serving until `slm-gateway` is
healthy; `slm-gateway` won't start until `vllm-engine`, `redis-queue`, and
`qdrant-db` are all healthy). The `vllm-engine` healthcheck allows up to
**180 seconds** for model + LoRA adapter loading before it's expected to
report healthy — a longer wait here is normal, not a failure.

Watch the vLLM startup log for the real KV-cache capacity it allocated:
```bash
docker compose logs -f vllm-engine | grep "GPU blocks"
```
This tells you the *actual* measured KV budget, not the estimate the
current `--max-num-seqs 64` setting was reasoned from — see §5.

Confirm the whole path works end-to-end:
```bash
curl -k https://localhost/health
```
(`-k` skips cert validation, only needed against a self-signed dev cert.)

---

## 5. Tuning for your actual hardware

The current `docker-compose.yml` targets **32GB VRAM** (see the comments
directly in `vllm-engine.command`). A commented **48GB profile** sits
right below it in the same file — swap which `command:` block is active
if you move to bigger hardware later; don't run both.

The `--max-num-seqs 64` value is a reasoned estimate, not a measured one
(the exact `Qwen3.5-9B` architecture config wasn't available when this was
tuned). After first boot, check the real `# GPU blocks: X` line from the
vLLM log (§4) and watch for `Sequence group ... preempted` warnings under
load — that's vLLM telling you `max-num-seqs` is set too high for the
real memory available. Adjust down if you see that; consider raising it
if you have consistent headroom and want more throughput.

---

## 6. Operational notes

**Redis is internal-only, on purpose.** `redis-queue` and `qdrant-db` do
not publish ports to the host — they're reachable only inside the Docker
network. Do not add `ports:` mappings back for them in production; that
would expose an authenticated-but-still-sensitive queue/vector store
directly to the internet.

**Nginx is the only public entry point.** `slm-gateway` itself has no host
port mapping either — everything goes through Nginx on 80/443. This gives
you SSE-correct streaming (`proxy_buffering off` on `/v1/chat` and
`/v1/audit` specifically), TLS termination, and a first line of
rate/connection limiting before requests ever reach the app.

**"10,000 concurrent users" means held-open connections, not simultaneous
GPU-resident requests.** The Redis queue accepts up to `SSENSE_MAX_QUEUE_DEPTH`
(default 5000) queued requests; only `--max-num-seqs` (currently 64) are
ever actually running against the GPU at once. This is correct, deliberate
design, not a shortfall — see `docs/SLM_Server_Architecture.md` for the
full math.

**Logs:**
```bash
docker compose logs -f slm-gateway    # app-level request handling, security events
docker compose logs -f vllm-engine    # model loading, inference errors, KV pressure
docker compose logs -f nginx          # proxy-level access/error logs
```

**Restarting after a `.env` change:**
```bash
docker compose up -d --force-recreate slm-gateway
```
(Redis password changes require recreating `redis-queue` too, and then
`slm-gateway` again, since the connection string is baked in at container
start.)

---

## 7. Full request lifecycle (for reference)

1. Extension content script (`extractor.ts`) extracts policy text + cookie
   metadata from the current page, with an SSRF guard blocking any policy
   link that resolves to a private/internal address.
2. Sent to the background service worker, which signs the request
   (HMAC-SHA256 + nonce + timestamp) and posts it to Nginx over HTTPS.
3. Nginx terminates TLS, applies coarse rate/connection limiting, and
   proxies to `slm-gateway` with buffering disabled for the SSE routes.
4. `slm-gateway` (FastAPI) verifies the API key + HMAC signature + nonce
   (replay-checked against shared Redis state), checks for prompt-injection
   patterns and extraction/distillation abuse, then either serves a cached
   result or queues the request in Redis.
5. The request is pulled off the queue in priority order (enterprise tier
   is derived from the authenticated API key, never client input) and sent
   to `vllm-engine`, which streams tokens back through the SLM server as
   Server-Sent Events.
6. The extension's `fetchServerStream()` consumes the SSE stream, and the
   result is persisted locally (audit → `history-store.ts`, chat →
   `chat-store.ts`) and shown in the UI.
