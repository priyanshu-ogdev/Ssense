# Ssense — User Manual

Ssense is a browser extension that audits the sites you visit for
compliance with India's Digital Personal Data Protection (DPDP) Act 2023,
and lets you ask questions about any site's privacy practices.

---

## 1. What Ssense actually does

- Reads the **privacy policy text** of the site you're on (not the whole
  page, not your personal data) and sends it to your own SLM server for
  audit.
- Reads **cookie metadata** (name, domain, expiry, first/third-party flag)
  — not cookie *values* — as additional audit signal.
- Shows a **trust score** (0–100) and any statutory violations found, with
  citations to the specific DPDP Act sections involved.
- Tracks which sites you've visited, how long you spent on each, and their
  audit history, stored locally on your machine.
- Lets you **chat** with an AI about the current site's privacy policy.

Ssense does **not** send your browsing history, page content, or personal
data anywhere except to the SLM server you configure — and even then, only
policy text and cookie metadata, never page content or your account data.

---

## 2. Install & set up

### 2.1 Install the extension
1. Load the extension in Chrome (Developer Mode → Load Unpacked, pointing
   at the built `apps/extension/dist` folder, until this is published to
   the Chrome Web Store).
2. Click the Ssense icon in your toolbar — you'll see the popup.

### 2.2 Connect to your SLM server
Ssense needs a running SLM server to audit anything (see `DEPLOYMENT.md`
for setting that server up, if you don't already have one).

1. Click **⚙️ Settings** in the popup or side panel (or right-click the
   extension icon → Options).
2. Fill in:
   - **Server URL** — your server's address, e.g. `https://yourdomain.example.com`
     in production, or `http://localhost:8080` for local testing.
   - **API Key** and **HMAC Secret** — get these from whoever set up your
     server (they're generated during server setup, see `DEPLOYMENT.md`
     §3). These are *not* passwords you invent — they must match exactly
     what the server was configured with.
3. Click **Test Connection** — you should see a green checkmark with a
   tokens/second figure. If it fails, see Troubleshooting below.
4. Click **Save**.

You only need to do this once. It's stored locally on your machine and
never leaves it except to authenticate requests to your server.

---

## 3. Using Ssense day to day

### 3.1 The toolbar popup
Click the icon on any site to see:
- An **audit seal** — a stamped ring showing the site's trust score.
  Cyan/full ring = certified compliant, amber = under review, red = 
  violations flagged.
- **View Full Report** — opens the detailed side panel for this site.
- **Settings** — jump back to configuration.

If the site hasn't been audited yet, or you're on a browser system page
(like `chrome://settings`), the popup will tell you plainly rather than
show a stale or fake score.

### 3.2 The full report (side panel)
Click **View Full Report** for:
- The complete list of violations found, each with its DPDP Act statute
  reference and the exact policy text it was flagged from.
- A chat interface to ask follow-up questions about the site's policy.
- A **🕘 History** button — see §3.4.

### 3.3 The floating chat bubble
On every page, a small floating bubble appears (bottom-right). Click it to
ask quick questions about that specific site's privacy policy without
opening the full panel. Conversations are saved per-site — reopening the
bubble on a site you've chatted with before shows your past conversation.

### 3.4 History page
Shows every site Ssense has seen you visit: last visit time, number of
visits, total time spent, last audit score, and violations found. You can:
- **Filter** by typing part of a domain name.
- **Sort** by most recent, most time spent, or most violations.
- **Clear All** to wipe your local history (cannot be undone).

This data stays on your machine (browser-local storage) — it is not
uploaded anywhere.

---

## 4. Troubleshooting

**"Ssense is not configured yet"**
You haven't entered a Server URL / API Key / HMAC Secret yet, or one of
them is wrong. Go to Settings and re-check them against what your server
admin gave you.

**Test Connection fails / "Could not reach the extension background
service"**
- Confirm the server is actually running (`docker compose ps` on the
  server should show all services `healthy`, not just `running`).
- Confirm the Server URL is reachable from your machine (try opening
  `<server-url>/health` directly in a browser tab).
- If using `https://`, confirm the TLS certificate is valid — a
  self-signed dev certificate will show a browser warning that also
  blocks the extension's request; you may need to visit the URL directly
  once first and accept the certificate warning.

**Popup shows "No audit on record for this site yet"**
Normal for a site you haven't triggered an audit on. Open the full report
panel — it will run the audit if one hasn't happened yet.

**Chat bubble doesn't appear on a page**
It's intentionally skipped on browser system pages (`chrome://...`) and
the Chrome Web Store — there's no policy to audit there.

**History looks empty even though I've been browsing**
History is recorded per active, focused tab. Background tabs you never
switched to won't accumulate time (this is intentional — it reflects
actual attention, not just open tabs).

---

## 5. Privacy notes

- Cookie **values** are never read or transmitted — only metadata (name,
  domain, expiry, party classification).
- The extension will not fetch policy links that resolve to internal/
  private network addresses (e.g. `localhost`, your own router, cloud
  metadata endpoints) — this is a deliberate security protection, not a
  bug, if you ever see a site's audit silently skip a policy link.
- All history and chat data lives in the extension's own local storage on
  your device. Uninstalling the extension removes it. Server-side, only
  hashed/audited policy text is cached — never tied back to your personal
  browsing identity.
