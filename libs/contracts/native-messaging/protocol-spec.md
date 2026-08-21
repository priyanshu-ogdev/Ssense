# Ssense Native Messaging Protocol

This is the live contract between the Chrome extension and the Rust edge daemon.

## Transport

Chrome Native Messaging uses:

- 4-byte unsigned little-endian message length
- UTF-8 JSON payload
- maximum message size enforced by the daemon

## Requests

| Type | Purpose |
|---|---|
| `AUDIT_POLICY` | Audit a privacy policy locally |
| `CHAT` | Ask the local DPDP co-pilot |
| `GET_TRUST_SCORE` | Read a locally cached score |
| `HEALTH_CHECK` | Check daemon health |
| `DOWNLOAD_MODELS` | Explicitly download the optional offline model bundle |

## Download events

`DOWNLOAD_MODELS` is long-running. The daemon **does not** answer with a progress response that resolves the request.

Instead it emits:

```json
{
  "type": "DOWNLOAD_PROGRESS",
  "requestId": "…",
  "file": "Forensic Audit Model (INT4)",
  "pct": 42.5,
  "mbPerSec": 18.4
}
```

When all artifacts are complete:

```json
{
  "type": "STATUS",
  "requestId": "…",
  "status": "success",
  "message": "Offline models ready."
}
```

The extension treats `DOWNLOAD_PROGRESS` and `STATUS` as events and keeps the download request pending until the terminal response.

## Offline model storage

The daemon resolves its application data directory using the Rust `directories` crate. Models are stored under:

`<OS local application data>/Ssense/ssense-native-daemon/models`

On Windows this resolves inside the user's local `AppData` area rather than a repository-relative `../../ml` directory.

Downloads use `.part` files and HTTP `Range` requests. If Chrome, the daemon, or the network is interrupted, the next explicit Offline Mode download resumes from the existing partial file.

## Privacy UX contract

1. The extension starts in **Cloud · Fast** mode.
2. Installing the extension does **not** download the model bundle.
3. The user explicitly enables **Private · Offline** mode.
4. The extension asks the daemon to download the model bundle.
5. The popup displays file-level percentage and throughput.
6. Offline mode is marked ready only after the daemon reports success.
