# Ssense — Chrome Web Store User Manual & Release Guide

## What Ssense does

Ssense is a DPDP compliance shield for Chrome. It can inspect privacy policies, surface compliance findings, and provide a DPDP-focused co-pilot.

Ssense is **cloud-first** for speed. On-device inference is optional.

## First launch

1. Install Ssense from Chrome Web Store.
2. Open the Ssense extension popup.
3. Ssense starts in **Cloud · Fast** mode.
4. Open **Settings** only if your deployment requires a server URL/API credentials.
5. Browse to a supported website and open the full report.

### Why does it not download 9GB immediately?

It intentionally does not. Large GGUF/embedding artifacts are optional and are downloaded only after the user chooses Offline Mode.

## Enable Private / Offline Mode

1. Open the Ssense popup.
2. Find **AI ENGINE**.
3. Switch **Cloud · Fast** to **Private · Offline**.
4. Ssense asks the Rust Native Messaging daemon to download the local model bundle.
5. A progress bar shows the current artifact, percentage, and transfer speed.
6. Downloads use resumable HTTP ranges and `.part` files, so an interrupted download can continue later.
7. When all artifacts are present, the popup shows **Offline models are installed and ready**.
8. Local inference can now run without sending policy text to the cloud.

The model bundle is stored in the operating system's local application-data directory.

## Switch back to Cloud Mode

Open the popup and switch the mode control off.

The existing local models are retained. Turning Cloud Mode back on does not delete the model bundle.

## If a download is interrupted

Do not manually delete the `.part` files.

Reopen Ssense and enable Offline Mode again. The daemon checks the partial files and sends HTTP `Range` requests to continue from the previous byte.

## Chrome permissions

Ssense uses permissions required for:

- the side panel and popup UI
- browser tab context
- local extension storage
- content inspection/enforcement
- Native Messaging communication with the installed edge daemon

The Native Messaging host is a separately installed local component. Chrome Web Store installation alone cannot silently install arbitrary native software.

---

# Maintainer: Chrome Web Store release

## 1. Build the Rust daemon

Build the release binary from the repository root.

The native host registration script expects:

`apps/native-daemon/target/release/ssense-native-daemon.exe`

on Windows.

## 2. Build the extension

From `apps/extension`:

`npm install`

`npm run build`

The uploadable extension directory is:

`apps/extension/dist`

## 3. Publish the extension

Upload the extension package through Chrome Web Store Developer Dashboard.

After publication, Chrome assigns the extension its permanent extension ID.

## 4. Register the native host

The Native Messaging manifest must contain the published extension ID in `allowed_origins`.

Run:

`node scripts/register-nmh.js <PUBLISHED_EXTENSION_ID>`

The script registers the host for the current Windows user.

## 5. Keep native software separate

For a production Chrome Web Store release, distribute the Rust daemon/native host through a signed installer or a clearly documented companion-app installer.

Do not attempt to package the Rust executable as an automatically downloaded Web Store asset.

## 6. Store listing UX promises

Recommended listing language:

- **Cloud by default:** fast inference without a multi-gigabyte first-run download.
- **Optional Offline Mode:** download models only when the user chooses.
- **Resumable downloads:** interrupted model downloads continue from the last saved byte.
- **Local inference:** once installed, supported AI inference runs on the user's hardware.

Avoid claiming that the Web Store package itself installs the native daemon automatically.

## 7. Pre-submission QA

- Test fresh installation with no model files.
- Confirm the popup opens in Cloud Mode.
- Confirm no model download starts during installation.
- Confirm Offline Mode requires an explicit user action.
- Interrupt a model download and verify resume.
- Confirm the popup receives progress events without freezing.
- Confirm completion changes the mode to Offline.
- Switch back to Cloud Mode.
- Test the native host after a browser restart.
- Test the extension with the daemon unavailable.
- Test Windows AppData paths containing spaces.
- Verify the production native host uses the published extension ID.
- Remove development credentials before packaging.
- Review Chrome Web Store permissions and privacy disclosures.
