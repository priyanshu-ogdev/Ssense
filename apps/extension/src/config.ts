// apps/extension/src/config.ts
//
// The extension only ever talks to one of two fixed endpoints. There is
// intentionally no way to type an arbitrary URL into the UI anymore —
// that was a support/security liability (typos, phishing endpoints,
// stale tunnels). Change these two constants at build time if the
// production domain changes; end users just pick "Local" or "Online".

/** Local SLM server, e.g. `docker compose up` on the same machine. */
export const LOCAL_SERVER_URL = 'http://localhost:8080';

/** Hosted production SLM server, behind Nginx/TLS. */
export const ONLINE_SERVER_URL = 'https://api.ssense.app';

export type ServerMode = 'auto' | 'local' | 'online';

export const DEFAULT_SERVER_MODE: ServerMode = 'auto';

export function urlForMode(mode: Exclude<ServerMode, 'auto'>): string {
  return mode === 'local' ? LOCAL_SERVER_URL : ONLINE_SERVER_URL;
}
