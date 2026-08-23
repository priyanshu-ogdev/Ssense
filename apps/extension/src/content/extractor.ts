// apps/extension/src/content/extractor.ts

export {}; // marks this file as an ES module so `declare global` below is valid

declare global {
  interface Window {
    __ssenseExtractorLoaded?: boolean;
  }
}

// Guard the whole file against re-injection into an already-loaded page
// realm (see the identical guard + comment in dark-pattern-blocker.ts for
// why this happens and what it fixes).
if (!window.__ssenseExtractorLoaded) {
window.__ssenseExtractorLoaded = true;

console.log('[Ssense] Policy Extractor injected.');

// 🚀 SOTA FIX: Safely resolves relative paths while preventing Protocol Smuggling
function resolveUrl(path: string): string | null {
  try {
    // document.baseURI respects <base href="..."> tags if present, 
    // otherwise defaults to window.location.href.
    const resolved = new URL(path, document.baseURI).href;
    
    // CRITICAL SECURITY CHECK: Prevent Protocol Smuggling
    // new URL() will happily parse "file:///" or "javascript:" URIs.
    // We must strictly enforce HTTP/HTTPS before the SW attempts to fetch.
    if (resolved.startsWith('http://') || resolved.startsWith('https://')) {
      return resolved;
    }
    return null;
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// SSRF GUARD
// ═══════════════════════════════════════════════════════════════
// A malicious page could publish a "privacy policy" link pointing at
// localhost, a private LAN address, or a cloud metadata endpoint
// (169.254.169.254). The service worker would then fetch it with the
// extension's network privileges. Block anything that resolves to a
// non-public host BEFORE it's ever sent to the background PROXY_FETCH.
const BLOCKED_HOSTNAME_PATTERNS = [
  /^localhost$/i,
  /^127\./,                    // loopback
  /^0\.0\.0\.0$/,
  /^10\./,                     // RFC1918
  /^172\.(1[6-9]|2\d|3[01])\./,// RFC1918
  /^192\.168\./,                // RFC1918
  /^169\.254\./,                // link-local / cloud metadata (AWS/GCP/Azure)
  /^\[?::1\]?$/,                 // IPv6 loopback
  /^\[?fe80:/i,                  // IPv6 link-local
  /^\[?fc[0-9a-f]{2}:/i,         // IPv6 unique local
];

function isSafePublicUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return !BLOCKED_HOSTNAME_PATTERNS.some((p) => p.test(parsed.hostname));
  } catch {
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════
// AUTHORITATIVE DISCOVERY (Check <head> first)
// ═══════════════════════════════════════════════════════════════
function findAuthoritativePolicyUrl(): string | null {
  // 1. Check <link rel="privacy-policy" href="...">
  const linkTag = document.querySelector('link[rel~="privacy-policy"]');
  if (linkTag) {
    const href = (linkTag as HTMLLinkElement).getAttribute('href');
    if (href) return resolveUrl(href);
  }

  // 2. Check <meta name="privacy-policy" content="...">
  const metaTag = document.querySelector('meta[name="privacy-policy"]');
  if (metaTag) {
    const content = metaTag.getAttribute('content');
    if (content) return resolveUrl(content);
  }

  return null;
}

// ═══════════════════════════════════════════════════════════════
// FALLBACK DISCOVERY (DOM Scanning)
// ═══════════════════════════════════════════════════════════════
const PRIVACY_LINK_PATTERNS = [
  /privacy/i,
  /data[-\s]?protection/i,
  /cookie[-\s]?policy/i,
];

const PRIVACY_LINK_SELECTORS = [
  'a[href*="privacy"]',
  'a[href*="data-protection"]',
  'a[href*="legal/privacy"]',
  'footer a',
  '[class*="footer"] a',
];

// OPTIMIZATION: the previous version ran a separate document.querySelectorAll()
// per selector (up to 5 full-DOM traversals), and 'footer a' /
// '[class*="footer"] a' overlap heavily on most sites, so the same anchors
// were frequently walked twice. A single combined querySelectorAll() does
// one DOM traversal instead of up to five, and a Set dedupes anchors that
// match more than one selector. Priority order (specific-href selectors
// before generic footer selectors) is preserved by sorting matches back
// into the original PRIVACY_LINK_SELECTORS order before scanning.
const COMBINED_PRIVACY_LINK_SELECTOR = PRIVACY_LINK_SELECTORS.join(', ');

function findFallbackPolicyUrl(): string | null {
  const allMatches = document.querySelectorAll<HTMLAnchorElement>(COMBINED_PRIVACY_LINK_SELECTOR);
  if (allMatches.length === 0) return null;

  // Bucket each matched anchor under the highest-priority selector it
  // satisfies, so we still prefer a[href*="privacy"] over a bare footer link.
  const seen = new Set<HTMLAnchorElement>();
  const buckets: HTMLAnchorElement[][] = PRIVACY_LINK_SELECTORS.map(() => []);

  allMatches.forEach(anchor => {
    if (seen.has(anchor)) return;
    seen.add(anchor);
    for (let i = 0; i < PRIVACY_LINK_SELECTORS.length; i++) {
      if (anchor.matches(PRIVACY_LINK_SELECTORS[i])) {
        buckets[i].push(anchor);
        break; // only its highest-priority bucket
      }
    }
  });

  for (const bucket of buckets) {
    for (const anchor of bucket) {
      const href = anchor.getAttribute('href');
      const text = anchor.textContent?.toLowerCase() || '';

      if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
        const absoluteUrl = resolveUrl(href);
        if (absoluteUrl) {
          if (PRIVACY_LINK_PATTERNS.some(pattern => pattern.test(absoluteUrl)) ||
              text.includes('privacy') || text.includes('data protection')) {
            return absoluteUrl;
          }
        }
      }
    }
  }
  return null;
}

// ═══════════════════════════════════════════════════════════════
// EXTRACTION & PARSING
// ═══════════════════════════════════════════════════════════════
async function extractPolicyText(url: string): Promise<string | null> {
  try {
    const proxyResponse = await chrome.runtime.sendMessage({
      type: 'PROXY_FETCH',
      url: url
    });

    if (!proxyResponse || !proxyResponse.success) {
      console.warn(`[Ssense] Proxy fetch failed: ${proxyResponse?.error}`);
      return null;
    }

    const html = proxyResponse.html;

    if (html.trim().startsWith('%PDF')) {
      console.warn('[Ssense] Policy is a PDF. Extraction aborted to save LLM compute.');
      return null;
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    // OPTIMIZATION: combined into one querySelectorAll (single DOM pass)
    // instead of one pass per selector — order doesn't matter for removal,
    // so this is a pure win on the noisiest pages (many cookie/banner nodes).
    const NOISE_SELECTOR = [
      'script', 'style', 'nav', 'header', 'aside',
      '[class*="cookie"]', '[class*="banner"]',
      '[id*="cookie"]', '[id*="banner"]', 'footer', 'form'
    ].join(', ');

    doc.querySelectorAll(NOISE_SELECTOR).forEach(el => el.remove());

    const contentSelectors = [
      'main', 'article', '[role="main"]',
      '[class*="policy-content"]', '[class*="privacy-content"]',
      '[id*="privacy"]', '[id*="policy"]', '.content', '#content',
    ];

    let contentElement: Element | null = null;
    for (const selector of contentSelectors) {
      contentElement = doc.querySelector(selector);
      if (contentElement) break;
    }

    // CRITICAL: Use textContent (works on detached DOM nodes, no layout reflow)
    const rawText = (contentElement || doc.body).textContent || '';

    const cleanText = rawText
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .join('\n');

    if (cleanText.length > 500) {
      return cleanText;
    }

    console.warn('[Ssense] Extracted text too short, might not be a policy.');
    return null;
  } catch (err) {
    console.error('[Ssense] Error extracting policy:', err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// MAIN EXECUTION
// ═══════════════════════════════════════════════════════════════
(async () => {
  console.log('[Ssense] Searching for privacy policy...');
  
  let policyUrl = findAuthoritativePolicyUrl();
  
  if (!policyUrl) {
    policyUrl = findFallbackPolicyUrl();
  }
  
  if (!policyUrl) {
    console.log('[Ssense] No privacy policy link found on this page.');
    return;
  }

  console.log(`[Ssense] Found privacy policy: ${policyUrl}`);

  if (!isSafePublicUrl(policyUrl)) {
    console.warn('[Ssense] Policy URL resolves to a private/internal host. Blocked (SSRF guard).');
    return;
  }

  const policyText = await extractPolicyText(policyUrl);

  if (!policyText) {
    console.warn('[Ssense] Failed to extract policy text.');
    return;
  }

  // Truncate to the SLM server's supported policy-text budget. This keeps
  // the payload small (network + JSON overhead) and safely inside the
  // model's context window on the single SLM server (no local daemon mode).
  const MAX_CHARS = 16000;
  let safeText = policyText;
  if (safeText.length > MAX_CHARS) {
    safeText = safeText.substring(0, MAX_CHARS);
    console.log(`[Ssense] Truncated policy text from ${policyText.length} to ${MAX_CHARS} characters.`);
  } else {
    console.log(`[Ssense] Extracted ${safeText.length} characters of policy text.`);
  }

  try {
    // Store a local privacy snapshot before inference. The snapshot contains
    // the exact source URL, extracted text length and hash, and is never
    // uploaded by the content script itself. The service worker records the
    // eventual Cloud/Offline transport after the audit completes.
    await chrome.runtime.sendMessage({
      type: 'PRIVACY_SNAPSHOT',
      domain: window.location.hostname,
      pageUrl: window.location.href,
      policyUrl,
      policyText: safeText,
    });

    await chrome.runtime.sendMessage({
      type: 'AUDIT_POLICY',
      domain: window.location.hostname,
      policyText: safeText,
    });
    console.log('[Ssense] Privacy snapshot stored and policy queued for audit.');
  } catch (err) {
    console.error('[Ssense] Failed to send policy to Service Worker:', err);
  }
})();
} // end __ssenseExtractorLoaded guard