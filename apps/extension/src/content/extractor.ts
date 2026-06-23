// apps/extension/src/content/extractor.ts

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

function findFallbackPolicyUrl(): string | null {
  for (const selector of PRIVACY_LINK_SELECTORS) {
    const links = document.querySelectorAll(selector);
    for (const link of links) {
      const anchor = link as HTMLAnchorElement;
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

    const noiseSelectors = [
      'script', 'style', 'nav', 'header', 'aside', 
      '[class*="cookie"]', '[class*="banner"]',
      '[id*="cookie"]', '[id*="banner"]', 'footer', 'form'
    ];
    
    noiseSelectors.forEach(selector => {
      doc.querySelectorAll(selector).forEach(el => el.remove());
    });

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

  const policyText = await extractPolicyText(policyUrl);

  if (!policyText) {
    console.warn('[Ssense] Failed to extract policy text.');
    return;
  }

  // 🚀 SOTA FIX: IPC Bandwidth Alignment
  // We truncate to exactly 16,000 characters to perfectly match the Rust daemon's 
  // MAX_POLICY_CHARS constant. This reduces the IPC payload from 900KB to ~64KB,
  // eliminating serialization overhead while safely fitting inside the 8192 token context window.
  const MAX_CHARS = 16000;
  let safeText = policyText;
  if (safeText.length > MAX_CHARS) {
    safeText = safeText.substring(0, MAX_CHARS);
    console.log(`[Ssense] Truncated policy text from ${policyText.length} to ${MAX_CHARS} characters.`);
  } else {
    console.log(`[Ssense] Extracted ${safeText.length} characters of policy text.`);
  }

  try {
    await chrome.runtime.sendMessage({
      type: 'AUDIT_POLICY',
      domain: window.location.hostname,
      policyText: safeText,
    });
    console.log('[Ssense] Policy sent to Service Worker for audit.');
  } catch (err) {
    console.error('[Ssense] Failed to send policy to Service Worker:', err);
  }
})();