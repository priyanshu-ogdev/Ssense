// apps/extension/src/content/dark-pattern-blocker.ts
import type { DpdpAuditReport, NetworkAction, Violation } from '../types/native-protocol';

console.log('[Ssense] DOM Enforcer injected and listening.');

declare global {
  interface Window {
    __ssenseObserverAttached?: boolean;
  }
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'ENFORCE_DPDP_RULES' && message.report) {
    executeNetworkActions(message.report);
  }
});

function executeNetworkActions(report: DpdpAuditReport) {
  if (window.__ssenseObserverAttached) return;
  window.__ssenseObserverAttached = true;

  console.log(`[Ssense] Activating Real-Time Shield. Trust Score: ${report.dpdp_trust_score}`);

  const badDomains: string[] = [];
  const quotesToHighlight: string[] = [];
  let injectGPC = false;
  let spoofHardware = false;

  // 1. Parse the Rust Daemon's Enforcement Directives
  report.violations.forEach((violation: Violation) => {
    switch (violation.network_action as NetworkAction) {
      case 'BLOCK_THIRD_PARTY':
        if (violation.offending_entities) {
          badDomains.push(...violation.offending_entities.map(e => e.toLowerCase()));
        }
        break;
      case 'WARN_USER_ONLY':
      case 'STRIP_TELEMETRY_HEADER':
        if (violation.evidence_quote) {
          // Normalize whitespace for robust matching against dynamic DOM text
          quotesToHighlight.push(violation.evidence_quote.toLowerCase().replace(/\s+/g, ' ').trim().substring(0, 50));
        }
        break;
      case 'INJECT_GPC_SIGNAL':
        injectGPC = true;
        break;
      case 'SPOOF_HARDWARE_API':
        spoofHardware = true;
        break;
    }
  });

  // 🚀 SOTA: Execute MAIN world injections (Bypasses Isolated World limitations)
  if (injectGPC) injectGlobalPrivacyControl();
  if (spoofHardware) spoofHardwareAPIs();

  // 2. Initial Sweep (catch existing elements on page load)
  if (badDomains.length > 0) sweepAndBlock(document.body, badDomains);
  if (quotesToHighlight.length > 0) highlightViolationsInNodes([document.body], quotesToHighlight);

  // 3. Real-Time MutationObserver (Optimized for SPAs & Dynamic Injections)
  const observer = new MutationObserver((mutations) => {
    const elementsToCheck: Element[] = [];
    
    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const el = node as Element;
            elementsToCheck.push(el);
            // Check descendants of the added node
            const descendants = el.querySelectorAll?.('iframe, script, img, p, span, div, li, h1, h2, h3, h4, h5, h6');
            descendants?.forEach(desc => elementsToCheck.push(desc));
          }
        });
      } 
      // 🚀 SOTA FIX: Catch dynamic SPA src/href changes on existing elements
      else if (mutation.type === 'attributes' && mutation.target) {
        elementsToCheck.push(mutation.target as Element);
      }
    }

    if (elementsToCheck.length > 0) {
      requestAnimationFrame(() => {
        if (badDomains.length > 0) blockElements(elementsToCheck, badDomains);
        if (quotesToHighlight.length > 0) highlightViolationsInNodes(elementsToCheck, quotesToHighlight);
      });
    }
  });

  observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'href'], // 🚀 SOTA: Catches dynamic tracker injections
    attributeOldValue: false
  });
  
  console.log('[Ssense] MutationObserver active. Shield is locked in.');
}

// ═══════════════════════════════════════════════════════════════
// 🚀 SOTA: MAIN WORLD INJECTIONS
// ═══════════════════════════════════════════════════════════════

function injectGlobalPrivacyControl() {
  try {
    const script = document.createElement('script');
    script.textContent = `
      try {
        Object.defineProperty(Navigator.prototype, 'globalPrivacyControl', {
          get: function() { return true; },
          configurable: false
        });
      } catch(e) { /* CSP blocked */ }
    `;
    (document.head || document.documentElement).appendChild(script);
    script.remove();
    console.log('[Ssense] Global Privacy Control (GPC) signal injected.');
  } catch (e) {
    console.warn('[Ssense] Failed to inject GPC:', e);
  }
}

function spoofHardwareAPIs() {
  try {
    const script = document.createElement('script');
    script.textContent = `
      try {
        // Spoof Canvas Fingerprinting by adding microscopic noise
        const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
        HTMLCanvasElement.prototype.toDataURL = function(type) {
          if (type === 'image/png' || type === 'image/jpeg') {
            const ctx = this.getContext('2d');
            if (ctx) {
              const imageData = ctx.getImageData(0, 0, this.width, this.height);
              for (let i = 0; i < imageData.data.length; i += 16) {
                imageData.data[i] += Math.floor(Math.random() * 2); // Red channel noise
              }
              ctx.putImageData(imageData, 0, 0);
            }
          }
          return originalToDataURL.apply(this, arguments);
        };
      } catch(e) { /* CSP blocked */ }
    `;
    (document.head || document.documentElement).appendChild(script);
    script.remove();
    console.log('[Ssense] Hardware API fingerprinting spoofed.');
  } catch (e) {
    console.warn('[Ssense] Failed to spoof hardware APIs:', e);
  }
}

// ═══════════════════════════════════════════════════════════════
// DOM MANIPULATION (Optimized for 60fps)
// ═══════════════════════════════════════════════════════════════

function blockElements(elements: Element[], badDomains: string[]) {
  try {
    for (const el of elements) {
      if (el.classList.contains('ssense-blocked-element')) continue;

      const tag = el.tagName;
      if (tag === 'IFRAME' || tag === 'SCRIPT' || tag === 'IMG') {
        const src = (el as HTMLIFrameElement | HTMLScriptElement | HTMLImageElement).src;
        if (src && badDomains.some(domain => src.toLowerCase().includes(domain))) {
          el.classList.add('ssense-blocked-element');
        }
      }
    }
  } catch (err) {
    console.error('[Ssense] Error blocking elements:', err);
  }
}

function sweepAndBlock(rootNode: HTMLElement, badDomains: string[]) {
  try {
    const elements = rootNode.querySelectorAll('iframe, script, img');
    blockElements(Array.from(elements), badDomains);
  } catch (err) {
    console.error('[Ssense] Error sweeping elements:', err);
  }
}

// 🚀 SOTA: Incremental Highlighting (Only scans provided nodes)
function highlightViolationsInNodes(nodes: Element[], quotes: string[]) {
  try {
    for (const el of nodes) {
      // Only process text-bearing elements to save CPU
      if (['P', 'SPAN', 'DIV', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'TD', 'TH'].includes(el.tagName)) {
        if (el.classList.contains('ssense-highlight-violation')) continue;
        
        const text = (el.textContent || '').toLowerCase().replace(/\s+/g, ' ');
        
        for (const quote of quotes) {
          if (text.includes(quote)) {
            el.classList.add('ssense-highlight-violation');
            el.setAttribute('title', '⚠️ Ssense: DPDP Violation Detected');
            break; // Only highlight once per element
          }
        }
      }
    }
  } catch (err) {
    console.error('[Ssense] Error highlighting violations:', err);
  }
}