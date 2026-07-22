// apps/extension/src/content/api-spoof.ts

(function () {
  // 1. Singleton Native Code Masking Registry
  const nativeToString = Function.prototype.toString;
  const spoofedFunctions = new WeakSet<object>();

  Function.prototype.toString = function () {
    if (spoofedFunctions.has(this)) {
      return `function ${this.name || ''}() { [native code] }`;
    }
    return nativeToString.call(this);
  };
  
  // Mask our own toString override
  spoofedFunctions.add(Function.prototype.toString);

  const stealthProxy = <T extends object, K extends keyof T>(
    targetObj: T,
    targetMethod: K,
    proxyHandler: ProxyHandler<T[K]>
  ) => {
    try {
      const original = targetObj[targetMethod];
      if (typeof original !== 'function') return;

      const proxy = new Proxy(original as any, proxyHandler);
      spoofedFunctions.add(proxy as any);
      targetObj[targetMethod] = proxy as any;
    } catch (e) { /* Ignore strict mode/CSP blocks */ }
  };

  const applySpoofs = (targetWindow: any) => {
    try {
      // 2. Enforce Global Privacy Control (GPC)
      if (targetWindow.Navigator && !('globalPrivacyControl' in targetWindow.Navigator.prototype)) {
        const gpcGetter = function globalPrivacyControl() { return true; };
        spoofedFunctions.add(gpcGetter);
        Object.defineProperty(targetWindow.Navigator.prototype, 'globalPrivacyControl', {
          get: gpcGetter, configurable: false, enumerable: true,
        });
      }

      // 🚀 SOTA FIX: Spoof CPU Core Count (Defeats Hardware Fingerprinting)
      if (targetWindow.Navigator && 'hardwareConcurrency' in targetWindow.Navigator.prototype) {
        const hwGetter = function hardwareConcurrency() { return 8; }; // Standardize to 8 cores
        spoofedFunctions.add(hwGetter);
        Object.defineProperty(targetWindow.Navigator.prototype, 'hardwareConcurrency', {
          get: hwGetter, configurable: false, enumerable: true,
        });
      }

      // 🚀 SOTA FIX: Spoof AudioContext Latency (Defeats Audio Fingerprinting)
      if (targetWindow.AudioContext && targetWindow.AudioContext.prototype) {
        const latencyGetter = function baseLatency() { return 0.005; };
        spoofedFunctions.add(latencyGetter);
        Object.defineProperty(targetWindow.AudioContext.prototype, 'baseLatency', {
          get: latencyGetter, configurable: true, enumerable: true,
        });
      }

      // 🚀 SOTA: Spoof WebGPU Adapter Info (Defeats modern WebGPU tracking primitives)
      if (targetWindow.navigator && 'gpu' in targetWindow.navigator && targetWindow.navigator.gpu) {
        stealthProxy(targetWindow.navigator.gpu, 'requestAdapter', {
          async apply(target, thisArg, args) {
            const adapter = await Reflect.apply(target, thisArg, args);
            if (adapter && 'requestAdapterInfo' in adapter) {
              stealthProxy(adapter, 'requestAdapterInfo', {
                async apply() {
                  return { vendor: 'Intel Inc.', architecture: 'Gen12', device: 'Intel Iris OpenGL Engine', description: 'Standard GPU' };
                }
              });
            }
            return adapter;
          }
        });
      }
    } catch (e) {}

    // 3. Spoof WebGL Renderer
    if (targetWindow.WebGLRenderingContext) {
      stealthProxy(targetWindow.WebGLRenderingContext.prototype, 'getParameter', {
        apply(target, thisArg, args) {
          const parameter = args[0];
          if (parameter === 37445) return 'Intel Inc.'; 
          if (parameter === 37446) return 'Intel Iris OpenGL Engine'; 
          return Reflect.apply(target, thisArg, args);
        },
      });
    }
    
    // 4. Spoof Canvas Fingerprinting
    if (targetWindow.HTMLCanvasElement) {
      stealthProxy(targetWindow.HTMLCanvasElement.prototype, 'toDataURL', {
        apply(target, thisArg, args) {
          const type = args[0];
          if (type === 'image/png' || type === 'image/jpeg') {
            try {
              const ctx = thisArg.getContext('2d');
              if (ctx) {
                const imageData = ctx.getImageData(0, 0, thisArg.width, thisArg.height);
                imageData.data[0] = imageData.data[0] ^ 1; // Flip LSB of Red channel
                ctx.putImageData(imageData, 0, 0);
              }
            } catch (e) { /* Tainted canvas */ }
          }
          return Reflect.apply(target, thisArg, args);
        },
      });
    }

    if (targetWindow.CanvasRenderingContext2D && targetWindow.CanvasRenderingContext2D.prototype) {
      stealthProxy(targetWindow.CanvasRenderingContext2D.prototype, 'getImageData', {
        apply(target, thisArg, args) {
          const imageData = Reflect.apply(target, thisArg, args);
          if (imageData && imageData.data && imageData.data.length > 0) {
            imageData.data[0] = imageData.data[0] ^ 1; // Flip LSB of Red channel
          }
          return imageData;
        }
      });
    }
  };

  applySpoofs(window);

  // 5. Defeat the "Clean Room" iframe bypass (Node Hook)
  if (typeof Node !== 'undefined') {
    stealthProxy(Node.prototype, 'appendChild', {
      apply(target, thisArg, args) {
        const node = args[0] as Node;
        const result = Reflect.apply(target, thisArg, args);
        if (node && node.nodeName === 'IFRAME') {
          const iframe = node as HTMLIFrameElement;
          if (iframe.contentWindow) {
            try { applySpoofs(iframe.contentWindow); } catch (e) {}
          }
        }
        return result;
      }
    });
  }

  // 6. Defeat the Getter Bypass
  try {
    if (typeof HTMLIFrameElement !== 'undefined') {
      const originalContentWindow = Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype, 'contentWindow');
      if (originalContentWindow && originalContentWindow.get) {
        const contentWindowGetter = function contentWindow() {
          const cw = originalContentWindow.get!.call(this);
          if (cw) { try { applySpoofs(cw); } catch (e) {} }
          return cw;
        };
        spoofedFunctions.add(contentWindowGetter);
        Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
          get: contentWindowGetter, configurable: true, enumerable: true,
        });
      }
    }
  } catch (e) {}

  // 7. Network Telemetry & Tracker Interception Hooks (fetch / XHR)
  try {
    if (typeof window !== 'undefined' && window.fetch) {
      stealthProxy(window, 'fetch', {
        apply(target, thisArg, args) {
          const urlStr = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) ? args[0].url : '';
          const badDomains: string[] = (window as any).__ssenseBlockedDomains || [];
          if (urlStr && badDomains.some(d => urlStr.toLowerCase().includes(d))) {
            console.warn(`[Ssense Shield] Aborted blocked fetch request: ${urlStr}`);
            return Promise.reject(new TypeError('Failed to fetch (Ssense DPDP Shield Blocked)'));
          }
          if ((window as any).__ssenseStripTelemetry && args[1] && args[1].headers) {
            const cleanHeaders = new Headers(args[1].headers);
            ['x-telemetry', 'x-tracker', 'x-analytics', 'x-mixpanel', 'x-client-data'].forEach(h => cleanHeaders.delete(h));
            args[1].headers = cleanHeaders;
          }
          return Reflect.apply(target, thisArg, args);
        }
      });
    }

    if (typeof XMLHttpRequest !== 'undefined' && XMLHttpRequest.prototype.open) {
      stealthProxy(XMLHttpRequest.prototype, 'open', {
        apply(target, thisArg, args) {
          const urlStr = typeof args[1] === 'string' ? args[1] : '';
          const badDomains: string[] = (window as any).__ssenseBlockedDomains || [];
          if (urlStr && badDomains.some(d => urlStr.toLowerCase().includes(d))) {
            console.warn(`[Ssense Shield] Aborted blocked XHR request: ${urlStr}`);
            (thisArg as any).__ssenseBlocked = true;
          }
          return Reflect.apply(target, thisArg, args);
        }
      });

      stealthProxy(XMLHttpRequest.prototype, 'send', {
        apply(target, thisArg, args) {
          if ((thisArg as any).__ssenseBlocked) {
            thisArg.dispatchEvent(new Event('error'));
            return;
          }
          return Reflect.apply(target, thisArg, args);
        }
      });
    }
  } catch (e) {}
})();