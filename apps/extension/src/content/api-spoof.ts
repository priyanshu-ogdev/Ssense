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
})();