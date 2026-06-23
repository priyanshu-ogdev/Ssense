// apps/extension/src/content/api-spoof.js
// 🚀 SOTA: Runs in MAIN world at document_start. 
// Preemptively spoofs APIs using stealth Proxies and prevents iframe escapes.

(function() {
  // Utility to seamlessly proxy a function without breaking its native .toString() signature
  const stealthProxy = (targetObj, targetMethod, proxyHandler) => {
    try {
      const original = targetObj[targetMethod];
      const proxy = new Proxy(original, proxyHandler);
      
      // Mask the Proxy to look like a native function if inspected
      const proxyToString = new Proxy(Function.prototype.toString, {
        apply(target, thisArg, args) {
          if (thisArg === proxy) return `function ${targetMethod}() { [native code] }`;
          return Reflect.apply(target, thisArg, args);
        }
      });
      
      Object.defineProperty(Function.prototype, 'toString', {
        value: proxyToString,
        configurable: true,
        enumerable: false,
        writable: true
      });

      targetObj[targetMethod] = proxy;
    } catch (e) { /* Ignore strict mode/CSP blocks */ }
  };

  const applySpoofs = (targetWindow) => {
    try {
      // 1. Enforce Global Privacy Control (GPC)
      if (!('globalPrivacyControl' in targetWindow.Navigator.prototype)) {
        Object.defineProperty(targetWindow.Navigator.prototype, 'globalPrivacyControl', {
          get: () => true,
          configurable: false,
          enumerable: true
        });
      }
    } catch (e) {}

    // 2. Spoof Canvas Fingerprinting (Proxy implementation)
    if (targetWindow.HTMLCanvasElement) {
      stealthProxy(targetWindow.HTMLCanvasElement.prototype, 'toDataURL', {
        apply: function(target, thisArg, args) {
          const type = args[0];
          if (type === 'image/png' || type === 'image/jpeg') {
            try {
              const ctx = thisArg.getContext('2d');
              if (ctx) {
                const imageData = ctx.getImageData(0, 0, thisArg.width, thisArg.height);
                // Flip the least significant bit of the first pixel's red channel
                imageData.data[0] = imageData.data[0] ^ 1; 
                ctx.putImageData(imageData, 0, 0);
              }
            } catch (e) { /* Tainted canvas */ }
          }
          return Reflect.apply(target, thisArg, args);
        }
      });
    }

    // 3. Spoof WebGL Renderer (Proxy implementation)
    if (targetWindow.WebGLRenderingContext) {
      stealthProxy(targetWindow.WebGLRenderingContext.prototype, 'getParameter', {
        apply: function(target, thisArg, args) {
          const parameter = args[0];
          if (parameter === 37445) return 'Intel Inc.'; // UNMASKED_VENDOR_WEBGL
          if (parameter === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
          return Reflect.apply(target, thisArg, args);
        }
      });
    }
  };

  // Apply to the main window immediately
  applySpoofs(window);

  // 4. Defeat the "Clean Room" iframe bypass
  // Intercept element attachment to catch newly minted iframes
  stealthProxy(Node.prototype, 'appendChild', {
    apply: function(target, thisArg, args) {
      const node = args[0];
      const result = Reflect.apply(target, thisArg, args);
      
      // If the appended node is an iframe, immediately poison its contentWindow
      if (node && node.tagName === 'IFRAME' && node.contentWindow) {
        try {
          applySpoofs(node.contentWindow);
        } catch (e) { /* Cross-origin frame, let browser security handle it */ }
      }
      return result;
    }
  });
})();