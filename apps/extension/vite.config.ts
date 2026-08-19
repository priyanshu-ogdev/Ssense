import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: '',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    // 🚀 SOTA FIX: Prevent Vite from injecting module preloads into content scripts
    modulePreload: { polyfill: false },
    // 🚀 SOTA FIX: Forces all CSS into a single style.css file for easy manifest.json injection
    cssCodeSplit: false,
    rollupOptions: {
      input: {
        sidepanel: resolve(__dirname, 'sidepanel.html'),
        options: resolve(__dirname, 'options.html'),
        popup: resolve(__dirname, 'popup.html'),
        'background/service-worker': resolve(__dirname, 'src/background/service-worker.ts'),
        'content/extractor': resolve(__dirname, 'src/content/extractor.ts'),
        'content/dark-pattern-blocker': resolve(__dirname, 'src/content/dark-pattern-blocker.ts'),
        'content/chat-widget': resolve(__dirname, 'src/content/chat-widget.ts'),
        'content/api-spoof': resolve(__dirname, 'src/content/api-spoof.ts'),
      },
      output: {
        // Manifest V3 requires exact filenames for background/content scripts. No hashing allowed here.
        entryFileNames: '[name].js',
        // React chunks and assets can be safely hashed since HTML loads them dynamically
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]'
      },
    },
  },
});