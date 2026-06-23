// apps/extension/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: '', 
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false, 
    // 🚀 SOTA FIX: Prevent Vite from injecting module preloads into content scripts
    modulePreload: { polyfill: false }, 
    rollupOptions: {
      input: {
        sidepanel: resolve(__dirname, 'sidepanel.html'),
        'background/service-worker': resolve(__dirname, 'src/background/service-worker.ts'),
        'content/extractor': resolve(__dirname, 'src/content/extractor.ts'),
        'content/dark-pattern-blocker': resolve(__dirname, 'src/content/dark-pattern-blocker.ts'),
        // 🚀 SOTA FIX: Restore the API Spoofer to the build pipeline
        'content/api-spoof': resolve(__dirname, 'src/content/api-spoof.ts'), 
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]'
      },
    },
  },
});