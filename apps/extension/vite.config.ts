// apps/extension/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

// 🚀 SOTA FIX: Polyfill __dirname for ESM environments.
// If package.json has "type": "module", native __dirname is undefined and crashes the build.
const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Forces relative paths for Chrome Extension compatibility
  base: '', 
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Disable inline source maps for production security and size
    sourcemap: false, 
    rollupOptions: {
      input: {
        // Point to the root directory HTML so Vite processes the <script> tags
        sidepanel: resolve(__dirname, 'sidepanel.html'),
        'background/service-worker': resolve(__dirname, 'src/background/service-worker.ts'),
        'content/extractor': resolve(__dirname, 'src/content/extractor.ts'),
        'content/dark-pattern-blocker': resolve(__dirname, 'src/content/dark-pattern-blocker.ts'),
      },
      output: {
        // Force clean filenames so manifest.json can easily locate them
        entryFileNames: '[name].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]'
      },
    },
  },
});