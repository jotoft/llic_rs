import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
  },
  server: {
    fs: {
      // Allow serving files from pkg directory (symlinked from parent)
      allow: ['..'],
    },
  },
  assetsInclude: ['**/*.wasm'],
});
