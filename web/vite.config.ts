import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'esnext',
    sourcemap: false,
  },
  server: { port: 5173 },
});
