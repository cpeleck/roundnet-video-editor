import { defineConfig } from 'vite';
export default defineConfig({
  server: { port: 5173, strictPort: true,
    fs: { deny: ['.env', '.env.*', '*.{crt,pem}', '**/.git/**', '**/.roundnet/**', '**/.venv/**'] },
    proxy: { '/api': process.env.VITE_API_URL ?? 'http://127.0.0.1:8000' } },
  build: { outDir: 'dist' },
});
