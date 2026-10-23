// Dev server for the standalone demo (world-ocean/demo/).
// Root is the package folder so the demo can import ../index.ts, ../three/index.ts and ../stubs/*.
// Binary assets (foam-pattern.png, sky-*.jpg, sky-radiance.bin) are served from the reference repo's
// public/ folder at the site root; code never hard-codes these paths: it asks the asset resolver
// (assets.ts) for a logical id, and the demo maps ids -> URLs (demo/demo-assets.ts).
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const here = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root: here,
  publicDir: fileURLToPath(new URL('../public', import.meta.url)),
  server: {
    host: '127.0.0.1',
    port: 5180,
    strictPort: true,
    fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] },
  },
  plugins: [
    {
      name: 'demo-index-redirect',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // GET /__exists?path=index.ts -> {"exists": bool}: lets the demo probe for files that other
          // builders have not written yet without a 404 in the console.
          if (req.url?.startsWith('/__exists')) {
            const rel = new URL(req.url, 'http://x').searchParams.get('path') ?? '';
            const abs = resolve(here, rel);
            const inside = abs.startsWith(resolve(here)) && !rel.includes('..');
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ exists: inside && existsSync(abs) }));
            return;
          }
          if (req.url === '/' || req.url?.startsWith('/?')) {
            res.statusCode = 302;
            res.setHeader('Location', '/demo/' + (req.url.length > 1 ? req.url.slice(1) : ''));
            res.end();
            return;
          }
          next();
        });
      },
    },
  ],
  build: {
    target: 'es2022',
    outDir: 'dist',
    rollupOptions: { input: { demo: fileURLToPath(new URL('./demo/index.html', import.meta.url)) } },
  },
});
