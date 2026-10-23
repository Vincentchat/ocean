#!/usr/bin/env node
// Start a Vite dev server for the WebGL reference (repo root) or the WebGPU demo (world-ocean/).
//   node tools/serve.mjs --reference [--port 5173]   -> http://127.0.0.1:5173/          (repo root index.html)
//   node tools/serve.mjs --demo      [--port 5180]   -> http://127.0.0.1:5180/demo/
//   node tools/serve.mjs --both                      -> both, on 5173 and 5180
// Equivalent npm commands: (repo root) `npm run dev -- --port 5173 --strictPort`; (world-ocean) `npm run demo`.
// Stops on Ctrl+C / SIGTERM, or after --ttl <seconds> if given.
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { parseArgs } from './lib/browser.mjs';

const opts = parseArgs(process.argv.slice(2));
const pkgDir = fileURLToPath(new URL('..', import.meta.url));
const repoDir = fileURLToPath(new URL('../..', import.meta.url));

async function start(kind, port) {
  const server = await createServer(kind === 'reference'
    ? { root: repoDir, configFile: `${repoDir}vite.config.js`, server: { host: '127.0.0.1', port, strictPort: true }, clearScreen: false, logLevel: 'warn' }
    : { root: pkgDir, configFile: `${pkgDir}vite.config.mjs`, server: { host: '127.0.0.1', port, strictPort: true }, clearScreen: false, logLevel: 'warn' });
  await server.listen();
  const url = `http://127.0.0.1:${port}/${kind === 'demo' ? 'demo/' : ''}`;
  console.log(`${kind} ready: ${url}`);
  return server;
}

const servers = [];
if (opts.reference || opts.both) servers.push(await start('reference', Number(opts.both ? 5173 : opts.port ?? 5173)));
if (opts.demo || opts.both || (!opts.reference && !opts.both)) servers.push(await start('demo', Number(opts.both ? 5180 : opts.port ?? 5180)));

const stop = async () => {
  await Promise.all(servers.map((s) => s.close()));
  process.exit(0);
};
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
if (opts.ttl) setTimeout(stop, Number(opts.ttl) * 1000);
