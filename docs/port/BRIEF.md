# WebGPU port — shared brief for every agent

Read this first, then `docs/Ocean-WebGPU-Port-Rules.md` (THE SPEC — every rule there is binding).

## Repo layout
- `src/`, `index.html`, `public/` — the ORIGINAL WebGL ocean (three 0.180). It is the visual + behavioural
  reference. Do not change its look. Run it with `npm run dev -- --port 5173` from the repo root.
- `world-ocean/` — the NEW module (`@dreamp/world-ocean`), laid out exactly as spec §2 so the folder can be
  copied to `packages/world/ocean/`. Has its own `package.json` + `node_modules` with `three@0.186.1`.
  - core (`index.ts`, `types.ts`, `spectrum.ts`, `cpu-fft.ts`, `presets/`) — NEVER imports three; runs in Node.
  - `three/` — WebGPU/TSL adapter (`index.ts` createOceanGpu, `fft.ts`, `material.ts`, `passes.ts`, `lod.ts`, ...)
  - `demo/` — standalone page (`index.html`, `demo.ts`), photo-sky stand-in for SkyLight/env, free camera,
    preset buttons, URL flags `?parity`, `?shot`, `?tier=`, `?preset=`, `?cam=`, `?origin=`, `?debug=`.
  - `stubs/world-core.ts` — the contracts from spec §3 + tiny flat-shape / floating-origin / wind stubs.
  - `tests/*.test.ts` — `node --experimental-strip-types --test tests/` (Node here is 22.14).
  - `tools/` — capture/perf harness (headless Edge with WebGPU via playwright-core).
- `docs/port/` — port working docs: `REFERENCE.md` (analysis of the WebGL ocean), `reference-shots/`
  (screenshots of the WebGL ocean at canonical cameras), `shots/` (WebGPU captures), `reviews/` (reviewer reports),
  `STATUS.json` (progress, read by the orchestrator).

## Hard constraints (summary — the spec wins on conflict)
- WebGPU + TSL only. No GLSL, ShaderMaterial, RawShaderMaterial, WebGLRenderer, WebGL fallback. FFT = compute.
- three 0.186.x APIs only. TypeScript, erasable syntax only, `.ts` extensions in relative imports, `import type`.
- Never assume +Y up in the core; floating origin: camera-relative rendering, `origin.wrapped(period)` uniforms.
- Explicit time, deterministic, no per-frame allocation, CPU heights match GPU heights (< 2 % of Hs).
- Linear HDR output, × preExposure, no tonemap in the water shader (demo does tonemap after).
- Every quality tier (low/medium/high/ultra) is a real FFT ocean. Never trade visual quality for convenience.
- No code from Minecraft shader packs.

## Practicalities
- Windows 10, PowerShell 5.1 (no `&&`). GPU: NVIDIA RTX 2070. Browser for captures: Edge
  (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`) driven by playwright-core with WebGPU flags.
- NOBODY commits, stages, resets or otherwise writes to git (user's rule). Read-only git commands only if needed.
- Each agent uses the dev-server port it was assigned, and stops servers it started before finishing.
- Never edit files owned by another concurrently running piece unless your brief says so.
- Be honest in reports: what works, what does not, measured numbers, screenshots paths.
