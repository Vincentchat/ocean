// Static checks on the render-agnostic core (spec Â§1 rules 3, 4; Â§7 checks 2, 8).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const CORE = ['index.ts', 'types.ts', 'spectrum.ts', 'cpu-fft.ts', ...readdirSync(root + 'presets').map((f) => 'presets/' + f)];

function code(file: string): string[] {
  // strip // comments and /* */ comments (good enough for these files)
  const src = readFileSync(root + file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  return src.split(/\r?\n/).map((l) => l.replace(/\/\/.*$/, ''));
}

test('core never imports three (or anything outside the core + stubs)', () => {
  for (const f of CORE) {
    for (const line of code(f)) {
      const m = /\bfrom\s+['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]/.exec(line);
      if (!m) continue;
      const spec = m[1] ?? m[2];
      assert.ok(!/three/.test(spec), `${f} imports ${spec}`);
      assert.ok(spec.startsWith('./') || spec.startsWith('../'), `${f} imports package ${spec}`);
    }
  }
});

test('no ".y" used as height in the core: .y only appears in full x/y/z vector expressions', () => {
  // Allowed: a line that also touches .x and .z (dot products, vector sets), or a component-wise
  // line of a vector expression where every member access is .y (e.g. out.y = vE * east.y + ...).
  for (const f of CORE) {
    code(f).forEach((line, i) => {
      if (!/\.y\b/.test(line)) return;
      const full = /\.x\b/.test(line) && /\.z\b/.test(line);
      const accesses = line.match(/\.[a-zA-Z_]\w*\b(?!\s*\()/g) ?? [];
      const componentWise = accesses.length >= 3 && accesses.every((a) => a === '.y');
      assert.ok(full || componentWise, `${f}:${i + 1} uses .y outside a vector expression: ${line.trim()}`);
    });
  }
});
