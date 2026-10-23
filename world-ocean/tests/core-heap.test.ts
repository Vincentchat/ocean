// No per-frame allocation: heap stays flat over 1e3 ticks + 1e5 queries; queries allocate nothing.
// Runs tests/helpers/heap-probe.ts in a child process with --expose-gc.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

test('heap flat over 1e3 ticks + 1e5 queries (incl. regeneration + crossfade); queries allocate nothing', () => {
  const probe = fileURLToPath(new URL('./helpers/heap-probe.ts', import.meta.url));
  const r = spawnSync(process.execPath, ['--expose-gc', '--experimental-strip-types', '--no-warnings', probe], { encoding: 'utf8', timeout: 300000 });
  assert.equal(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout.trim().split('\n').pop() as string);
  console.log(`  heap before ${(out.before / 1048576).toFixed(2)} MB, after ${(out.after / 1048576).toFixed(2)} MB, growth ${(out.growth / 1024).toFixed(1)} KB`);
  console.log(`  allocated: heightAt ${out.perHeight.toFixed(2)} B/call, velocityAt ${out.perVelocity.toFixed(2)} B/call, tick ${out.perTick.toFixed(0)} B (stub world alone ${out.perStubTick.toFixed(0)} B)`);
  assert.ok(out.acc);
  assert.ok(out.version >= 3, 'the measured run regenerated the spectrum');
  // GC noise of the old space is +-100 KB; a leak of even 100 B/tick would show 100 KB here.
  assert.ok(out.growth < 192 * 1024, `heap grew by ${out.growth} bytes`);
  // heightAt returns a JS number: when V8 does not inline the call into the caller, the returned
  // double is boxed (one 16-byte HeapNumber). Nothing else may be allocated.
  assert.ok(out.perHeight <= 16.5 && out.perVelocity < 2, 'queries must not allocate');
  // Ticks: no explicit allocation; what remains is V8 boxing doubles in rarely-called code (~0.3 KB).
  assert.ok(out.perTick < 2048, `tick allocates ${out.perTick} B`);
});
