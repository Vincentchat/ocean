// No per-frame allocation (spec rule 7): measured exactly with a pinned semi-space in a child process.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

test('zero allocation: tick, tick + 100 heightsAt + 10 velocityAt, velocityAt, heightsAt; heap flat', () => {
  const probe = fileURLToPath(new URL('./helpers/heap-probe.ts', import.meta.url));
  const r = spawnSync(process.execPath, [
    '--expose-gc', '--min-semi-space-size=64', '--max-semi-space-size=64', '--experimental-strip-types', '--no-warnings', probe,
  ], { encoding: 'utf8', timeout: 300000 });
  assert.equal(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout.trim().split('\n').pop() as string);
  console.log(`  heap growth over 1e3 ticks + 1.1e5 queries (incl. a regeneration): ${(out.growth / 1024).toFixed(1)} KB`);
  console.log(`  allocated (stub world step ${out.perStubStep.toFixed(1)} B subtracted): tick ${out.perTick.toFixed(2)} B, tick + 100 heightsAt + 10 velocityAt ${out.perTickLoad.toFixed(2)} B, ` +
    `velocityAt ${out.perVelocity.toFixed(2)} B/call, heightsAt ${out.perBatchPoint.toFixed(3)} B/point, heightAt ${out.perHeight.toFixed(2)} B/call`);
  assert.ok(out.acc);
  assert.ok(out.version >= 3, 'the measured run regenerated the spectrum');
  assert.ok(out.growth < 64 * 1024, `heap grew by ${out.growth} bytes`);
  // A single 16-byte boxed double per tick would already fail these.
  assert.ok(out.perTick < 8, `tick allocates ${out.perTick} B`);
  assert.ok(out.perTickLoad < 8, `tick + queries allocate ${out.perTickLoad} B`);
  assert.ok(out.perVelocity < 1, `velocityAt allocates ${out.perVelocity} B`);
  assert.ok(out.perBatchPoint < 0.2, `heightsAt allocates ${out.perBatchPoint} B/point`); // < 20 B per 100-point call (harness)
  // heightAt returns a JS number: V8 boxes the returned double (one 16-byte HeapNumber) when the
  // call is not inlined into the caller. Documented contract cost; heightsAt is the zero-alloc path.
  assert.ok(out.perHeight <= 16.5, `heightAt allocates ${out.perHeight} B (more than the boxed return)`);
});
