// Query throughput (spec Â§6: aim >= 1 M heightAt calls/s in Node). The minimum can be lowered on a
// loaded machine with OCEAN_PERF_MIN (calls/s).
import test from 'node:test';
import assert from 'node:assert/strict';
import { measure } from './bench-queries.ts';

test('heightAt >= 1 M calls/s (best of 5), tick cost reported', () => {
  const min = Number(process.env.OCEAN_PERF_MIN ?? 1e6);
  const r = measure('windy', 5, 200000);
  console.log(`  heightAt ${(r.heightAtPerSec / 1e6).toFixed(2)} M/s (local), ${(r.heightAtRandomPerSec / 1e6).toFixed(2)} M/s (random 3 km), velocityAt ${(r.velocityAtPerSec / 1e6).toFixed(2)} M/s`);
  console.log(`  tick: steady ${r.tickMsSteady.toFixed(2)} ms, during regeneration/crossfade ${r.tickMsRegen.toFixed(2)} ms, max ${r.tickMsMax.toFixed(1)} ms`);
  assert.ok(r.heightAtPerSec >= min, `heightAt ${r.heightAtPerSec.toFixed(0)} calls/s < ${min}`);
});
