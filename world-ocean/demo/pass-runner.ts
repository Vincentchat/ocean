// DEMO ONLY: stand-in for the engine render graph. Runs pass objects in spec §5 stage order.
// The demo main loop is just runner.execute(ctx); every renderer.render / renderer.compute call lives in a pass.
import { STAGES, type Pass, type PassContext } from '../three/pass-types.ts';

export interface PassRunner {
  add(pass: Pass): void;
  remove(pass: Pass): void;
  /** Runs every enabled pass, stage order, insertion order within a stage. No allocation. */
  execute(ctx: PassContext): void;
  readonly passes: readonly Pass[];
  /** Per-pass CPU ms of the last execute (debug readout). */
  readonly cpuMs: Float64Array;
}

export function createPassRunner(perf?: { begin(m: string, p: string): void; end(): void }): PassRunner {
  const passes: Pass[] = [];
  let cpuMs = new Float64Array(0);
  function sort() {
    // Stable sort by stage index.
    const order = passes.map((p, i) => ({ p, i }));
    order.sort((a, b) => (STAGES.indexOf(a.p.desc.stage) - STAGES.indexOf(b.p.desc.stage)) || a.i - b.i);
    for (let i = 0; i < order.length; i++) passes[i] = order[i].p;
    cpuMs = new Float64Array(passes.length);
  }
  const runner: PassRunner = {
    get passes() { return passes; },
    get cpuMs() { return cpuMs; },
    add(pass) {
      if (STAGES.indexOf(pass.desc.stage) < 0) throw new Error(`Pass ${pass.desc.id}: unknown stage ${pass.desc.stage}`);
      if (passes.some((p) => p.desc.id === pass.desc.id)) throw new Error(`Duplicate pass id ${pass.desc.id}`);
      passes.push(pass);
      sort();
    },
    remove(pass) {
      const i = passes.indexOf(pass);
      if (i >= 0) { passes.splice(i, 1); sort(); }
    },
    execute(ctx) {
      for (let i = 0; i < passes.length; i++) {
        const p = passes[i];
        if (p.enabled === false) continue;
        const t0 = performance.now();
        perf?.begin(p.desc.module, p.desc.id);
        p.execute(ctx);
        perf?.end();
        cpuMs[i] = performance.now() - t0;
      }
    },
  };
  return runner;
}
