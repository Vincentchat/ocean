// Worker for the CPU water query (core, no three import). One worker per cascade.
//
// It attaches a CascadeSim to the main thread's SharedArrayBuffers (slot h0, snapshot pool, frames,
// source table, control block) and serves, in priority order:
//  1. position-frame prefill for the next tick (the main thread adopts it instead of filling),
//  2. snapshot requests (evolve + 3 complex 2-D IFFTs + pack),
//  3. spectrum regeneration for this cascade, in 32-row chunks between the above (master h0 for the
//     GPU + this cascade's slot grid).
// Same code, same numbers: everything is bit-identical to the main-thread path.
//
// Started by index.ts (createOcean option `worker`, default true) with OceanWorkerData as
// workerData (Node worker_threads) or as the first message (browser Worker, module type).
import { CascadeSim, C_RSTATE, C_RSLOT, F_REGEN, S_DONE, S_REQUESTED } from './cpu-fft.ts';
import type { QueryCascadeSettings, SharedSimBuffers } from './cpu-fft.ts';
import {
  CASCADES, REFERENCE_SIZE, omegaIndices, gaussianField, spectrumScalars, generateRawRows, packInitialSpectrum,
} from './spectrum.ts';
import type { SpectrumParams, SpectrumScalars } from './spectrum.ts';

export interface OceanWorkerData {
  cascade: number;
  cfg: QueryCascadeSettings;
  g: number;
  seed: number;
  numSlots: number;
  poolSize: number;
  shared: SharedSimBuffers;
  /** Master-size (256) h0 of this cascade for every slot (what the GPU uploads). */
  masterH0: SharedArrayBuffer[];
}

const REGEN_ROWS = 32;

function start(data: OceanWorkerData): void {
  const def = CASCADES[data.cascade];
  const nm = REFERENCE_SIZE;
  const om = omegaIndices(def, data.g, new Float32Array(nm * nm));
  const sim = new CascadeSim(def, data.cfg, om, data.numSlots, data.poolSize, data.shared);
  const master = data.masterH0.map((b) => new Float32Array(b));
  const xi = gaussianField(data.seed, data.cascade, new Float64Array(2 * nm * nm));
  const raw = new Float64Array(2 * nm * nm);
  const params: SpectrumParams = { windSpeed: 0, windDirection: 0, fetch: 0, gamma: 0, spread: 0, standing: 0, choppiness: 0, amplitude: 0 };
  const sc: SpectrumScalars = { g: 0, U: 0, dirGrid: 0, wp: 0 };
  const { ctrl, ctrlF } = sim.control;
  let row = -1, slot = 0;
  const regenStep = (): boolean => {
    if (row < 0) {
      if (Atomics.load(ctrl, C_RSTATE) !== S_REQUESTED) return false;
      slot = ctrl[C_RSLOT];
      params.windSpeed = ctrlF[F_REGEN]; params.windDirection = ctrlF[F_REGEN + 1]; params.fetch = ctrlF[F_REGEN + 2];
      params.gamma = ctrlF[F_REGEN + 3]; params.spread = ctrlF[F_REGEN + 4]; params.standing = ctrlF[F_REGEN + 5];
      params.choppiness = ctrlF[F_REGEN + 6]; params.amplitude = ctrlF[F_REGEN + 7];
      spectrumScalars(params, data.g, sc);
      row = 0;
    }
    const r1 = Math.min(nm, row + REGEN_ROWS);
    generateRawRows(params, sc, def, xi, raw, row, r1);
    row = r1;
    if (row >= nm) {
      packInitialSpectrum(raw, params.standing, master[slot]);
      sim.loadSlot(slot, master[slot]);
      row = -1;
      Atomics.store(ctrl, C_RSTATE, S_DONE);
    }
    return true;
  };
  sim.serve(regenStep);
}

const proc = (globalThis as { process?: { versions?: { node?: string }; getBuiltinModule?: (id: string) => unknown } }).process;
if (proc?.versions?.node && proc.getBuiltinModule) {
  const wt = proc.getBuiltinModule('node:worker_threads') as { isMainThread: boolean; workerData: OceanWorkerData };
  if (!wt.isMainThread && wt.workerData) start(wt.workerData);
} else {
  const scope = globalThis as unknown as { onmessage: ((ev: { data: OceanWorkerData }) => void) | null };
  scope.onmessage = (ev) => { scope.onmessage = null; start(ev.data); };
}
