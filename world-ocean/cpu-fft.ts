// CPU FFT for the WATER_QUERY. Render-agnostic: never import three here.
//
// Allocation-free radix-2 complex FFT, 1-D and 2-D, in place on separate real / imaginary
// Float64Arrays. The plan (bit-reversal table + twiddles) is built once per size.
//
// SIGN / NORMALISATION CONVENTION (shared with spectrum.ts and the GPU FFT):
//   inverse  : x[n] = sum_m X[m] * exp(+2*pi*i*m*n/N)     (NO 1/N factor)
//   forward  : X[m] = sum_n x[n] * exp(-2*pi*i*m*n/N)     (NO 1/N factor; only used by tests)
// 2-D: rows (index x, contiguous) then columns (index y, stride N); array index = y * N + x.
// This is exactly the transform the reference GPU FFT (src/spectral-ocean.js FFT_FRAGMENT,
// twiddle w = (cos, +sin)) and the reference CPU capillary FFT (src/cpu-capillary-ocean.js
// inversePlan) perform.

import { LOOP_PERIOD, OMEGA_UNIT, resampleCentred } from './spectrum.ts';
import type { CascadeDef } from './spectrum.ts';

export interface FftPlan {
  readonly n: number;
  readonly log2n: number;
  readonly reverse: Uint32Array;
  /** cos(2*pi*j/n), j in [0, n/2) */
  readonly cos: Float64Array;
  /** sin(2*pi*j/n), j in [0, n/2) */
  readonly sin: Float64Array;
}

export function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
}

/** Build a plan for size n (power of two, >= 2). Setup-time only (allocates). */
export function createFftPlan(n: number): FftPlan {
  if (!isPowerOfTwo(n) || n < 2) throw new RangeError(`FFT size must be a power of two >= 2 (got ${n})`);
  const log2n = Math.round(Math.log2(n));
  const reverse = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0, v = i;
    for (let b = 0; b < log2n; b++) { r = (r << 1) | (v & 1); v >>= 1; }
    reverse[i] = r;
  }
  const half = n >> 1;
  const cos = new Float64Array(half), sin = new Float64Array(half);
  for (let j = 0; j < half; j++) {
    const a = (2 * Math.PI * j) / n;
    cos[j] = Math.cos(a);
    sin[j] = Math.sin(a);
  }
  return { n, log2n, reverse, cos, sin };
}

/**
 * In-place 1-D FFT of `n` complex values starting at `offset` with element stride `stride`.
 * sign = +1: inverse (exp(+i...)), sign = -1: forward (exp(-i...)). Unnormalised.
 */
export function fft1d(plan: FftPlan, re: Float64Array, im: Float64Array, offset: number, stride: number, sign: number): void {
  const n = plan.n, rev = plan.reverse, cs = plan.cos, sn = plan.sin;
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = offset + i * stride, b = offset + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let span = 2; span <= n; span <<= 1) {
    const half = span >> 1, step = n / span;
    for (let base = 0; base < n; base += span) {
      for (let j = 0; j < half; j++) {
        const w = j * step, c = cs[w], s = sign * sn[w];
        const a = offset + (base + j) * stride, b = a + half * stride;
        const br = re[b], bi = im[b];
        const tr = br * c - bi * s, ti = br * s + bi * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

/**
 * In-place 2-D FFT of an n x n complex grid (index y * n + x). Rows first, then columns.
 * The column pass runs the butterflies on whole rows at once (contiguous memory, cache friendly).
 * sign = +1 inverse, -1 forward; unnormalised. Allocation-free.
 */
export function fft2d(plan: FftPlan, re: Float64Array, im: Float64Array, sign: number): void {
  const n = plan.n;
  fftRows(plan, re, im, 0, n, sign);
  fftColumnsBitReverse(plan, re, im);
  for (let span = 2; span <= n; span <<= 1) fftColumnsStage(plan, re, im, span, sign);
}

// The 2-D transform split into resumable pieces (used to amortise snapshots over several ticks):
//   fftRows(0..n) ; fftColumnsBitReverse ; fftColumnsStage(span = 2, 4, ..., n)

/** 1-D FFT of rows [rowStart, rowEnd). Allocation-free. */
export function fftRows(plan: FftPlan, re: Float64Array, im: Float64Array, rowStart: number, rowEnd: number, sign: number): void {
  const n = plan.n;
  for (let y = rowStart; y < rowEnd; y++) fft1d(plan, re, im, y * n, 1, sign);
}

/** Column pass, step 1: bit-reverse whole rows. */
export function fftColumnsBitReverse(plan: FftPlan, re: Float64Array, im: Float64Array): void {
  const n = plan.n, rev = plan.reverse;
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = i * n, b = j * n;
      for (let x = 0; x < n; x++) {
        let t = re[a + x]; re[a + x] = re[b + x]; re[b + x] = t;
        t = im[a + x]; im[a + x] = im[b + x]; im[b + x] = t;
      }
    }
  }
}

/** Column pass, one butterfly stage of width `span` (2, 4, ..., n), vectorised over x (contiguous). */
export function fftColumnsStage(plan: FftPlan, re: Float64Array, im: Float64Array, span: number, sign: number): void {
  const n = plan.n, cs = plan.cos, sn = plan.sin;
  const half = span >> 1, step = n / span;
  for (let base = 0; base < n; base += span) {
    for (let j = 0; j < half; j++) {
      const w = j * step, c = cs[w], s = sign * sn[w];
      const a = (base + j) * n, b = a + half * n;
      for (let x = 0; x < n; x++) {
        const ia = a + x, ib = b + x;
        const br = re[ib], bi = im[ib];
        const tr = br * c - bi * s, ti = br * s + bi * c;
        re[ib] = re[ia] - tr; im[ib] = im[ia] - ti;
        re[ia] += tr; im[ia] += ti;
      }
    }
  }
}

/** Inverse 2-D FFT (exp(+i...), unnormalised), the transform used for the ocean fields. */
export function ifft2d(plan: FftPlan, re: Float64Array, im: Float64Array): void {
  fft2d(plan, re, im, 1);
}

/**
 * Naive 2-D DFT (O(n^4)) for tests. Allocates; never call in a frame.
 * sign = +1 inverse, -1 forward; unnormalised.
 */
export function naiveDft2d(n: number, re: Float64Array, im: Float64Array, sign: number): { re: Float64Array; im: Float64Array } {
  const outRe = new Float64Array(n * n), outIm = new Float64Array(n * n);
  for (let v = 0; v < n; v++) for (let u = 0; u < n; u++) {
    let sr = 0, si = 0;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const a = (sign * 2 * Math.PI * (u * x + v * y)) / n;
      const c = Math.cos(a), s = Math.sin(a);
      const r = re[y * n + x], i = im[y * n + x];
      sr += r * c - i * s; si += r * s + i * c;
    }
    outRe[v * n + u] = sr; outIm[v * n + u] = si;
  }
  return { re: outRe, im: outIm };
}

// =============================================================================================
// CPU cascade simulation for the WATER_QUERY
// =============================================================================================
//
// One CascadeSim per cascade evaluates the SAME spectrum the GPU draws (the centred sub-block of
// the 256 master h0, see spectrum.ts) at the CPU size n, following the conventions in spectrum.ts.
//
// Time: full fields ("snapshots": height + horizontal displacement AND their exact time
// derivatives, analytic i*omega terms, 3 complex IFFTs) are computed at times j * snapshotDt.
// Any time t is evaluated by cubic Hermite interpolation between the two bracketing snapshots
// using those exact derivatives (error ~ (omega dt)^4 / 384: < 1e-3 of the fastest mode's
// amplitude with the defaults). Velocities are the derivative of the same Hermite curve.
// Snapshots are computed AHEAD of time and amortised over ticks (resumable job: evolve rows,
// row FFTs, column stages, pack rows), so a tick never pays a whole 256^2 x 3 FFT.
//
// Snapshot layout: Float32Array n*n*6, texel (x, y) at 6*(y*n + x):
//   [ H (m, up), E (m, east, WITHOUT choppiness), N (m, north, without choppiness),
//     dH/dt, dE/dt, dN/dt (m/s) ]
// Frame layout ("materialised" field at one time, crossfade + choppiness applied): same 6 channels,
// E / N / dE / dN include choppiness.

export const FIELD_STRIDE = 6;

export interface QueryCascadeSettings {
  readonly size: number;
  readonly snapshotDt: number;
  readonly filter: 'bilinear' | 'cubic';
}

interface Snapshot {
  slot: number;
  version: number;
  j: number;
  valid: boolean;
  lastUse: number;
  readonly data: Float32Array;
}

const MAX_PINS = 12;

export class CascadeSim {
  readonly n: number;
  readonly domain: number;
  readonly offset: number;
  readonly snapshotDt: number;
  readonly cubic: boolean;
  readonly plan: FftPlan;
  /** Materialised fields (2 cache entries), FIELD_STRIDE floats per texel. */
  readonly frames: Float32Array[];
  /** Velocity frames (dH/dt, dE/dt, dN/dt), filled only when materialised withVel. */
  readonly velFrames: Float32Array[];
  readonly frameHasVel: Uint8Array;
  readonly frameTime: Float64Array;
  readonly frameStamp: Float64Array;
  readonly unitsPerSnapshot: number;
  /** Number of snapshots computed so far (diagnostics). */
  computed = 0;

  private readonly omegaIndex: Float64Array;
  private readonly kxn: Float64Array;
  private readonly kzn: Float64Array;
  private readonly slotH0: Float32Array[];
  private readonly slotVersion: Int32Array;
  private readonly re1: Float64Array; private readonly im1: Float64Array;
  private readonly re2: Float64Array; private readonly im2: Float64Array;
  private readonly re3: Float64Array; private readonly im3: Float64Array;
  private readonly pool: Snapshot[];
  private useCounter = 0;
  private readonly pinSlot: Int32Array;
  private readonly pinJ: Float64Array;
  private pinCount = 0;
  // Resumable job
  private jobSnap = -1;
  private jobPhase = 0;
  private jobProgress = 0;
  private readonly rowsEvolve: number;
  private readonly rowsFft: number;

  constructor(def: Readonly<CascadeDef>, cfg: Readonly<QueryCascadeSettings>, masterOmegaIndex: Float32Array, numSlots: number, poolSize: number) {
    const n = cfg.size;
    this.n = n;
    this.domain = def.domain;
    this.offset = def.offset;
    this.snapshotDt = cfg.snapshotDt;
    this.cubic = cfg.filter === 'cubic';
    this.plan = createFftPlan(n);
    const nn = n * n;
    const om = resampleCentred(masterOmegaIndex, new Float32Array(nn), n, 1, false);
    this.omegaIndex = new Float64Array(nn);
    this.kxn = new Float64Array(nn);
    this.kzn = new Float64Array(nn);
    const dk = (2 * Math.PI) / def.domain;
    for (let y = 0; y < n; y++) {
      const kz = Math.fround((y - n / 2) * dk);
      for (let x = 0; x < n; x++) {
        const i = y * n + x;
        const kx = Math.fround((x - n / 2) * dk);
        const kl = Math.hypot(kx, kz) + 1e-4; // reference: length(k) + .0001
        this.kxn[i] = kx / kl;
        this.kzn[i] = kz / kl;
        this.omegaIndex[i] = om[i];
      }
    }
    this.slotH0 = [];
    for (let s = 0; s < numSlots; s++) this.slotH0.push(new Float32Array(nn * 4));
    this.slotVersion = new Int32Array(numSlots).fill(-1);
    this.re1 = new Float64Array(nn); this.im1 = new Float64Array(nn);
    this.re2 = new Float64Array(nn); this.im2 = new Float64Array(nn);
    this.re3 = new Float64Array(nn); this.im3 = new Float64Array(nn);
    this.pool = [];
    for (let i = 0; i < poolSize; i++) {
      this.pool.push({ slot: -1, version: -1, j: 0, valid: false, lastUse: 0, data: new Float32Array(nn * FIELD_STRIDE) });
    }
    this.frames = [new Float32Array(nn * 3), new Float32Array(nn * 3)];
    this.velFrames = [new Float32Array(nn * 3), new Float32Array(nn * 3)];
    this.frameHasVel = new Uint8Array(2);
    this.frameTime = new Float64Array([NaN, NaN]);
    this.frameStamp = new Float64Array([-1, -1]);
    this.pinSlot = new Int32Array(MAX_PINS);
    this.pinJ = new Float64Array(MAX_PINS);
    // ~4096 cells per evolve / pack unit, ~16k butterflies per row-FFT unit, one column stage per unit.
    this.rowsEvolve = Math.max(1, Math.floor(4096 / n));
    this.rowsFft = Math.max(1, Math.floor(16384 / ((n / 2) * this.plan.log2n)));
    const evolveUnits = Math.ceil(n / this.rowsEvolve);
    this.unitsPerSnapshot = evolveUnits + 3 * Math.ceil(n / this.rowsFft) + 3 * (1 + this.plan.log2n) + evolveUnits;
  }

  /** Copy a crossfade slot's master h0 (256 layout) into this cascade's size. Setup / on regeneration. */
  setSlot(slot: number, masterH0: Float32Array, version: number): void {
    resampleCentred(masterH0, this.slotH0[slot], this.n, 4, true);
    this.slotVersion[slot] = version;
    if (this.jobSnap >= 0 && this.pool[this.jobSnap].slot === slot) this.jobSnap = -1;
    for (let i = 0; i < this.pool.length; i++) if (this.pool[i].slot === slot) this.pool[i].valid = false;
    this.frameStamp[0] = this.frameStamp[1] = -1;
  }

  private find(slot: number, j: number): number {
    const v = this.slotVersion[slot];
    for (let i = 0; i < this.pool.length; i++) {
      const s = this.pool[i];
      if (s.valid && s.slot === slot && s.j === j && s.version === v) return i;
    }
    return -1;
  }

  has(slot: number, j: number): boolean { return this.find(slot, j) >= 0; }

  private isPinned(s: Snapshot): boolean {
    for (let p = 0; p < this.pinCount; p++) if (this.pinSlot[p] === s.slot && this.pinJ[p] === s.j) return true;
    return false;
  }

  private acquire(slot: number, j: number): number {
    let best = -1, bestUse = Infinity;
    for (let i = 0; i < this.pool.length; i++) {
      const s = this.pool[i];
      if (i === this.jobSnap) continue;
      if (!s.valid || s.version !== this.slotVersion[s.slot < 0 ? 0 : s.slot]) { best = i; break; }
      if (this.isPinned(s)) continue;
      if (s.lastUse < bestUse) { bestUse = s.lastUse; best = i; }
    }
    if (best < 0) { // everything pinned: take the least recently used anyway
      for (let i = 0; i < this.pool.length; i++) {
        if (i !== this.jobSnap && this.pool[i].lastUse < bestUse) { bestUse = this.pool[i].lastUse; best = i; }
      }
    }
    const s = this.pool[best];
    s.slot = slot; s.version = this.slotVersion[slot]; s.j = j; s.valid = false; s.lastUse = ++this.useCounter;
    return best;
  }

  /** Make snapshot (slot, j) available now (computes it synchronously if needed). Returns its pool index. */
  ensure(slot: number, j: number): number {
    let i = this.find(slot, j);
    if (i >= 0) { this.pool[i].lastUse = ++this.useCounter; return i; }
    if (this.jobSnap >= 0) {
      const js = this.pool[this.jobSnap];
      if (js.slot === slot && js.j === j && js.version === this.slotVersion[slot]) {
        i = this.jobSnap;
        this.runJob(1e9);
        this.pool[i].lastUse = ++this.useCounter;
        return i;
      }
      this.jobSnap = -1; // abandon the background job, it restarts later
    }
    i = this.acquire(slot, j);
    this.startJob(i);
    this.runJob(1e9);
    return i;
  }

  private startJob(i: number): void {
    this.jobSnap = i;
    this.jobPhase = 0;
    this.jobProgress = 0;
    this.pool[i].valid = false;
  }

  /** Advance the background job by up to `budget` units. Returns units used. */
  private runJob(budget: number): number {
    let used = 0;
    const n = this.n, log2n = this.plan.log2n;
    while (this.jobSnap >= 0 && used < budget) {
      const snap = this.pool[this.jobSnap];
      if (this.jobPhase === 0) { // evolve
        const r0 = this.jobProgress, r1 = Math.min(n, r0 + this.rowsEvolve);
        this.evolveRows(snap.slot, snap.j * this.snapshotDt, r0, r1);
        this.jobProgress = r1;
        if (r1 >= n) { this.jobPhase = 1; this.jobProgress = 0; }
      } else if (this.jobPhase === 1) { // row FFTs, 3 arrays
        const rowsTotal = 3 * n;
        const p0 = this.jobProgress, p1 = Math.min(rowsTotal, p0 + this.rowsFft);
        let p = p0;
        while (p < p1) {
          const a = (p / n) | 0, r = p - a * n, rEnd = Math.min(n, r + (p1 - p));
          const re = a === 0 ? this.re1 : a === 1 ? this.re2 : this.re3;
          const im = a === 0 ? this.im1 : a === 1 ? this.im2 : this.im3;
          fftRows(this.plan, re, im, r, rEnd, 1);
          p += rEnd - r;
        }
        this.jobProgress = p1;
        if (p1 >= rowsTotal) { this.jobPhase = 2; this.jobProgress = 0; }
      } else if (this.jobPhase === 2) { // column pass: per array, bit-reverse + log2n stages
        const per = 1 + log2n;
        const a = (this.jobProgress / per) | 0, k = this.jobProgress - a * per;
        const re = a === 0 ? this.re1 : a === 1 ? this.re2 : this.re3;
        const im = a === 0 ? this.im1 : a === 1 ? this.im2 : this.im3;
        if (k === 0) fftColumnsBitReverse(this.plan, re, im);
        else fftColumnsStage(this.plan, re, im, 1 << k, 1);
        this.jobProgress++;
        if (this.jobProgress >= 3 * per) { this.jobPhase = 3; this.jobProgress = 0; }
      } else { // pack
        const r0 = this.jobProgress, r1 = Math.min(n, r0 + this.rowsEvolve);
        this.packRows(snap.data, r0, r1);
        this.jobProgress = r1;
        if (r1 >= n) {
          snap.valid = true;
          snap.lastUse = ++this.useCounter;
          this.jobSnap = -1;
          this.computed++;
        }
      }
      used++;
    }
    return used;
  }

  private evolveRows(slot: number, t: number, r0: number, r1: number): void {
    const n = this.n, h0 = this.slotH0[slot];
    const om = this.omegaIndex, kxn = this.kxn, kzn = this.kzn;
    const re1 = this.re1, im1 = this.im1, re2 = this.re2, im2 = this.im2, re3 = this.re3, im3 = this.im3;
    let tm = t % LOOP_PERIOD;
    if (tm < 0) tm += LOOP_PERIOD;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < n; x++) {
        const i = y * n + x, q = 4 * i;
        const m = om[i];
        const omega = m * OMEGA_UNIT;
        const phase = OMEGA_UNIT * ((m * tm) % LOOP_PERIOD);
        const c = Math.cos(phase), s = Math.sin(phase);
        const hr = h0[q], hi = h0[q + 1], mr = h0[q + 2], mi = h0[q + 3];
        // h0 * e^{-i w t}, h0m * e^{+i w t}
        const ar = hr * c + hi * s, ai = hi * c - hr * s;
        const br = mr * c - mi * s, bi = mi * c + mr * s;
        const htr = ar + br, hti = ai + bi;
        // dht = w * (-i a + i b) = w * ((ai - bi), (br - ar))
        const dtr = omega * (ai - bi), dti = omega * (br - ar);
        const kx = kxn[i], kz = kzn[i];
        // C1 = ht (1 + kx/k)          -> Hraw + i Da
        re1[i] = htr * (1 + kx); im1[i] = hti * (1 + kx);
        // C2 = -i (kz/k) ht + i dht    -> Db + i dHraw
        re2[i] = kz * hti - dti; im2[i] = -kz * htr + dtr;
        // C3 = dht (kz/k - i kx/k)     -> dDa + i dDb
        re3[i] = kx * dti + kz * dtr; im3[i] = -kx * dtr + kz * dti;
      }
    }
  }

  private packRows(out: Float32Array, r0: number, r1: number): void {
    const n = this.n;
    const re1 = this.re1, im1 = this.im1, re2 = this.re2, im2 = this.im2, re3 = this.re3, im3 = this.im3;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < n; x++) {
        const i = y * n + x, o = i * FIELD_STRIDE;
        const s = ((x + y) & 1) === 0 ? 1 : -1; // centred-spectrum checkerboard
        out[o] = -re1[i] * s;      // up (reference: -f.x)
        out[o + 1] = im1[i] * s;   // east  = +Da
        out[o + 2] = -re2[i] * s;  // north = -Db (grid b is south)
        out[o + 3] = -im2[i] * s;
        out[o + 4] = re3[i] * s;
        out[o + 5] = -im3[i] * s;
      }
    }
  }

  /**
   * Per-tick scheduling at tick time T: snapshots j0 = floor(T/dt) and j0+1 of the active slots are
   * made available (synchronously if a prefetch was not finished), then the background job works
   * towards j0+2 (and the pending slot's j0..j0+2, so a crossfade can start without a hitch).
   * Allocation-free.
   */
  schedule(T: number, tickDt: number, slotA: number, slotB: number, pending: number): void {
    const dt = this.snapshotDt;
    const j0 = Math.floor(T / dt);
    this.pinCount = 0;
    this.pin(slotA, j0); this.pin(slotA, j0 + 1); this.pin(slotA, j0 + 2);
    if (slotB >= 0) { this.pin(slotB, j0); this.pin(slotB, j0 + 1); this.pin(slotB, j0 + 2); }
    if (pending >= 0) { this.pin(pending, j0); this.pin(pending, j0 + 1); this.pin(pending, j0 + 2); }
    this.ensure(slotA, j0); this.ensure(slotA, j0 + 1);
    if (slotB >= 0) { this.ensure(slotB, j0); this.ensure(slotB, j0 + 1); }
    // Background budget: enough to finish one snapshot per active slot (+2 for a pending slot)
    // per snapshot interval, with 50 % margin.
    const perInterval = 1 + (slotB >= 0 ? 1 : 0) + (pending >= 0 ? 2 : 0);
    let budget = Math.ceil((this.unitsPerSnapshot * perInterval * Math.max(tickDt, 1e-4) / dt) * 1.5) + 1;
    while (budget > 0) {
      if (this.jobSnap < 0) {
        let si = -1, sj = 0;
        if (pending >= 0 && !this.has(pending, j0)) { si = pending; sj = j0; }
        else if (pending >= 0 && !this.has(pending, j0 + 1)) { si = pending; sj = j0 + 1; }
        else if (!this.has(slotA, j0 + 2)) { si = slotA; sj = j0 + 2; }
        else if (slotB >= 0 && !this.has(slotB, j0 + 2)) { si = slotB; sj = j0 + 2; }
        else if (pending >= 0 && !this.has(pending, j0 + 2)) { si = pending; sj = j0 + 2; }
        if (si < 0) break;
        this.startJob(this.acquire(si, sj));
      }
      budget -= this.runJob(budget);
    }
  }

  private pin(slot: number, j: number): void {
    if (this.pinCount < MAX_PINS) { this.pinSlot[this.pinCount] = slot; this.pinJ[this.pinCount] = j; this.pinCount++; }
  }

  /**
   * Materialise the field at time t into frames[frame] (H, E, N; stride 3) and, when withVel, into
   * velFrames[frame] (dH/dt, dE/dt, dN/dt; stride 3): sum over the slots (slot1 < 0: one slot) of
   * W_k * Hermite(slot k), horizontal channels times choppiness `chop`; the velocity channels use
   * the Hermite derivative plus the dW_k / dchop terms. Allocation-free.
   */
  materialise(frame: number, t: number, stamp: number, withVel: boolean,
    slot0: number, w0: number, dw0: number, slot1: number, w1: number, dw1: number, chop: number, dchop: number): void {
    const dt = this.snapshotDt;
    const j = Math.floor(t / dt);
    const u = (t - j * dt) / dt;
    const u2 = u * u, u3 = u2 * u;
    const h00 = 2 * u3 - 3 * u2 + 1, h10 = (u3 - 2 * u2 + u) * dt, h01 = -2 * u3 + 3 * u2, h11 = (u3 - u2) * dt;
    const d00 = (6 * u2 - 6 * u) / dt, d10 = 3 * u2 - 4 * u + 1, d01 = (-6 * u2 + 6 * u) / dt, d11 = 3 * u2 - 2 * u;
    const out = this.frames[frame], vel = this.velFrames[frame];
    const nn = this.n * this.n;
    const two = slot1 >= 0 && !(w1 === 0 && dw1 === 0);
    // Pin the snapshots while acquiring the others (pool >= 6 entries).
    const ia0 = this.ensure(slot0, j); const p0 = this.pool[ia0]; p0.lastUse = 1e15;
    const ib0 = this.ensure(slot0, j + 1); const q0 = this.pool[ib0]; q0.lastUse = 1e15;
    let A1 = p0.data, B1 = q0.data;
    if (two) {
      const ia1 = this.ensure(slot1, j); const p1 = this.pool[ia1]; p1.lastUse = 1e15;
      const ib1 = this.ensure(slot1, j + 1);
      A1 = p1.data; B1 = this.pool[ib1].data;
      p1.lastUse = ++this.useCounter; this.pool[ib1].lastUse = ++this.useCounter;
    }
    const A0 = p0.data, B0 = q0.data;
    p0.lastUse = ++this.useCounter; q0.lastUse = ++this.useCounter;
    for (let i = 0; i < nn; i++) {
      const o = i * FIELD_STRIDE, q = i * 3;
      for (let c = 0; c < 3; c++) {
        let a = A0[o + c], av = A0[o + c + 3], b = B0[o + c], bv = B0[o + c + 3];
        let v = w0 * (h00 * a + h10 * av + h01 * b + h11 * bv);
        let d = 0;
        if (withVel) d = w0 * (d00 * a + d10 * av + d01 * b + d11 * bv) + dw0 * (h00 * a + h10 * av + h01 * b + h11 * bv);
        if (two) {
          a = A1[o + c]; av = A1[o + c + 3]; b = B1[o + c]; bv = B1[o + c + 3];
          const v1 = h00 * a + h10 * av + h01 * b + h11 * bv;
          v += w1 * v1;
          if (withVel) d += w1 * (d00 * a + d10 * av + d01 * b + d11 * bv) + dw1 * v1;
        }
        if (c === 0) { out[q] = v; if (withVel) vel[q] = d; }
        else { out[q + c] = chop * v; if (withVel) vel[q + c] = chop * d + dchop * v; }
      }
    }
    this.frameTime[frame] = t;
    this.frameStamp[frame] = stamp;
    this.frameHasVel[frame] = withVel ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Field sampling (allocation-free) of a materialised frame (stride 3). Results go to SAMPLE[0..2].
// ---------------------------------------------------------------------------------------------
export const SAMPLE = new Float64Array(3);
/** Floats per texel of a materialised frame / velocity frame. */
export const FRAME_STRIDE = 3;

/**
 * Bilinear sample of the 3 channels of a periodic n x n frame at tile position (a, b) metres
 * (texel (x, y) sits at (x, y) * L / n). Writes SAMPLE[0..2].
 */
export function sampleBilinear(data: Float32Array, n: number, L: number, a: number, b: number): void {
  const inv = n / L, mask = n - 1;
  let sa = (a - L * Math.floor(a / L)) * inv;
  let sb = (b - L * Math.floor(b / L)) * inv;
  let x0 = Math.floor(sa), y0 = Math.floor(sb);
  const fx = sa - x0, fy = sb - y0;
  x0 &= mask; y0 &= mask;
  const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
  const i00 = (y0 * n + x0) * FRAME_STRIDE, i10 = (y0 * n + x1) * FRAME_STRIDE;
  const i01 = (y1 * n + x0) * FRAME_STRIDE, i11 = (y1 * n + x1) * FRAME_STRIDE;
  const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
  SAMPLE[0] = w00 * data[i00] + w10 * data[i10] + w01 * data[i01] + w11 * data[i11];
  SAMPLE[1] = w00 * data[i00 + 1] + w10 * data[i10 + 1] + w01 * data[i01 + 1] + w11 * data[i11 + 1];
  SAMPLE[2] = w00 * data[i00 + 2] + w10 * data[i10 + 2] + w01 * data[i01 + 2] + w11 * data[i11 + 2];
}

/** Horizontal-displacement gradient of the last sampleBilinearGrad: [dE/da, dE/db, dN/da, dN/db]. */
export const GRAD = new Float64Array(4);

/** sampleBilinear(ch = 0) plus the exact gradient of the bilinear E / N channels (for Newton inversion). */
export function sampleBilinearGrad(data: Float32Array, n: number, L: number, a: number, b: number): void {
  const inv = n / L, mask = n - 1;
  const sa = (a - L * Math.floor(a / L)) * inv;
  const sb = (b - L * Math.floor(b / L)) * inv;
  let x0 = Math.floor(sa), y0 = Math.floor(sb);
  const fx = sa - x0, fy = sb - y0;
  x0 &= mask; y0 &= mask;
  const x1 = (x0 + 1) & mask, y1 = (y0 + 1) & mask;
  const i00 = (y0 * n + x0) * FRAME_STRIDE, i10 = (y0 * n + x1) * FRAME_STRIDE;
  const i01 = (y1 * n + x0) * FRAME_STRIDE, i11 = (y1 * n + x1) * FRAME_STRIDE;
  const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
  SAMPLE[0] = w00 * data[i00] + w10 * data[i10] + w01 * data[i01] + w11 * data[i11];
  const e00 = data[i00 + 1], e10 = data[i10 + 1], e01 = data[i01 + 1], e11 = data[i11 + 1];
  const n00 = data[i00 + 2], n10 = data[i10 + 2], n01 = data[i01 + 2], n11 = data[i11 + 2];
  SAMPLE[1] = w00 * e00 + w10 * e10 + w01 * e01 + w11 * e11;
  SAMPLE[2] = w00 * n00 + w10 * n10 + w01 * n01 + w11 * n11;
  GRAD[0] = ((1 - fy) * (e10 - e00) + fy * (e11 - e01)) * inv;
  GRAD[1] = ((1 - fx) * (e01 - e00) + fx * (e11 - e10)) * inv;
  GRAD[2] = ((1 - fy) * (n10 - n00) + fy * (n11 - n01)) * inv;
  GRAD[3] = ((1 - fx) * (n01 - n00) + fx * (n11 - n10)) * inv;
}

/** sampleCubic(ch = 0) plus the exact gradient of the Catmull-Rom E / N channels (into GRAD). */
export function sampleCubicGrad(data: Float32Array, n: number, L: number, a: number, b: number): void {
  const inv = n / L, mask = n - 1;
  const sa = (a - L * Math.floor(a / L)) * inv;
  const sb = (b - L * Math.floor(b / L)) * inv;
  const xf = Math.floor(sa), yf = Math.floor(sb);
  const fx = sa - xf, fy = sb - yf;
  const fx2 = fx * fx, fx3 = fx2 * fx, fy2 = fy * fy, fy3 = fy2 * fy;
  const wx0 = -0.5 * fx3 + fx2 - 0.5 * fx, wx1 = 1.5 * fx3 - 2.5 * fx2 + 1, wx2 = -1.5 * fx3 + 2 * fx2 + 0.5 * fx, wx3 = 0.5 * fx3 - 0.5 * fx2;
  const dx0 = -1.5 * fx2 + 2 * fx - 0.5, dx1 = 4.5 * fx2 - 5 * fx, dx2 = -4.5 * fx2 + 4 * fx + 0.5, dx3 = 1.5 * fx2 - fx;
  const x0 = ((xf - 1) & mask), x1 = (xf & mask), x2 = ((xf + 1) & mask), x3 = ((xf + 2) & mask);
  let s0 = 0, s1 = 0, s2 = 0, ea = 0, eb = 0, na = 0, nb = 0;
  for (let r = 0; r < 4; r++) {
    const yy = (yf - 1 + r) & mask;
    let wy: number, dy: number;
    if (r === 0) { wy = -0.5 * fy3 + fy2 - 0.5 * fy; dy = -1.5 * fy2 + 2 * fy - 0.5; }
    else if (r === 1) { wy = 1.5 * fy3 - 2.5 * fy2 + 1; dy = 4.5 * fy2 - 5 * fy; }
    else if (r === 2) { wy = -1.5 * fy3 + 2 * fy2 + 0.5 * fy; dy = -4.5 * fy2 + 4 * fy + 0.5; }
    else { wy = 0.5 * fy3 - 0.5 * fy2; dy = 1.5 * fy2 - fy; }
    const row = yy * n;
    const i0 = (row + x0) * FRAME_STRIDE, i1 = (row + x1) * FRAME_STRIDE;
    const i2 = (row + x2) * FRAME_STRIDE, i3 = (row + x3) * FRAME_STRIDE;
    const h = wx0 * data[i0] + wx1 * data[i1] + wx2 * data[i2] + wx3 * data[i3];
    const e = wx0 * data[i0 + 1] + wx1 * data[i1 + 1] + wx2 * data[i2 + 1] + wx3 * data[i3 + 1];
    const m = wx0 * data[i0 + 2] + wx1 * data[i1 + 2] + wx2 * data[i2 + 2] + wx3 * data[i3 + 2];
    const ed = dx0 * data[i0 + 1] + dx1 * data[i1 + 1] + dx2 * data[i2 + 1] + dx3 * data[i3 + 1];
    const md = dx0 * data[i0 + 2] + dx1 * data[i1 + 2] + dx2 * data[i2 + 2] + dx3 * data[i3 + 2];
    s0 += wy * h; s1 += wy * e; s2 += wy * m;
    ea += wy * ed; na += wy * md; eb += dy * e; nb += dy * m;
  }
  SAMPLE[0] = s0; SAMPLE[1] = s1; SAMPLE[2] = s2;
  GRAD[0] = ea * inv; GRAD[1] = eb * inv; GRAD[2] = na * inv; GRAD[3] = nb * inv;
}

/** Catmull-Rom (interpolating bicubic) sample; same contract as sampleBilinear. */
export function sampleCubic(data: Float32Array, n: number, L: number, a: number, b: number): void {
  const inv = n / L, mask = n - 1;
  const sa = (a - L * Math.floor(a / L)) * inv;
  const sb = (b - L * Math.floor(b / L)) * inv;
  const xf = Math.floor(sa), yf = Math.floor(sb);
  const fx = sa - xf, fy = sb - yf;
  // Catmull-Rom weights
  const fx2 = fx * fx, fx3 = fx2 * fx, fy2 = fy * fy, fy3 = fy2 * fy;
  const wx0 = -0.5 * fx3 + fx2 - 0.5 * fx, wx1 = 1.5 * fx3 - 2.5 * fx2 + 1, wx2 = -1.5 * fx3 + 2 * fx2 + 0.5 * fx, wx3 = 0.5 * fx3 - 0.5 * fx2;
  const wy0 = -0.5 * fy3 + fy2 - 0.5 * fy, wy1 = 1.5 * fy3 - 2.5 * fy2 + 1, wy2 = -1.5 * fy3 + 2 * fy2 + 0.5 * fy, wy3 = 0.5 * fy3 - 0.5 * fy2;
  const x0 = ((xf - 1) & mask), x1 = (xf & mask), x2 = ((xf + 1) & mask), x3 = ((xf + 2) & mask);
  let s0 = 0, s1 = 0, s2 = 0;
  for (let r = 0; r < 4; r++) {
    const yy = (yf - 1 + r) & mask;
    const wy = r === 0 ? wy0 : r === 1 ? wy1 : r === 2 ? wy2 : wy3;
    const row = yy * n;
    const i0 = (row + x0) * FRAME_STRIDE, i1 = (row + x1) * FRAME_STRIDE;
    const i2 = (row + x2) * FRAME_STRIDE, i3 = (row + x3) * FRAME_STRIDE;
    s0 += wy * (wx0 * data[i0] + wx1 * data[i1] + wx2 * data[i2] + wx3 * data[i3]);
    s1 += wy * (wx0 * data[i0 + 1] + wx1 * data[i1 + 1] + wx2 * data[i2 + 1] + wx3 * data[i3 + 1]);
    s2 += wy * (wx0 * data[i0 + 2] + wx1 * data[i1 + 2] + wx2 * data[i2 + 2] + wx3 * data[i3 + 2]);
  }
  SAMPLE[0] = s0; SAMPLE[1] = s1; SAMPLE[2] = s2;
}
