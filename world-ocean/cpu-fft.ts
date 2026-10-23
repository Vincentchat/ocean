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

// ---------------------------------------------------------------------------------------------
// Faster 2-D path used by the simulation: column passes (contiguous over x) + transposes.
// ---------------------------------------------------------------------------------------------

/** In-place transpose of an n x n complex grid (blocked, cache friendly). */
export function transposeSquare(n: number, re: Float64Array, im: Float64Array): void {
  const B = 16;
  for (let by = 0; by < n; by += B) {
    for (let bx = by; bx < n; bx += B) {
      const yEnd = Math.min(by + B, n), xEnd = Math.min(bx + B, n);
      for (let y = by; y < yEnd; y++) {
        for (let x = bx === by ? y + 1 : bx; x < xEnd; x++) {
          const a = y * n + x, b = x * n + y;
          let t = re[a]; re[a] = re[b]; re[b] = t;
          t = im[a]; im[a] = im[b]; im[b] = t;
        }
      }
    }
  }
}

/**
 * Column pass, TWO butterfly stages at once (widths `span` and 2*span; radix-2^2), vectorised over x.
 * Same result as fftColumnsStage(span) then fftColumnsStage(2*span), half the memory passes.
 */
export function fftColumnsStage2(plan: FftPlan, re: Float64Array, im: Float64Array, span: number, sign: number): void {
  const n = plan.n, cs = plan.cos, sn = plan.sin;
  const h = span >> 1, span2 = span << 1;
  const step1 = n / span, step2 = n / span2;
  for (let base = 0; base < n; base += span2) {
    for (let j = 0; j < h; j++) {
      const w1 = j * step1, c1 = cs[w1], s1 = sign * sn[w1];
      const w2 = j * step2, c2 = cs[w2], s2 = sign * sn[w2];
      const w3 = (j + h) * step2, c3 = cs[w3], s3 = sign * sn[w3];
      const ra = (base + j) * n, rb = ra + h * n, rc = ra + span * n, rd = rc + h * n;
      for (let x = 0; x < n; x++) {
        const ia = ra + x, ib = rb + x, ic = rc + x, id = rd + x;
        // stage `span`: (a, b) and (c, d) with w1
        let tr = re[ib] * c1 - im[ib] * s1, ti = re[ib] * s1 + im[ib] * c1;
        const ar = re[ia] + tr, ai = im[ia] + ti, br = re[ia] - tr, bi = im[ia] - ti;
        tr = re[id] * c1 - im[id] * s1; ti = re[id] * s1 + im[id] * c1;
        const cr = re[ic] + tr, ci = im[ic] + ti, dr = re[ic] - tr, di = im[ic] - ti;
        // stage `2 span`: (a, c) with w2, (b, d) with w3
        tr = cr * c2 - ci * s2; ti = cr * s2 + ci * c2;
        re[ia] = ar + tr; im[ia] = ai + ti; re[ic] = ar - tr; im[ic] = ai - ti;
        tr = dr * c3 - di * s3; ti = dr * s3 + di * c3;
        re[ib] = br + tr; im[ib] = bi + ti; re[id] = br - tr; im[id] = bi - ti;
      }
    }
  }
}

/** Steps of one column pass done with fftColumnsStep: bit-reverse + radix-2^2 stage pairs (+ one radix-2 stage if log2 n is odd). */
export function columnPassSteps(plan: FftPlan): number {
  return 1 + (plan.log2n >> 1) + (plan.log2n & 1);
}

/** Step k (0 .. columnPassSteps - 1) of a column pass (transform along the row index y). */
export function fftColumnsStep(plan: FftPlan, re: Float64Array, im: Float64Array, k: number, sign: number): void {
  if (k === 0) { fftColumnsBitReverse(plan, re, im); return; }
  const pairs = plan.log2n >> 1;
  if (k <= pairs) fftColumnsStage2(plan, re, im, 1 << (2 * k - 1), sign);
  else fftColumnsStage(plan, re, im, plan.n, sign); // the odd last stage
}

/**
 * Same transform as fft2d, computed as column pass, transpose, column pass, transpose: every pass
 * runs contiguous over x (about 1.5x faster than the row-by-row 1-D FFTs for n = 256).
 */
export function fft2dFast(plan: FftPlan, re: Float64Array, im: Float64Array, sign: number): void {
  const steps = columnPassSteps(plan);
  for (let k = 0; k < steps; k++) fftColumnsStep(plan, re, im, k, sign);
  transposeSquare(plan.n, re, im);
  for (let k = 0; k < steps; k++) fftColumnsStep(plan, re, im, k, sign);
  transposeSquare(plan.n, re, im);
}

// =============================================================================================
// CPU cascade simulation for the WATER_QUERY
// =============================================================================================
//
// One CascadeSim per cascade evaluates the SAME spectrum the GPU draws (the centred sub-block of
// the 256 master h0, see spectrum.ts) at the CPU size n, following the conventions in spectrum.ts.
//
// Sources: the evolved spectrum is h0(t) = lerp(h0[from], h0[to], w(t)) with the crossfade weight
// w(t) = smoothstep((t - start) / duration) (w = 1 when from == to): exactly what the GPU draws.
// A source is (id, from, to, start, duration); at most two are alive per cascade: the ACTIVE one
// and a PENDING one (the next crossfade, defined ahead of time with a start in the future so its
// snapshots can be computed in the background; switching to it changes nothing before `start`).
//
// Time: full fields ("snapshots": height + horizontal displacement AND their exact time
// derivatives, analytic i*omega terms incl. the dw/dt crossfade term, 3 complex IFFTs) are computed
// at times j * snapshotDt. Any time t is evaluated by cubic Hermite interpolation between the two
// bracketing snapshots using those exact derivatives (error ~ (omega dt)^4 / 384: < 0.3 % of Hs
// with the defaults). Velocities are the derivative of the same Hermite curve.
//
// Cost control (the 60 Hz budget):
//  - Snapshots are computed AHEAD of time (two intervals ahead) by a resumable job (evolve row
//    chunks, column-FFT steps, transposes, pack row chunks), at a steady rate per tick, so a tick
//    never pays a whole 256^2 x 3 FFT.
//  - Frames are LAZY: materialise() only records the Hermite coefficients and two snapshot
//    references. Positions and velocities are filled per 16 x 16 texel block on the first query
//    that touches the block (ensurePos / ensureVel). No query ever fills a whole frame.
//
// Snapshot layout: Float32Array n*n*6, texel (x, y) at 6*(y*n + x):
//   [ H (m, up), E (m, east, WITHOUT choppiness), N (m, north, without choppiness),
//     dH/dt, dE/dt, dN/dt (m/s) ]
// Frames ("materialised" field at one time, choppiness applied): positions Float32Array n*n*3
// [H, E, N] and velocities n*n*3 [dH/dt, dE/dt, dN/dt]. Two frames are cached.

export const FIELD_STRIDE = 6;


export interface QueryCascadeSettings {
  readonly size: number;
  readonly snapshotDt: number;
  readonly filter: 'bilinear' | 'cubic';
}

interface Snapshot {
  src: number;   // source id
  j: number;     // time index (t = j * snapshotDt); a float64 integer (no int32 overflow)
  valid: boolean;
  lastUse: number;
  readonly data: Float32Array;
}

/** Active j0-1 .. j0+3 + pending jp .. jp+2. */
const MAX_PINS = 8;
/** lastUse marker that keeps a snapshot from being evicted during materialise() (a Smi: no boxing). */
const PINNED_USE = 0x3fffffff;
/** Lazy-frame block size, texels per side (a power of two >= 4 so a 4-tap filter spans <= 2 blocks). */
const BLOCK = 16;
/** Target work per job unit, cells: evolve / pack chunks of this many cells cost about one FFT step (~0.1-0.2 ms). */
const UNIT_CELLS = 4096;
/** Margin on the steady per-tick work rate (the job also catches up when it falls behind). */
const RATE_MARGIN = 1.15;

/**
 * Shared memory of a CascadeSim and its snapshot worker (ocean-worker.ts): slot h0 sub-grids, the
 * snapshot pool, the source table and a small control block. Main thread and worker run the same
 * code on the same numbers, so a snapshot is bit-identical whichever thread computed it.
 */
export interface SharedSimBuffers {
  readonly slots: SharedArrayBuffer[];
  readonly pool: SharedArrayBuffer[];
  readonly srcI: SharedArrayBuffer;   // Int32 x 6: id[2], from[2], to[2]
  readonly srcF: SharedArrayBuffer;   // Float64 x 4: start[2], duration[2]
  readonly frames: SharedArrayBuffer[]; // 3 position frames: two in use, one being prefilled
  readonly ctrl: SharedArrayBuffer;   // Int32 x 12, see C_*
  readonly ctrlF: SharedArrayBuffer;  // Float64 x 16: [j, prefill t, prefill h00 h10 h01 h11 chop, -, regen params x 8]
}
// Control block. Snapshot request: state / pool / entry; prefill request: pstate / pa / pb / pbuf;
// ready / quit; bell (incremented with every request; the worker sleeps on it).
const C_STATE = 0, C_POOL = 1, C_ENTRY = 2, C_READY = 3, C_QUIT = 4, C_PSTATE = 5, C_PA = 6, C_PB = 7, C_PBUF = 8, C_BELL = 9;
/** Spectrum regeneration request (served by ocean-worker.ts): state, target slot. Params at ctrlF[F_REGEN..+7]. */
export const C_RSTATE = 10, C_RSLOT = 11, F_REGEN = 8;
const F_J = 0, F_PT = 1, F_PCOEF = 2;

/**
 * Hermite-in-time position fill of one lazy block (positions; choppiness on the horizontal
 * channels). coef[c..c+3] = h00 h10 h01 h11, coef[chopIdx] = choppiness. The single implementation
 * used by the main thread (fillBlock) and the worker prefill: identical numbers either way.
 */
function fillPositions(out: Float32Array, A0: Float32Array, B0: Float32Array, coef: Float64Array, c: number, chopIdx: number,
  n: number, shift: number, bx: number, by: number): void {
  const B = 1 << shift;
  const a0 = coef[c], a1 = coef[c + 1], b0 = coef[c + 2], b1 = coef[c + 3], chop = coef[chopIdx];
  const ca0 = chop * a0, ca1 = chop * a1, cb0 = chop * b0, cb1 = chop * b1;
  for (let y = by * B, yEnd = y + B; y < yEnd; y++) {
    for (let x = bx * B, xEnd = x + B; x < xEnd; x++) {
      const i = y * n + x, o = i * 6, q = i * 3;
      out[q] = a0 * A0[o] + a1 * A0[o + 3] + b0 * B0[o] + b1 * B0[o + 3];
      out[q + 1] = ca0 * A0[o + 1] + ca1 * A0[o + 4] + cb0 * B0[o + 1] + cb1 * B0[o + 4];
      out[q + 2] = ca0 * A0[o + 2] + ca1 * A0[o + 5] + cb0 * B0[o + 2] + cb1 * B0[o + 5];
    }
  }
}
export const S_IDLE = 0, S_REQUESTED = 1, S_DONE = 2;

export class CascadeSim {
  readonly n: number;
  readonly domain: number;
  readonly offset: number;
  readonly snapshotDt: number;
  readonly cubic: boolean;
  readonly plan: FftPlan;
  /**
   * Per-instance argument block for calls taking doubles (setSource: [start, duration]; schedule:
   * [T, tickDt, switchTime]; materialise: [t, chop, dchop]). Doubles passed as call arguments to a
   * non-inlined function are boxed by V8 (a heap allocation per call); a typed-array slot is not.
   */
  readonly args = new Float64Array(4);
  /** Materialised positions (2 cache entries), 3 floats per texel: H, E, N. */
  readonly frames: Float32Array[];
  /** Materialised velocities (dH/dt, dE/dt, dN/dt), same layout. */
  readonly velFrames: Float32Array[];
  readonly frameTime: Float64Array;
  readonly frameStamp: Float64Array;
  /** Lazy blocks: blockValid[frame][by * nb + bx] (positions), velValid[frame][...] (velocities). */
  readonly blockShift: number;
  readonly nb: number;
  readonly blockValid: Uint8Array[];
  readonly velValid: Uint8Array[];
  readonly unitsPerSnapshot: number;
  /** Number of snapshots computed so far (diagnostics). */
  computed = 0;
  /** Snapshots that had to be computed synchronously (prefetch missed; diagnostics). */
  syncComputed = 0;
  /** Lazy blocks filled so far (positions / velocities; diagnostics). */
  posBlocksFilled = 0;
  velBlocksFilled = 0;

  private readonly omegaIndex: Float64Array;
  private readonly kxn: Float64Array;
  private readonly kzn: Float64Array;
  private readonly slotH0: Float32Array[];
  // Source table: entry 0 = active, 1 = pending (id -1 = empty).
  private readonly srcId: Int32Array;
  private readonly srcFrom: Int32Array;
  private readonly srcTo: Int32Array;
  private readonly srcStart: Float64Array;
  private readonly srcDur: Float64Array;
  private readonly re1: Float64Array; private readonly im1: Float64Array;
  private readonly re2: Float64Array; private readonly im2: Float64Array;
  private readonly re3: Float64Array; private readonly im3: Float64Array;
  private readonly pool: Snapshot[];
  private useCounter = 0;
  private readonly pinSrc: Int32Array;
  private readonly pinJ: Float64Array;
  private pinCount = 0;
  // Resumable job
  private jobSnap = -1;
  private jobPhase = 0;
  private jobProgress = 0;
  private readonly rowsChunk: number;
  private readonly colSteps: number;
  // Lazy frame state: per frame 2 pool references (A, B; -1 = none) + coefficients.
  private readonly lazyRef: Int32Array;
  /** Per frame (stride LC): h00 h10 h01 h11 d00 d10 d01 d11 chop dchop. */
  readonly lazyCoef: Float64Array;

  /** Shared buffers when the snapshot work can run in a worker (null: everything on this thread). */
  readonly shared: SharedSimBuffers | null;
  private readonly ctrl: Int32Array | null;
  private readonly ctrlF: Float64Array | null;
  /** Shared position-frame buffers (worker mode): frames[f] = frameBufs[frameBuf[f]], prefill target frameBufs[preBuf]. */
  private readonly frameBufs: Float32Array[];
  private readonly frameBuf = new Int32Array([0, 1]);
  private preBuf = 2;
  /** Prefill in flight / done for time preT with pool refs (preA, preB). */
  private preBusy = false;
  private preA = -1;
  private preB = -1;
  /** Frames adopted from the worker's prefill (diagnostics). */
  prefillsAdopted = 0;
  private remoteSnap = -1;
  private remoteSrc = -1;
  private remoteJ = 0;
  /** Snapshots computed by the worker (diagnostics). */
  remoteComputed = 0;

  /**
   * shared: undefined = plain memory; true = allocate SharedArrayBuffers (main thread, a worker may
   * attach); a SharedSimBuffers object = attach to them (worker side).
   */
  constructor(def: Readonly<CascadeDef>, cfg: Readonly<QueryCascadeSettings>, masterOmegaIndex: Float32Array, numSlots: number, poolSize: number,
    shared?: true | SharedSimBuffers) {
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
    let sh: SharedSimBuffers | null = null;
    if (shared === true) {
      const slots: SharedArrayBuffer[] = [], pool: SharedArrayBuffer[] = [];
      for (let s = 0; s < numSlots; s++) slots.push(new SharedArrayBuffer(nn * 4 * 4));
      for (let i = 0; i < poolSize; i++) pool.push(new SharedArrayBuffer(nn * FIELD_STRIDE * 4));
      const frames = [new SharedArrayBuffer(nn * 3 * 4), new SharedArrayBuffer(nn * 3 * 4), new SharedArrayBuffer(nn * 3 * 4)];
      sh = { slots, pool, frames, srcI: new SharedArrayBuffer(6 * 4), srcF: new SharedArrayBuffer(4 * 8), ctrl: new SharedArrayBuffer(12 * 4), ctrlF: new SharedArrayBuffer(16 * 8) };
    } else if (shared) sh = shared;
    this.shared = sh;
    this.slotH0 = [];
    for (let s = 0; s < numSlots; s++) this.slotH0.push(sh ? new Float32Array(sh.slots[s]) : new Float32Array(nn * 4));
    const srcI = sh ? new Int32Array(sh.srcI) : new Int32Array(6), srcF = sh ? new Float64Array(sh.srcF) : new Float64Array(4);
    this.srcId = srcI.subarray(0, 2);
    this.srcFrom = srcI.subarray(2, 4);
    this.srcTo = srcI.subarray(4, 6);
    this.srcStart = srcF.subarray(0, 2);
    this.srcDur = srcF.subarray(2, 4);
    if (shared === true || !sh) { this.srcId.fill(-1); this.srcDur.fill(1); }
    this.ctrl = sh ? new Int32Array(sh.ctrl) : null;
    this.ctrlF = sh ? new Float64Array(sh.ctrlF) : null;
    this.re1 = new Float64Array(nn); this.im1 = new Float64Array(nn);
    this.re2 = new Float64Array(nn); this.im2 = new Float64Array(nn);
    this.re3 = new Float64Array(nn); this.im3 = new Float64Array(nn);
    this.pool = [];
    for (let i = 0; i < poolSize; i++) {
      this.pool.push({ src: -1, j: 0, valid: false, lastUse: 0, data: sh ? new Float32Array(sh.pool[i]) : new Float32Array(nn * FIELD_STRIDE) });
    }
    this.frameBufs = sh ? [new Float32Array(sh.frames[0]), new Float32Array(sh.frames[1]), new Float32Array(sh.frames[2])] : [];
    this.frames = sh ? [this.frameBufs[0], this.frameBufs[1]] : [new Float32Array(nn * 3), new Float32Array(nn * 3)];
    this.velFrames = [new Float32Array(nn * 3), new Float32Array(nn * 3)];
    this.frameTime = new Float64Array([NaN, NaN]);
    this.frameStamp = new Float64Array([-1, -1]);
    const block = Math.min(BLOCK, n);
    this.blockShift = Math.round(Math.log2(block));
    this.nb = n >> this.blockShift;
    this.blockValid = [new Uint8Array(this.nb * this.nb), new Uint8Array(this.nb * this.nb)];
    this.velValid = [new Uint8Array(this.nb * this.nb), new Uint8Array(this.nb * this.nb)];
    this.lazyRef = new Int32Array(4).fill(-1);
    this.lazyCoef = new Float64Array(20);
    this.pinSrc = new Int32Array(MAX_PINS);
    this.pinJ = new Float64Array(MAX_PINS);
    this.rowsChunk = Math.max(1, Math.floor(UNIT_CELLS / n));
    this.colSteps = columnPassSteps(this.plan);
    const chunks = Math.ceil(n / this.rowsChunk);
    this.unitsPerSnapshot = chunks + 3 * (2 * this.colSteps + 2) + chunks;
  }

  /**
   * Copy a slot's master h0 (256 layout) into this cascade's size. Snapshots of sources that use
   * the slot become invalid. Setup / on regeneration (never per tick).
   */
  setSlot(slot: number, masterH0: Float32Array): void {
    resampleCentred(masterH0, this.slotH0[slot], this.n, 4, true);
    for (let e = 0; e < 2; e++) {
      const id = this.srcId[e];
      if (id >= 0 && (this.srcFrom[e] === slot || this.srcTo[e] === slot)) this.invalidateSource(id);
    }
  }

  /** Define table entry e (0 active, 1 pending) as source `id` = lerp(slot from -> to), start = args[0], duration = args[1]. id -1 clears. */
  setSource(e: number, id: number, from: number, to: number): void {
    const old = this.srcId[e];
    if (old >= 0 && old !== id && this.srcId[1 - e] !== old) this.invalidateSource(old);
    this.srcId[e] = id;
    this.srcFrom[e] = from;
    this.srcTo[e] = to;
    this.srcStart[e] = this.args[0];
    this.srcDur[e] = this.args[1] > 0 ? this.args[1] : 1;
  }

  // ---- worker offload (main thread side) -----------------------------------------------------

  /** True when a worker has attached to the shared buffers and serves snapshot requests. */
  remoteReady(): boolean {
    return this.ctrl !== null && Atomics.load(this.ctrl, C_READY) === 1;
  }

  /** Accept a finished remote snapshot (if still wanted). Allocation-free, never blocks. */
  private pollRemote(): void {
    const ctrl = this.ctrl;
    if (ctrl === null || this.remoteSnap < 0 || Atomics.load(ctrl, C_STATE) !== S_DONE) return;
    const s = this.pool[this.remoteSnap];
    if (s.src === this.remoteSrc && s.j === this.remoteJ && this.entryOf(s.src) >= 0) {
      s.valid = true;
      s.lastUse = ++this.useCounter;
      this.computed++;
      this.remoteComputed++;
    }
    this.remoteSnap = -1;
    Atomics.store(ctrl, C_STATE, S_IDLE);
  }

  /** Ask the worker for snapshot (src, j) into a fresh pool entry. */
  private submitRemote(src: number, j: number): void {
    const ctrl = this.ctrl as Int32Array;
    const i = this.acquire(src, j);
    this.remoteSnap = i; this.remoteSrc = src; this.remoteJ = j;
    ctrl[C_POOL] = i;
    ctrl[C_ENTRY] = this.entryOf(src);
    (this.ctrlF as Float64Array)[F_J] = j;
    Atomics.store(ctrl, C_STATE, S_REQUESTED);
    this.ring();
  }

  private ring(): void {
    const ctrl = this.ctrl as Int32Array;
    Atomics.add(ctrl, C_BELL, 1);
    Atomics.notify(ctrl, C_BELL);
  }

  /**
   * Ask the worker to fill a whole position frame for time t = args[0], choppiness args[1], ahead of
   * time (the next tick). Only when both snapshots are cached and the worker is idle for prefill.
   * The next materialise() at exactly that time adopts it instead of filling blocks on this thread.
   */
  requestPrefill(src: number): void {
    const ctrl = this.ctrl;
    if (ctrl === null || this.cubic || !this.remoteReady()) return;
    if (this.preBusy) {
      if (Atomics.load(ctrl, C_PSTATE) === S_REQUESTED) return; // still working on an older one
      this.preBusy = false; // done but not adopted (stale): drop it
      Atomics.store(ctrl, C_PSTATE, S_IDLE);
    }
    const t = this.args[0], chop = this.args[1], dt = this.snapshotDt;
    const j = Math.floor(t / dt);
    const ia = this.find(src, j), ib = this.find(src, j + 1);
    if (ia < 0 || ib < 0) return;
    const u = (t - j * dt) / dt, u2 = u * u, u3 = u2 * u;
    const f = this.ctrlF as Float64Array;
    f[F_PT] = t;
    // Same expressions as materialise() (bit-identical coefficients).
    f[F_PCOEF] = 2 * u3 - 3 * u2 + 1; f[F_PCOEF + 1] = (u3 - 2 * u2 + u) * dt; f[F_PCOEF + 2] = -2 * u3 + 3 * u2; f[F_PCOEF + 3] = (u3 - u2) * dt;
    f[F_PCOEF + 4] = chop;
    ctrl[C_PA] = ia; ctrl[C_PB] = ib; ctrl[C_PBUF] = this.preBuf;
    this.preA = ia; this.preB = ib; this.preBusy = true;
    Atomics.store(ctrl, C_PSTATE, S_REQUESTED);
    this.ring();
  }

  /**
   * Ask the worker to regenerate this cascade's spectrum into slot (master h0 + this sim's slot
   * grid) with params written by setRegenParams. Poll regenDone(); then call slotChanged(slot).
   */
  requestRegen(slot: number, params: Float64Array): boolean {
    const ctrl = this.ctrl;
    if (ctrl === null || !this.remoteReady() || Atomics.load(ctrl, C_RSTATE) === S_REQUESTED) return false;
    const f = this.ctrlF as Float64Array;
    for (let i = 0; i < 8; i++) f[F_REGEN + i] = params[i];
    ctrl[C_RSLOT] = slot;
    Atomics.store(ctrl, C_RSTATE, S_REQUESTED);
    this.ring();
    return true;
  }
  regenDone(): boolean { return this.ctrl !== null && Atomics.load(this.ctrl, C_RSTATE) === S_DONE; }
  ackRegen(): void { if (this.ctrl !== null) Atomics.store(this.ctrl, C_RSTATE, S_IDLE); }
  /** A slot grid was rewritten by the worker: drop snapshots of sources that use it. */
  slotChanged(slot: number): void {
    for (let e = 0; e < 2; e++) {
      const id = this.srcId[e];
      if (id >= 0 && (this.srcFrom[e] === slot || this.srcTo[e] === slot)) this.invalidateSource(id);
    }
  }
  /** Worker side: resample a master h0 into this sim's slot grid (no invalidation: main does that). */
  loadSlot(slot: number, masterH0: Float32Array): void { resampleCentred(masterH0, this.slotH0[slot], this.n, 4, true); }
  /** Worker side: the control block (for extra work served between requests). */
  get control(): { ctrl: Int32Array; ctrlF: Float64Array } { return { ctrl: this.ctrl as Int32Array, ctrlF: this.ctrlF as Float64Array }; }

  /** Ask a serving worker to exit. */
  stopRemote(): void {
    if (this.ctrl === null) return;
    Atomics.store(this.ctrl, C_QUIT, 1);
    this.ring();
  }

  // ---- worker side ---------------------------------------------------------------------------

  /**
   * Worker loop: wait for a request, compute the snapshot into the shared pool entry, signal done.
   * Never returns (until quit). Blocks with Atomics.wait (allowed in workers only).
   */
  serve(extra?: () => boolean): void {
    const ctrl = this.ctrl as Int32Array, ctrlF = this.ctrlF as Float64Array;
    Atomics.store(ctrl, C_READY, 1);
    for (;;) {
      const bell = Atomics.load(ctrl, C_BELL);
      if (Atomics.load(ctrl, C_QUIT) === 1) return;
      let worked = false;
      if (Atomics.load(ctrl, C_PSTATE) === S_REQUESTED) { // prefill first: short and needed next tick
        const out = this.frameBufs[ctrl[C_PBUF]], A = this.pool[ctrl[C_PA]].data, B = this.pool[ctrl[C_PB]].data;
        for (let by = 0; by < this.nb; by++) for (let bx = 0; bx < this.nb; bx++) fillPositions(out, A, B, ctrlF, F_PCOEF, F_PCOEF + 4, this.n, this.blockShift, bx, by);
        Atomics.store(ctrl, C_PSTATE, S_DONE);
        worked = true;
      }
      if (Atomics.load(ctrl, C_STATE) === S_REQUESTED) {
        const i = ctrl[C_POOL], e = ctrl[C_ENTRY], j = ctrlF[F_J];
        this.computeInto(this.pool[i].data, e, j);
        Atomics.store(ctrl, C_STATE, S_DONE);
        worked = true;
      }
      if (extra && extra()) worked = true; // one chunk of background work per round (e.g. spectrum regeneration)
      if (!worked) Atomics.wait(ctrl, C_BELL, bell);
    }
  }

  /** Full snapshot of source table entry e at time index j into out (same steps as the job). */
  private computeInto(out: Float32Array, e: number, j: number): void {
    const n = this.n, S = this.colSteps;
    this.evolveRows(e, j, 0, (n >> 1) + 1);
    for (let a = 0; a < 3; a++) {
      const re = a === 0 ? this.re1 : a === 1 ? this.re2 : this.re3;
      const im = a === 0 ? this.im1 : a === 1 ? this.im2 : this.im3;
      for (let k = 0; k < S; k++) fftColumnsStep(this.plan, re, im, k, 1);
      transposeSquare(n, re, im);
      for (let k = 0; k < S; k++) fftColumnsStep(this.plan, re, im, k, 1);
      transposeSquare(n, re, im);
    }
    this.packRows(out, 0, n);
  }

  private invalidateSource(id: number): void {
    if (this.jobSnap >= 0 && this.pool[this.jobSnap].src === id) this.jobSnap = -1;
    for (let i = 0; i < this.pool.length; i++) if (this.pool[i].src === id) { this.pool[i].valid = false; this.pool[i].src = -1; }
    for (let f = 0; f < 2; f++) {
      const a = this.lazyRef[2 * f];
      if (a >= 0 && this.pool[a].src === -1) { this.frameStamp[f] = -1; this.lazyRef[2 * f] = this.lazyRef[2 * f + 1] = -1; }
    }
  }

  private entryOf(id: number): number {
    return this.srcId[0] === id ? 0 : this.srcId[1] === id ? 1 : -1;
  }

  private find(src: number, j: number): number {
    for (let i = 0; i < this.pool.length; i++) {
      const s = this.pool[i];
      if (s.valid && s.src === src && s.j === j) return i;
    }
    return -1;
  }

  has(src: number, j: number): boolean { return this.find(src, j) >= 0; }

  private isPinned(s: Snapshot): boolean {
    for (let p = 0; p < this.pinCount; p++) if (this.pinSrc[p] === s.src && this.pinJ[p] === s.j) return true;
    return false;
  }

  private isLazyRef(i: number): boolean {
    const r = this.lazyRef;
    return r[0] === i || r[1] === i || r[2] === i || r[3] === i || (this.preBusy && (this.preA === i || this.preB === i));
  }

  private acquire(src: number, j: number): number {
    let best = -1, bestUse = Infinity;
    for (let i = 0; i < this.pool.length; i++) {
      const s = this.pool[i];
      if (i === this.jobSnap || i === this.remoteSnap) continue;
      if (!s.valid || this.entryOf(s.src) < 0) { best = i; break; }
      if (this.isPinned(s) || this.isLazyRef(i)) continue;
      if (s.lastUse < bestUse) { bestUse = s.lastUse; best = i; }
    }
    if (best < 0) { // everything pinned: take the least recently used anyway
      for (let i = 0; i < this.pool.length; i++) {
        if (i !== this.jobSnap && i !== this.remoteSnap && this.pool[i].lastUse < bestUse) { bestUse = this.pool[i].lastUse; best = i; }
      }
    }
    // A lazy frame still reading this entry becomes invalid (recomputed on demand).
    for (let f = 0; f < 2; f++) {
      if (this.lazyRef[2 * f] === best || this.lazyRef[2 * f + 1] === best) {
        this.frameStamp[f] = -1;
        this.lazyRef[2 * f] = this.lazyRef[2 * f + 1] = -1;
      }
    }
    const s = this.pool[best];
    s.src = src; s.j = j; s.valid = false; s.lastUse = ++this.useCounter;
    return best;
  }

  /** Make snapshot (src, j) available now (computes it synchronously if needed). Returns its pool index. */
  ensure(src: number, j: number): number {
    let i = this.find(src, j);
    if (i >= 0) { this.pool[i].lastUse = ++this.useCounter; return i; }
    if (this.entryOf(src) < 0) throw new Error(`CascadeSim: unknown source ${src}`);
    if (this.remoteSnap >= 0) {
      this.pollRemote();
      if (this.remoteSnap >= 0 && this.remoteSrc === src && this.remoteJ === j) {
        // The worker is computing exactly this one: wait for it (spin: Atomics.wait is not allowed
        // on a browser main thread). Rare: the prefetch runs two snapshot intervals ahead.
        const ctrl = this.ctrl as Int32Array;
        while (Atomics.load(ctrl, C_STATE) !== S_DONE) { /* spin */ }
        this.pollRemote();
        i = this.find(src, j);
        if (i >= 0) { this.pool[i].lastUse = ++this.useCounter; this.syncComputed++; return i; }
      }
    }
    if (this.jobSnap >= 0) {
      const js = this.pool[this.jobSnap];
      if (js.src === src && js.j === j) {
        i = this.jobSnap;
        this.runJob(1e9);
        this.pool[i].lastUse = ++this.useCounter;
        return i;
      }
      this.jobSnap = -1; // abandon the background job, it restarts later
    }
    i = this.acquire(src, j);
    this.startJob(i);
    this.runJob(1e9);
    this.syncComputed++;
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
    const n = this.n;
    while (this.jobSnap >= 0 && used < budget) {
      const snap = this.pool[this.jobSnap];
      const e = this.entryOf(snap.src);
      if (e < 0) { this.jobSnap = -1; break; } // source dropped
      if (this.jobPhase === 0) { // evolve (pair-rows 0 .. n/2, each covering two rows)
        const pairRows = (n >> 1) + 1;
        const r0 = this.jobProgress, r1 = Math.min(pairRows, r0 + Math.max(1, this.rowsChunk >> 1));
        this.evolveRows(e, snap.j, r0, r1);
        this.jobProgress = r1;
        if (r1 >= pairRows) { this.jobPhase = 1; this.jobProgress = 0; }
      } else if (this.jobPhase === 1) { // 2-D FFT of the 3 arrays: column pass, transpose, column pass, transpose
        const S = this.colSteps, per = 2 * S + 2;
        const a = (this.jobProgress / per) | 0, k = this.jobProgress - a * per;
        const re = a === 0 ? this.re1 : a === 1 ? this.re2 : this.re3;
        const im = a === 0 ? this.im1 : a === 1 ? this.im2 : this.im3;
        if (k < S) fftColumnsStep(this.plan, re, im, k, 1);
        else if (k === S || k === 2 * S + 1) transposeSquare(n, re, im);
        else fftColumnsStep(this.plan, re, im, k - S - 1, 1);
        this.jobProgress++;
        if (this.jobProgress >= 3 * per) { this.jobPhase = 2; this.jobProgress = 0; }
      } else { // pack
        const r0 = this.jobProgress, r1 = Math.min(n, r0 + this.rowsChunk);
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

  /**
   * Evolve pair-rows [r0, r1) (r in 0 .. n/2) of source table entry e at t = j * snapshotDt into the
   * 3 packed spectra. Row r and its mirror row (n - r) % n are done together: cell k and cell -k share
   * |k|, hence omega and the phase, so each cos / sin pair serves two cells.
   */
  private evolveRows(e: number, j: number, r0: number, r1: number): void {
    const n = this.n, t = j * this.snapshotDt;
    // Crossfade weight and its time derivative at t (w = 1 for a single-slot source).
    let w = 1, dw = 0;
    if (this.srcFrom[e] !== this.srcTo[e]) {
      const dur = this.srcDur[e], x = (t - this.srcStart[e]) / dur;
      if (x <= 0) w = 0;
      else if (x < 1) { w = x * x * (3 - 2 * x); dw = (6 * x * (1 - x)) / dur; }
    }
    const A = this.slotH0[this.srcFrom[e]], B = this.slotH0[this.srcTo[e]];
    const om = this.omegaIndex, kxn = this.kxn, kzn = this.kzn;
    const re1 = this.re1, im1 = this.im1, re2 = this.re2, im2 = this.im2, re3 = this.re3, im3 = this.im3;
    let tm = t % LOOP_PERIOD;
    if (tm < 0) tm += LOOP_PERIOD;
    for (let y = r0; y < r1; y++) {
      const ym = (n - y) & (n - 1);
      const self = ym === y; // rows 0 and n/2 are their own mirror: do half of them
      const xEnd = self ? (n >> 1) + 1 : n;
      for (let x = 0; x < xEnd; x++) {
        const i0 = y * n + x;
        const im = ym * n + ((n - x) & (n - 1));
        const m = om[i0];
        const omega = m * OMEGA_UNIT;
        const phase = OMEGA_UNIT * ((m * tm) % LOOP_PERIOD);
        const c = Math.cos(phase), s = Math.sin(phase);
        // The cell and its mirror (same |k|, same omega): same body, inlined (no call: doubles stay unboxed).
        for (let pass = 0; pass < 2; pass++) {
          const i = pass === 0 ? i0 : im;
          if (pass === 1 && im === i0) break;
          const q = 4 * i;
          // h0 = lerp(A, B, w) (as the GPU); dh0/dt = dw (B - A)
          const Ar = A[q], Ai = A[q + 1], Amr = A[q + 2], Ami = A[q + 3];
          const Dr = B[q] - Ar, Di = B[q + 1] - Ai, Dmr = B[q + 2] - Amr, Dmi = B[q + 3] - Ami;
          const hr = Ar + w * Dr, hi = Ai + w * Di, mr = Amr + w * Dmr, mi = Ami + w * Dmi;
          // h0 * e^{-i w t}, h0m * e^{+i w t}
          const ar = hr * c + hi * s, ai = hi * c - hr * s;
          const br = mr * c - mi * s, bi = mi * c + mr * s;
          const htr = ar + br, hti = ai + bi;
          // dht = omega * (-i a + i b) + dh0 e^{-i w t} + dh0m e^{+i w t}
          let dtr = omega * (ai - bi), dti = omega * (br - ar);
          if (dw !== 0) {
            dtr += dw * (Dr * c + Di * s + Dmr * c - Dmi * s);
            dti += dw * (Di * c - Dr * s + Dmi * c + Dmr * s);
          }
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
   * Per-tick scheduling at tick time T = args[0] (tick length args[1]): snapshots j0 = floor(T/dt)
   * and j0+1 of the active source are made available (synchronously only if the prefetch fell
   * behind), then the background job advances at a steady rate towards j0+2 and j0+3 and the
   * pending source's snapshots around its switch time args[2] (spread over the ticks until then).
   * Snapshot j0-1 stays pinned so queries at the interpolated render time never stall.
   * Allocation-free.
   */
  schedule(active: number, pending: number): void {
    const T = this.args[0], tickDt = Math.max(this.args[1], 1e-4), switchTime = this.args[2];
    const dt = this.snapshotDt;
    const j0 = Math.floor(T / dt);
    const jp = Math.floor(Math.max(T, switchTime) / dt);
    this.pinCount = 0;
    this.pin(active, j0 - 1); this.pin(active, j0); this.pin(active, j0 + 1); this.pin(active, j0 + 2); this.pin(active, j0 + 3);
    if (pending >= 0) { this.pin(pending, jp); this.pin(pending, jp + 1); this.pin(pending, jp + 2); }
    this.pollRemote();
    this.ensure(active, j0); this.ensure(active, j0 + 1);
    if (this.remoteReady()) {
      // Background snapshots go to the worker: one request in flight, picked in priority order.
      if (this.remoteSnap < 0) {
        if (!this.has(active, j0 + 2)) this.submitRemote(active, j0 + 2);
        else if (pending >= 0 && !this.has(pending, jp)) this.submitRemote(pending, jp);
        else if (pending >= 0 && !this.has(pending, jp + 1)) this.submitRemote(pending, jp + 1);
        else if (pending >= 0 && !this.has(pending, jp + 2)) this.submitRemote(pending, jp + 2);
        else if (!this.has(active, j0 + 3)) this.submitRemote(active, j0 + 3);
      }
      return;
    }
    // Steady rate: one active snapshot per interval, plus the pending source's missing snapshots
    // spread over the ticks left until its switch. Catch up at double rate when behind.
    let rate = (this.unitsPerSnapshot * tickDt) / dt;
    if (!this.has(active, j0 + 2)) rate *= 2;
    if (pending >= 0) {
      let missing = 0;
      if (!this.has(pending, jp)) missing++;
      if (!this.has(pending, jp + 1)) missing++;
      if (!this.has(pending, jp + 2)) missing++;
      rate += (this.unitsPerSnapshot * missing) / Math.max(1, (switchTime - T) / tickDt);
    }
    let budget = Math.ceil(rate * RATE_MARGIN);
    while (budget > 0) {
      if (this.jobSnap < 0) {
        let si = -1, sj = 0;
        if (!this.has(active, j0 + 2)) { si = active; sj = j0 + 2; }
        else if (pending >= 0 && !this.has(pending, jp)) { si = pending; sj = jp; }
        else if (pending >= 0 && !this.has(pending, jp + 1)) { si = pending; sj = jp + 1; }
        else if (pending >= 0 && !this.has(pending, jp + 2)) { si = pending; sj = jp + 2; }
        else if (!this.has(active, j0 + 3)) { si = active; sj = j0 + 3; }
        if (si < 0) break;
        this.startJob(this.acquire(si, sj));
      }
      budget -= this.runJob(budget);
    }
  }

  private pin(src: number, j: number): void {
    if (this.pinCount < MAX_PINS) { this.pinSrc[this.pinCount] = src; this.pinJ[this.pinCount] = j; this.pinCount++; }
  }

  /**
   * Materialise the field of source `src` at time t = args[0] with choppiness chop = args[1]
   * (d chop / dt = args[2]) into frame `frame`: records the Hermite coefficients and the two
   * snapshot references (computing the snapshots synchronously only if they are not cached).
   * Blocks are filled on demand by ensurePos / ensureVel. Allocation-free.
   */
  materialise(frame: number, stamp: number, src: number): void {
    const t = this.args[0], chop = this.args[1], dchop = this.args[2];
    const dt = this.snapshotDt;
    const j = Math.floor(t / dt);
    const u = (t - j * dt) / dt;
    const u2 = u * u, u3 = u2 * u;
    const ia = this.ensure(src, j); const pa = this.pool[ia]; pa.lastUse = PINNED_USE;
    const ib = this.ensure(src, j + 1); const pb = this.pool[ib];
    pa.lastUse = ++this.useCounter; pb.lastUse = ++this.useCounter;
    const c = frame * 10, k = this.lazyCoef;
    k[c] = 2 * u3 - 3 * u2 + 1; k[c + 1] = (u3 - 2 * u2 + u) * dt; k[c + 2] = -2 * u3 + 3 * u2; k[c + 3] = (u3 - u2) * dt;
    k[c + 4] = (6 * u2 - 6 * u) / dt; k[c + 5] = 3 * u2 - 4 * u + 1; k[c + 6] = (-6 * u2 + 6 * u) / dt; k[c + 7] = 3 * u2 - 2 * u;
    k[c + 8] = chop; k[c + 9] = dchop;
    this.lazyRef[2 * frame] = ia; this.lazyRef[2 * frame + 1] = ib;
    this.velValid[frame].fill(0);
    this.frameTime[frame] = t;
    this.frameStamp[frame] = stamp;
    // Adopt the worker's prefill for exactly this frame (same snapshots, time and choppiness).
    const ctrl = this.ctrl;
    if (this.preBusy && ctrl !== null) {
      const f = this.ctrlF as Float64Array;
      // Never wait: if the worker has not finished (it may be busy with a snapshot), fill lazily here.
      if (Atomics.load(ctrl, C_PSTATE) === S_DONE && f[F_PT] === t && this.preA === ia && this.preB === ib && f[F_PCOEF + 4] === chop) {
        const b = this.frameBuf[frame];
        this.frameBuf[frame] = this.preBuf; this.preBuf = b;
        this.frames[frame] = this.frameBufs[this.frameBuf[frame]];
        this.preBusy = false;
        Atomics.store(ctrl, C_PSTATE, S_IDLE);
        this.blockValid[frame].fill(1);
        this.prefillsAdopted++;
        return;
      }
    }
    this.blockValid[frame].fill(0);
  }

  /** Fill the position blocks covering texel columns x0..x1 and rows y0..y1 (wrapped, each span <= BLOCK). */
  ensurePos(f: number, x0: number, y0: number, x1: number, y1: number): void {
    const sh = this.blockShift, nb = this.nb, bv = this.blockValid[f], m = this.n - 1;
    const bx0 = (x0 & m) >> sh, bx1 = (x1 & m) >> sh, by0 = (y0 & m) >> sh, by1 = (y1 & m) >> sh;
    if (bv[by0 * nb + bx0] === 0) this.fillBlock(f, bx0, by0);
    if (bv[by0 * nb + bx1] === 0) this.fillBlock(f, bx1, by0);
    if (bv[by1 * nb + bx0] === 0) this.fillBlock(f, bx0, by1);
    if (bv[by1 * nb + bx1] === 0) this.fillBlock(f, bx1, by1);
  }

  /** Same for the velocity blocks. */
  ensureVel(f: number, x0: number, y0: number, x1: number, y1: number): void {
    const sh = this.blockShift, nb = this.nb, bv = this.velValid[f], m = this.n - 1;
    const bx0 = (x0 & m) >> sh, bx1 = (x1 & m) >> sh, by0 = (y0 & m) >> sh, by1 = (y1 & m) >> sh;
    if (bv[by0 * nb + bx0] === 0) this.fillVelBlock(f, bx0, by0);
    if (bv[by0 * nb + bx1] === 0) this.fillVelBlock(f, bx1, by0);
    if (bv[by1 * nb + bx0] === 0) this.fillVelBlock(f, bx0, by1);
    if (bv[by1 * nb + bx1] === 0) this.fillVelBlock(f, bx1, by1);
  }

  private fillBlock(f: number, bx: number, by: number): void {
    const c = f * 10;
    fillPositions(this.frames[f], this.pool[this.lazyRef[2 * f]].data, this.pool[this.lazyRef[2 * f + 1]].data, this.lazyCoef, c, c + 8, this.n, this.blockShift, bx, by);
    this.blockValid[f][by * this.nb + bx] = 1;
    this.posBlocksFilled++;
  }

  private fillVelBlock(f: number, bx: number, by: number): void {
    const n = this.n, B = 1 << this.blockShift, c = f * 10, k = this.lazyCoef;
    const A0 = this.pool[this.lazyRef[2 * f]].data, B0 = this.pool[this.lazyRef[2 * f + 1]].data;
    const out = this.velFrames[f];
    const h00 = k[c], h10 = k[c + 1], h01 = k[c + 2], h11 = k[c + 3];
    const d00 = k[c + 4], d10 = k[c + 5], d01 = k[c + 6], d11 = k[c + 7], chop = k[c + 8], dchop = k[c + 9];
    for (let y = by * B, yEnd = y + B; y < yEnd; y++) {
      for (let x = bx * B, xEnd = x + B; x < xEnd; x++) {
        const i = y * n + x, o = i * 6, q = i * 3;
        out[q] = d00 * A0[o] + d10 * A0[o + 3] + d01 * B0[o] + d11 * B0[o + 3];
        for (let ch = 1; ch < 3; ch++) {
          const a = A0[o + ch], av = A0[o + ch + 3], b = B0[o + ch], bv = B0[o + ch + 3];
          out[q + ch] = chop * (d00 * a + d10 * av + d01 * b + d11 * bv) + dchop * (h00 * a + h10 * av + h01 * b + h11 * bv);
        }
      }
    }
    this.velValid[f][by * this.nb + bx] = 1;
    this.velBlocksFilled++;
  }

  /** Snapshot data the frame's Hermite interpolation reads (A = time index j, B = j + 1). For direct sampling. */
  frameSnapA(f: number): Float32Array { return this.pool[this.lazyRef[2 * f]].data; }
  frameSnapB(f: number): Float32Array { return this.pool[this.lazyRef[2 * f + 1]].data; }

  /** Release the big buffers (the sim must not be used afterwards). */
  dispose(): void {
    this.pool.length = 0;
    this.slotH0.length = 0;
    this.frames.length = 0;
    this.velFrames.length = 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Field sampling (allocation-free) of a materialised frame (stride 3). Results go to SAMPLE[0..2].
// ---------------------------------------------------------------------------------------------
export const SAMPLE = new Float64Array(3);
/** Floats per texel of a materialised frame / velocity frame. */
export const FRAME_STRIDE = 3;


/** Gradient of the last sample*Grad: [dE/da, dE/db, dN/da, dN/db, dH/da, dH/db] (H only for cubic). */
export const GRAD = new Float64Array(6);


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
  let s0 = 0, s1 = 0, s2 = 0, ea = 0, eb = 0, na = 0, nb = 0, ha = 0, hb = 0;
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
    const hd = dx0 * data[i0] + dx1 * data[i1] + dx2 * data[i2] + dx3 * data[i3];
    s0 += wy * h; s1 += wy * e; s2 += wy * m;
    ea += wy * ed; na += wy * md; eb += dy * e; nb += dy * m; ha += wy * hd; hb += dy * h;
  }
  SAMPLE[0] = s0; SAMPLE[1] = s1; SAMPLE[2] = s2;
  GRAD[0] = ea * inv; GRAD[1] = eb * inv; GRAD[2] = na * inv; GRAD[3] = nb * inv; GRAD[4] = ha * inv; GRAD[5] = hb * inv;
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

