// Web worker of the GPU FFT (spec §6: heavy precompute off the main thread). Pure maths from fft-tiers.ts.
//   in : { id, kind: 'capillary', params, seed, g, n }  -> out { id, h0, omega }   (ultra's opt-in band)
//   in : { id, kind: 'dropped', h0: Float32Array (c2 master), n, domain }  -> out { id, slope }
import { generateCapillary, droppedSlopeVariance } from './fft-tiers.ts';

/** Handle one request (also used synchronously as the fallback when Worker is unavailable). */
export function handleFftJob(msg: any): { result: any; transfer: ArrayBuffer[] } {
  if (msg.kind === 'capillary') {
    const r = generateCapillary(msg.params, msg.seed, msg.g, msg.n);
    return { result: { id: msg.id, h0: r.h0, omega: r.omega }, transfer: [r.h0.buffer as ArrayBuffer, r.omega.buffer as ArrayBuffer] };
  }
  if (msg.kind === 'dropped') {
    return { result: { id: msg.id, slope: droppedSlopeVariance(msg.h0, msg.n, msg.domain) }, transfer: [] };
  }
  throw new Error(`fft-worker: unknown job ${msg.kind}`);
}

const scope: any = typeof self !== 'undefined' ? self : null;
if (scope && typeof scope.postMessage === 'function' && typeof scope.document === 'undefined') {
  scope.onmessage = (e: MessageEvent) => {
    try {
      const { result, transfer } = handleFftJob(e.data);
      scope.postMessage(result, transfer);
    } catch (err) {
      scope.postMessage({ id: e.data?.id, error: String((err as Error)?.stack ?? err) });
    }
  };
}
