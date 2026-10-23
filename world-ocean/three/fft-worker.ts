// Web worker of the GPU FFT: tier spectrum generation + omega tables off the main thread (spec §6: heavy
// precompute may go to a worker). Pure maths from fft-tiers.ts / spectrum.ts; no three import.
//   in : { id, kind: 'spectrum', tier, params, seed, g }  -> out { id, h0: Float32Array[], unresolved }
//   in : { id, kind: 'omega', tier, g }                    -> out { id, omega: Float32Array }
import { FFT_TIERS, tierCascades, generateTierSpectrum, tierOmega, type OceanTier } from './fft-tiers.ts';
import { CASCADES } from '../spectrum.ts';

const refSlope = new Map<string, number>();

/** Handle one request (also used synchronously as the fallback when Worker is unavailable). */
export function handleFftJob(msg: any): { result: any; transfer: ArrayBuffer[] } {
  const tier = msg.tier as OceanTier;
  const spec = FFT_TIERS[tier];
  if (msg.kind === 'omega') {
    const omega = tierOmega(spec, tierCascades(tier), msg.g);
    return { result: { id: msg.id, omega }, transfer: [omega.buffer as ArrayBuffer] };
  }
  const sp = generateTierSpectrum(spec, tierCascades(tier), msg.params, msg.seed, msg.g);
  // Reference slope variance of the same parameters (the full spectrum the reference draws).
  const key = JSON.stringify([msg.params, msg.seed, msg.g]);
  let ref = refSlope.get(key);
  if (ref === undefined) {
    ref = spec.reference ? sp.slopeVariance : generateTierSpectrum(FFT_TIERS.medium, CASCADES, msg.params, msg.seed, msg.g).slopeVariance;
    if (refSlope.size > 16) refSlope.clear();
    refSlope.set(key, ref);
  }
  const unresolved = Math.max(0, ref - sp.slopeVariance);
  return { result: { id: msg.id, h0: sp.h0, unresolved }, transfer: sp.h0.map((a) => a.buffer as ArrayBuffer) };
}

const scope: any = typeof self !== 'undefined' ? self : null;
if (scope && typeof scope.postMessage === 'function' && typeof (scope as any).document === 'undefined') {
  scope.onmessage = (e: MessageEvent) => {
    try {
      const { result, transfer } = handleFftJob(e.data);
      scope.postMessage(result, transfer);
    } catch (err) {
      scope.postMessage({ id: e.data?.id, error: String((err as Error)?.stack ?? err) });
    }
  };
}
