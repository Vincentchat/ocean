// Asset loader hook: the module refers to binary assets by LOGICAL ID only, never by path.
// On import into the engine these become an asset pack (assets/<pack>/pack.json) and the engine
// installs its own resolver. No three.js import here (render-agnostic).

/** Logical asset ids used by the ocean module (and, prefixed `demo/`, by the standalone demo only). */
export const OCEAN_ASSETS = {
  /** RGBA8 tileable foam breakup pattern (R: breaking lace, G: surface foam). 2048^2 PNG, MIT (nagi-ocean-sim). */
  foamPattern: 'world/ocean/foam-pattern',
} as const;

/** Demo-only assets (photo sky stand-in). Not part of the module's asset pack. */
export const DEMO_ASSETS = {
  /** 4096x2048 equirect sky, SDR base codes (sRGB-ish JPEG). */
  skyBase: 'demo/sky-base',
  /** 4096x2048 equirect sky, log gain map (JPEG). */
  skyGain: 'demo/sky-gain',
  /** 1024x512 RGBA half-float cloud radiance (raw little-endian, 4 MiB). */
  skyRadiance: 'demo/sky-radiance',
} as const;

/** Maps a logical id to a fetchable URL. Throw (or return null) for unknown ids. */
export type AssetResolver = (id: string) => string | null;

let resolver: AssetResolver | null = null;

/** Installed once by the host (engine asset system, or the demo). */
export function setAssetResolver(fn: AssetResolver | null): void {
  resolver = fn;
}

/** URL for a logical asset id. Throws when no resolver is installed or the id is unknown. */
export function resolveAsset(id: string): string {
  if (!resolver) throw new Error(`No asset resolver installed (asset "${id}")`);
  const url = resolver(id);
  if (!url) throw new Error(`Unknown asset id "${id}"`);
  return url;
}
