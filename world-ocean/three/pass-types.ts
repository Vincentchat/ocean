// Render-graph pass shape (spec §5). Every renderer.render / renderer.compute call lives inside a pass.
// In the engine these objects are registered with @dreamp/render-graph; in the side repo the demo's
// pass runner executes them in STAGES order.

/** Stages, in execution order (spec §5). */
export const STAGES = [
  'compute', 'shadow', 'scene', 'ao', 'lighting', 'atmosphere', 'volumetrics', 'transparent',
  'temporal', 'motionblur', 'dof', 'bloom', 'exposure', 'tonemap', 'ldr', 'overlay',
] as const;
export type Stage = (typeof STAGES)[number];

export interface PassDesc {
  /** 'ocean.fft', 'demo.scene', ... */
  readonly id: string;
  /** Owning module: 'world/ocean', 'demo'. */
  readonly module: string;
  readonly stage: Stage;
  /** Target names read, e.g. 'scene.color', 'ocean.displacement'. */
  readonly reads: readonly string[];
  /** Target names written. A pass whose writes nobody reads may be culled by the engine graph. */
  readonly writes: readonly string[];
}

/** Per-frame context handed to every pass. The same object is reused every frame (no allocation). */
export interface PassContext {
  /** three/webgpu WebGPURenderer (typed loosely to keep this file free of three imports). */
  renderer: any;
  /** Real seconds since the last frame. */
  dt: number;
  /** Frame counter. */
  frame: number;
  /** Drawing buffer size in pixels. */
  width: number;
  height: number;
  tier: 'low' | 'medium' | 'high' | 'ultra';
  /** Simulation time to DRAW at (FrameInfo.interpolatedTime), seconds. */
  time: number;
  /** Shared named targets ('scene.color' -> RenderTarget, ...). Owned by the host. */
  targets: Record<string, any>;
}

export interface Pass {
  readonly desc: PassDesc;
  /** Runs every frame. No allocation. */
  execute(ctx: PassContext): void;
  /** Optional: skip this frame when false. */
  enabled?: boolean;
}
