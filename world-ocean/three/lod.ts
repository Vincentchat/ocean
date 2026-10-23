// Ocean mesh LOD: camera-centred geometry clipmap, snapped per level, geomorphed, drawn camera-relative.
// Isolated so a future world/lod-tree can replace it: the material only talks to `lod.nodes` (+ lod.getFoot).
//
// API
//   const lod = createOceanLod({ tier: 'high', shape, origin, maxTier?, coverage?, cullMargin? });
//   waterScene.add(lod.mesh);                     // the grid: ONE draw (or a few frustum-culled ranges, see setMaterial)
//   lod.setMaterial(material);                    // = mesh.material + per-sector frustum culling (recommended)
//   material.positionNode = lod.nodes.localPos.add(displacementLocal);   // REQUIRED (see below)
//   material.vertexNode   = lod.nodes.clipPosition;                       // camera-relative (highp) projection
//   mainScene.add(lod.enableClosure(horizonMaterial, { reversedDepth })); // horizon closure as a BACKGROUND draw
//   per frame (after the camera moved, before rendering):  lod.update(camera);   // allocation-free; reads the
//                                                camera's matrixWorld, projection (culling) and far (ring count)
//   lod.cullFor(otherCamera)                      // re-cull for another camera rendering the grid (reflection...)
//   lod.setTier('low');   // non-blocking: built in 2 ms slices, uploaded in 1 MB slices into the idle half of a
//                         // double-buffered geometry, old tier drawn until the swap (a drawRange change)
//   lod.getFoot(out)      // camera foot in world lattice coordinates (float64 m, east/south); false before 1st update
//   lod.dispose();
//
// lod.nodes (TSL, vertex stage; wrap in varying(...) to use per fragment):
//   localPos          vec3  undisplaced vertex position in the mesh's LOCAL frame, metres, float32, small:
//                           x = east, y = up, z = south (= -north), origin = the camera's foot point at sea level.
//                           The mesh is placed at the foot (render space) and rotated by the shape's tangent
//                           frame there, so "up" is local +Y for every shape — displace along local +Y.
//                           (On the flat shape local == world axes.)
//   previousLocalPos  vec3  where the same world point was in LAST frame's local frame (undisplaced), using last
//                           frame's foot AND last frame's geomorph; the horizon closure is camera-attached (= localPos).
//                           Add last frame's displacement and assign it to three's `positionPrevious` (VelocityNode
//                           multiplies it by the object's previous world matrix).
//   clipPosition      vec4  cameraProjectionMatrix * highpModelViewMatrix * positionLocal: the model-view product is
//                           done on the CPU in float64 per object and camera, so the geometry does not depend on how far
//                           the render-space camera is from the render origin (three's default multiplies the two
//                           matrices in float32 on the GPU: ~0.5 m error at 1e7 m).
//   closureClipPosition vec4 same, with z forced onto the far plane (z = w, or 0 with reversed depth): see enableClosure.
//   tileCoord(P)      vec2  periodic sampling coordinate for an FFT tile of period P metres, in tangent axes
//                           (east, south) = (world x, world z) on the flat shape: (world mod P) + small lattice
//                           offset. Built as anchor_P + q, anchor_P = (B_render + origin.wrapped(P)) mod P on the
//                           CPU in float64, q = the vertex's exact float32 lattice offset from B (B = lattice
//                           anchor near the camera). Never a raw world position on the GPU. uv = tileCoord(P) / P
//                           with repeat wrapping. Values are stable to < 1e-4 m when the camera or origin moves.
//                           Creates its uniform on first call: call it at material-build time, never per frame.
//   distanceToCamera  float 3D distance camera -> undisplaced vertex, metres (displacement fade, LOD).
//   morphFactor       float 0..1 geomorph weight towards the next coarser level.
//   cellSize          float local grid spacing (m) including the morph (footprint for explicit texture LOD).
//   isSkirt           float 1 on the horizon closure vertices, 0 on the grid.
//   level             float level index (0 = finest), -1 on the closure (debug views).
//   cameraLocal       vec3  LOD camera position in the local frame = (0, altitude, 0).
//
// Horizon closure. The rings end at ~2.4-3 km (or 0.6 far). Beyond them, up to the true horizon (eye level), the
// closure covers the rest: a sea-level annulus from the last ring outwards plus a strip up to eye level near the far
// plane. With enableClosure() it is a SEPARATE mesh whose clip z is exactly the far plane and which writes no depth:
// drawn first (renderOrder -1e9, opaque, in the main scene) it behaves like a background — every other surface drawn
// later wins the depth test, whatever the depth precision (LessEqual against a buffer cleared to far), so it can never
// hide terrain, ships, islands. Other cameras (reflection, probes) see it at far depth too: at worst a wrong-looking
// band near their horizon, never an occluder; it casts no shadow. Without enableClosure() the closure is drawn inline
// with the grid (old behaviour: near the far plane, can tie with terrain beyond ~0.985 far).
//
// Geometry: the 'position' attribute holds PACKED GRID DATA (gx, gy, level*4+kind), not positions — a material
// without `positionNode = lod.nodes.localPos ...` draws garbage. See three/lod-math.ts for the clipmap model
// (snapping, morph, clamp, bit-exact level boundaries) and tests/lod.test.ts for its proofs.
//
// Floating origin: update() reads the camera's render position (camera.matrixWorld), converts to world with
// origin.toWorld (float64), and recomputes everything from world lattice coordinates, so an origin shift changes
// nothing on screen. origin.onShift re-places the mesh and the wrapped anchors immediately (no one-frame jump).
//
// Up from the shape: foot = shape.atAltitude(camera, 0), frame = shape.tangentFrame(foot). Nothing assumes +Y.
// Sphere (DESIGN LIMIT, not just 'curvature missing'): the snapping lattice lives in the tangent frame at the camera
// foot. On a sphere that frame rotates as the camera moves, so vertices would swim; a sphere needs the lattice anchored
// to a fixed per-region frame (or a cube-sphere quadtree) plus curvature. tileAnchor's non-axis-aligned path wraps the
// float64 lattice coordinate (equivalent value) instead of origin.wrapped.
import { BufferAttribute, BufferGeometry, Matrix4, Mesh, Sphere, Vector2, Vector3, Vector4 } from 'three/webgpu';
import {
  abs, cameraFar, cameraProjectionMatrix, cameraViewMatrix, clamp, float, floor, highpModelViewMatrix, int, length,
  max, min, modelWorldMatrix, normalize, positionGeometry, positionLocal, select, uniform, uniformArray, vec2, vec3,
  vec4,
} from 'three/tsl';
import type { FloatingOrigin, TangentFrame, Vec3, WorldShape } from '../stubs/world-core.ts';
import {
  LOD_SECTORS, MAX_LOD_LEVELS, buildLodGeometry, createLodState, frustumSidePlanes, lodCullRanges,
  lodDrawCount, lodGeometryCounts, lodGeometrySteps, lodParams, tileAnchor, updateLodState,
  type LodGeometryData, type LodParams, type LodState, type LodTier,
} from './lod-math.ts';

export type { LodTier } from './lod-math.ts';

const TIER_ORDER: LodTier[] = ['low', 'medium', 'high', 'ultra'];

export interface OceanLodOptions {
  tier: LodTier;
  shape: WorldShape;
  origin: FloatingOrigin;
  /** Material for the grid (its positionNode must be built from lod.nodes.localPos). Can be set later. */
  material?: any;
  /** Rings reach at least this distance (m) before the closure takes over. Default 2400 (reference mesh radius). */
  coverage?: number;
  /**
   * Largest tier setTier() may switch to without reallocating (default 'ultra'). The geometry is double-buffered
   * at this tier's size: GPU+CPU memory = 2 x (12 B x vertices + 12 B x triangles) of that tier (ultra ~50 MB,
   * high ~28 MB). Switching above it reallocates (one-frame hitch).
   */
  maxTier?: LodTier;
  /** Frustum-cull safety margin, m (max horizontal + vertical displacement). Default 40. */
  cullMargin?: number;
}

export interface OceanLodNodes {
  readonly localPos: any;
  readonly previousLocalPos: any;
  readonly clipPosition: any;
  readonly closureClipPosition: any;
  tileCoord(period: number): any;
  readonly distanceToCamera: any;
  readonly morphFactor: any;
  readonly cellSize: any;
  readonly isSkirt: any;
  /** Debug: level index of the vertex (float), -1 on the closure. */
  readonly level: any;
  /** Camera position in the mesh's local frame: (0, altitude, 0). */
  readonly cameraLocal: any;
}

export interface OceanLodStats {
  tier: LodTier;
  vertices: number;
  triangles: number;
  /** Grid triangles in the draw range (all drawn levels) and after the sector frustum cull. */
  drawnTriangles: number;
  culledDrawTriangles: number;
  /** Draw ranges issued for the grid (1 without setMaterial). */
  drawRanges: number;
  drawnLevels: number;
  /** Finest drawn spacing, m. */
  spacing0: number;
  /** Max-norm reach of the last drawn ring, m (closure beyond). */
  ringReach: number;
  /** Tier being built / uploaded in the background (old tier still drawn), or null. */
  pendingTier: LodTier | null;
  closureEnabled: boolean;
}

export interface OceanLod {
  /** The grid mesh. */
  readonly mesh: any;
  /** The horizon-closure mesh (null until enableClosure). */
  readonly closure: any;
  readonly nodes: OceanLodNodes;
  readonly params: LodParams;
  /** Internal clipmap state (debug / parity only; use getFoot for the foot). */
  readonly state: Readonly<LodState>;
  /** Allocation-free. Call once per frame after the camera's matrixWorld is current. */
  update(camera: any, origin?: FloatingOrigin): void;
  /** Sets the grid material as a 1-element array so the per-sector frustum cull (geometry groups) applies. */
  setMaterial(material: any): void;
  /** Re-runs the sector cull for another camera rendering the grid (allocation-free). */
  cullFor(camera: any): void;
  /** Frustum culling on/off (default on; only effective after setMaterial). */
  culling: boolean;
  /**
   * Switches the horizon closure to a separate background mesh (see header): configures `material` (positionNode,
   * vertexNode, depthWrite false, opaque) and returns the mesh to add to the MAIN scene. The material must not read
   * scene.color / scene.depth (it is drawn inside the scene pass): a haze / horizon colour material.
   */
  enableClosure(material: any, opts?: { reversedDepth?: boolean }): any;
  /** Camera foot in world lattice coordinates (float64 metres, u = east, v = south). Returns false before update. */
  getFoot(out: { u: number; v: number }): boolean;
  /** Starts building the tier in time slices inside update(); the old tier stays drawn until the swap. */
  setTier(tier: LodTier): void;
  readonly pendingTier: LodTier | null;
  /** Fills `out` (or an internal object reused across calls). Allocation-free. */
  stats(out?: OceanLodStats): OceanLodStats;
  /** Debug/test hooks: offset the LOD's notion of the camera foot (m, east/south), disable snapping. */
  readonly debug: { footOffsetU: number; footOffsetV: number; noSnap: boolean };
  dispose(): void;
}

export function createOceanLod(opts: OceanLodOptions): OceanLod {
  const { shape } = opts;
  let origin = opts.origin;
  let params = lodParams(opts.tier, { coverage: opts.coverage });
  const cullMargin = opts.cullMargin ?? 40;
  const state = createLodState();
  const debug = { footOffsetU: 0, footOffsetV: 0, noSnap: false };

  // ------------------------------------------------------------------------------------ double-buffered geometry
  // Two halves of capacity (VCAP vertices, ICAP indices) in ONE pair of attributes: the active half is drawn, the idle
  // half receives a new tier in 1 MB update ranges; the swap is a drawRange change. Uploaded once at start.
  const maxTierIdx = Math.max(TIER_ORDER.indexOf(opts.maxTier ?? 'ultra'), TIER_ORDER.indexOf(params.tier));
  let VCAP = 0, ICAP = 0;
  function capacityFor(tierIdx: number) {
    let v = 0, i = 0;
    for (let t = 0; t <= tierIdx; t++) {
      const c = lodGeometryCounts(lodParams(TIER_ORDER[t], { coverage: opts.coverage }));
      v = Math.max(v, c.vertexCount); i = Math.max(i, c.triangleCount * 3);
    }
    return { v, i };
  }
  let posAttr: BufferAttribute, idxAttr: BufferAttribute;
  const gridGeom = new BufferGeometry();
  const closureGeom = new BufferGeometry();
  gridGeom.name = 'ocean.lod'; closureGeom.name = 'ocean.lod.closure';
  // Positions are packed grid data: give a huge bounding volume so nothing culls / computes from them.
  gridGeom.boundingSphere = new Sphere(new Vector3(), 1e9);
  closureGeom.boundingSphere = new Sphere(new Vector3(), 1e9);
  function allocate(tierIdx: number) {
    const c = capacityFor(tierIdx);
    VCAP = c.v; ICAP = c.i;
    posAttr = new BufferAttribute(new Float32Array(2 * VCAP * 3), 3);
    idxAttr = new BufferAttribute(new Uint32Array(2 * ICAP), 1);
    gridGeom.setAttribute('position', posAttr); gridGeom.setIndex(idxAttr);
    closureGeom.setAttribute('position', posAttr); closureGeom.setIndex(idxAttr);
  }
  /** Copies data into half h (index values offset by the half's vertex base). Returns elements written. */
  function writeHalf(h: number, d: LodGeometryData, which: 0 | 1, from: number, count: number) {
    if (which === 0) {
      const src = d.packed, dst = posAttr.array as Float32Array, off = h * VCAP * 3;
      const n = Math.min(count, src.length - from);
      dst.set(src.subarray(from, from + n), off + from);
      return n;
    }
    const src = d.index, dst = idxAttr.array as Uint32Array, off = h * ICAP, vb = h * VCAP;
    const n = Math.min(count, src.length - from);
    for (let k = 0; k < n; k++) dst[off + from + k] = src[from + k] + vb;
    return n;
  }
  allocate(maxTierIdx);
  let data = buildLodGeometry(params);
  let active = 0;
  writeHalf(0, data, 0, 0, data.packed.length);
  writeHalf(0, data, 1, 0, data.index.length);

  // ------------------------------------------------------------------------------------ uniforms
  const levelVecs: Vector4[] = [];
  for (let i = 0; i <= MAX_LOD_LEVELS; i++) levelVecs.push(new Vector4(0, 0, 1, 0));
  const uLevels = uniformArray(levelVecs, 'vec4');       // per level: (C_l - B) (m), spacing (m), hasParent
  const uPhi = uniform(new Vector2());                   // foot - B (m)
  const uDFoot = uniform(new Vector2());                 // foot - previous foot (m)
  const uLast = uniform(0);                              // index of the last drawn level (closure attaches there)
  const uMorphA = uniform(params.morphStart);
  const uMorphB = uniform(params.morphEnd);
  const uHalfHole = uniform(params.halfCells / 2);
  const uCamHeight = uniform(0);                         // camera altitude above the foot (m)
  const uFarNdc = uniform(1);                            // closure clip z / w (1, or 0 with reversed depth)

  // ------------------------------------------------------------------------------------ vertex nodes
  // Mirrors lod-math.ts emulateVertex op for op on the grid (bit-exact level boundaries rely on the op order).
  const packed = positionGeometry;
  const lk = packed.z;
  const levelAttr = floor(lk.mul(0.25));
  const kind = lk.sub(levelAttr.mul(4));
  const skirt = kind.greaterThan(0.5);
  const lvlF = select(skirt, uLast, levelAttr);
  const A = uLevels.element(int(lvlF));
  const Pa = uLevels.element(int(lvlF.add(1)));
  const s = A.z;
  const g = packed.xy;
  const q0 = A.xy.add(g.mul(s));
  const p0 = q0.sub(uPhi);
  const dmax = max(abs(p0.x), abs(p0.y));
  const morph = clamp(dmax.div(s).sub(uMorphA).div(uMorphB.sub(uMorphA)), 0, 1).mul(A.w);
  const odd = g.sub(floor(g.mul(0.5)).mul(2));
  const q1 = q0.sub(odd.mul(s).mul(morph));
  const half = uHalfHole.mul(Pa.z);
  const q2 = select(A.w.greaterThan(0.5), clamp(q1, Pa.xy.sub(half), Pa.xy.add(half)), q1);
  const pg = q2.sub(uPhi);
  // Closure outer loops (kinds 2, 3): kind 3 at eye level (projects onto the true horizon line), kind 2 the flat
  // sea-level annulus edge, both placed near the rendering camera's far plane (1e6 m when far is infinite / huge) so
  // they stay inside the clip volume for any far. Their depth does not matter with enableClosure (forced to far).
  const outer = kind.greaterThan(1.5);
  const top = kind.greaterThan(2.5);
  const dirH = normalize(pg);
  // (cameraViewMatrix * modelWorldMatrix inline: the shared modelViewMatrix node gets emitted as a variable assigned
  // inside whichever branch references it first, leaving it unset for the other vertices.)
  const vdir = cameraViewMatrix.mul(modelWorldMatrix.mul(vec4(dirH.x, 0, dirH.y, 0))).xyz;
  const cosView = max(vdir.z.negate(), 0.2);
  const upViewZ = cameraViewMatrix.mul(modelWorldMatrix.mul(vec4(0, 1, 0, 0))).z;
  const depthFar = min(cameraFar.mul(0.999), 1e6);
  const depthTarget = select(top, depthFar, depthFar.mul(0.995));
  const yOffset = select(top, float(0), uCamHeight.negate()); // vertex height - camera height
  const tOut = max(depthTarget.add(yOffset.mul(upViewZ)), depthTarget.mul(0.5)).div(cosView);
  const pOut = dirH.mul(tOut);
  const pl = select(outer, pOut, pg);
  const ql = select(outer, pOut.add(uPhi), q2);
  const y = select(top, uCamHeight, float(0));
  const localPos = vec3(pl.x, y, pl.y);
  // Previous frame (motion vectors): the same WORLD point in last frame's local frame. Grid vertices: lattice point
  // minus last frame's foot, with LAST frame's morph (morph depends on the camera). Closure: camera-attached.
  const phiPrev = uPhi.sub(uDFoot);
  const p0Prev = q0.sub(phiPrev);
  const morphPrev = clamp(max(abs(p0Prev.x), abs(p0Prev.y)).div(s).sub(uMorphA).div(uMorphB.sub(uMorphA)), 0, 1).mul(A.w);
  const q1Prev = q0.sub(odd.mul(s).mul(morphPrev));
  const q2Prev = select(A.w.greaterThan(0.5), clamp(q1Prev, Pa.xy.sub(half), Pa.xy.add(half)), q1Prev);
  const pPrev = q2Prev.sub(phiPrev);
  const previousLocalPos = select(outer, localPos, vec3(pPrev.x, 0, pPrev.y));
  const distanceToCamera = length(vec3(pl.x, y.sub(uCamHeight), pl.y));
  const cellSize = select(skirt, distanceToCamera.mul(0.05), s.mul(morph.add(1)));
  const isSkirt = select(skirt, float(1), float(0));
  const level = select(skirt, float(-1), levelAttr);
  const cameraLocal = vec3(0, uCamHeight, 0);
  // positionLocal = the material's positionNode result (vertexNode is built after setupPosition).
  const clipPosition = cameraProjectionMatrix.mul(highpModelViewMatrix).mul(vec4(positionLocal, 1));
  const closureClipPosition = vec4(clipPosition.x, clipPosition.y, clipPosition.w.mul(uFarNdc), clipPosition.w);

  const periods: number[] = [];
  const anchors: any[] = [];
  const tileNodes: any[] = [];
  function tileCoord(period: number) {
    if (!(period > 0)) throw new RangeError('tileCoord: period must be > 0');
    let i = periods.indexOf(period);
    if (i < 0) {
      i = periods.length;
      periods.push(period);
      anchors.push(uniform(new Vector2()));
      tileNodes.push(vec2(anchors[i]).add(ql));
      if (state.hasPrev) updateAnchors();
    }
    return tileNodes[i];
  }

  const nodes: OceanLodNodes = {
    localPos, previousLocalPos, clipPosition, closureClipPosition, tileCoord, distanceToCamera, morphFactor: morph,
    cellSize, isSkirt, level, cameraLocal,
  };

  const mesh = new Mesh(gridGeom, opts.material);
  mesh.name = 'ocean.lod';
  mesh.frustumCulled = false; // whole-object culling is meaningless here: sector culling via groups instead
  let closureMesh: any = null;

  // ------------------------------------------------------------------------------------ frustum culling (groups)
  const MAX_RANGES = (MAX_LOD_LEVELS + 1) * LOD_SECTORS + 2;
  const ranges = new Int32Array(MAX_RANGES * 2);
  const groups: { start: number; count: number; materialIndex: number }[] = [];
  for (let i = 0; i < MAX_RANGES; i++) groups.push({ start: 0, count: 0, materialIndex: 1 }); // index 1 = no material: skipped
  gridGeom.groups = groups as any;
  const planes = new Float64Array(16);
  const pv = new Matrix4();
  let culledIndexCount = 0, rangeCount = 1;
  const lod = { culling: true };

  // ------------------------------------------------------------------------------------ per-frame update
  const camR: Vec3 = { x: 0, y: 0, z: 0 };
  const camW: Vec3 = { x: 0, y: 0, z: 0 };
  const footW: Vec3 = { x: 0, y: 0, z: 0 };
  const footR: Vec3 = { x: 0, y: 0, z: 0 };
  const wrappedTmp: Vec3 = { x: 0, y: 0, z: 0 };
  const south: Vec3 = { x: 0, y: 0, z: 0 };
  const frame: TangentFrame = { east: { x: 1, y: 0, z: 0 }, north: { x: 0, y: 0, z: -1 }, up: { x: 0, y: 1, z: 0 } };
  const vE = new Vector3(), vU = new Vector3(), vS = new Vector3();
  const basis = new Matrix4();
  let lastFar = 20000;

  function updateAnchors() {
    const E = frame.east;
    const bDotE = footR.x * E.x + footR.y * E.y + footR.z * E.z + (state.baseU - state.footU);
    const bDotS = footR.x * south.x + footR.y * south.y + footR.z * south.z + (state.baseV - state.footV);
    for (let i = 0; i < periods.length; i++) {
      const P = periods[i];
      origin.wrapped(P, wrappedTmp);
      (anchors[i].value as Vector2).set(
        tileAnchor(E, bDotE, state.baseU, wrappedTmp, P),
        tileAnchor(south, bDotS, state.baseV, wrappedTmp, P),
      );
    }
  }

  function setRanges() {
    const base = active * ICAP;
    const all = lodDrawCount(state, data);
    const skip = closureMesh ? data.skirtIndexCount : 0;
    gridGeom.setDrawRange(base + skip, all - skip);
    closureGeom.setDrawRange(base, data.skirtIndexCount);
  }

  function cull(camera: any) {
    let n = 0;
    culledIndexCount = 0;
    if (lod.culling && camera && camera.projectionMatrix && camera.matrixWorldInverse) {
      pv.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      frustumSidePlanes(pv.elements, planes);
      const E = frame.east;
      n = lodCullRanges(state, params, data, active * ICAP, planes, footR.x, footR.y, footR.z, E.x, E.y, E.z,
        south.x, south.y, south.z, cullMargin, ranges);
    } else {
      ranges[0] = active * ICAP + data.skirtIndexCount;
      ranges[1] = lodDrawCount(state, data) - data.skirtIndexCount;
      n = 1;
    }
    let gi = 0;
    if (!closureMesh) { // inline closure: always drawn
      const gr = groups[gi++]; gr.start = active * ICAP; gr.count = data.skirtIndexCount; gr.materialIndex = 0;
    }
    for (let i = 0; i < n; i++) {
      const gr = groups[gi++]; gr.start = ranges[i * 2]; gr.count = ranges[i * 2 + 1]; gr.materialIndex = 0;
      culledIndexCount += ranges[i * 2 + 1];
    }
    rangeCount = gi;
    for (; gi < MAX_RANGES; gi++) { groups[gi].count = 0; groups[gi].materialIndex = 1; }
  }

  function updateFromRender(advance: boolean) {
    origin.toWorld(camR, camW);
    const h = shape.altitude(camW);
    shape.atAltitude(camW, 0, footW);
    shape.tangentFrame(footW, frame);
    const E = frame.east, N = frame.north, U = frame.up;
    south.x = -N.x; south.y = -N.y; south.z = -N.z;
    const fu = footW.x * E.x + footW.y * E.y + footW.z * E.z + debug.footOffsetU;
    const fv = footW.x * south.x + footW.y * south.y + footW.z * south.z + debug.footOffsetV;
    state.noSnap = debug.noSnap;
    // Rings stay well inside a finite far plane: max-norm reach <= 0.6 far (corner 0.85 far).
    updateLodState(state, params, fu, fv, h, advance, Number.isFinite(lastFar) ? 0.6 * lastFar : Infinity);

    origin.toRender(footW, footR);
    footR.x += E.x * debug.footOffsetU + south.x * debug.footOffsetV;
    footR.y += E.y * debug.footOffsetU + south.y * debug.footOffsetV;
    footR.z += E.z * debug.footOffsetU + south.z * debug.footOffsetV;
    mesh.position.set(footR.x, footR.y, footR.z);
    vE.set(E.x, E.y, E.z); vU.set(U.x, U.y, U.z); vS.set(south.x, south.y, south.z);
    basis.makeBasis(vE, vU, vS);
    mesh.quaternion.setFromRotationMatrix(basis);
    if (closureMesh) { closureMesh.position.copy(mesh.position); closureMesh.quaternion.copy(mesh.quaternion); }

    const lv = state.level;
    for (let l = 0; l <= params.levels; l++) levelVecs[l].set(lv[l * 4], lv[l * 4 + 1], lv[l * 4 + 2], lv[l * 4 + 3]);
    (uPhi.value as Vector2).set(state.phiU, state.phiV);
    (uDFoot.value as Vector2).set(state.dFootU, state.dFootV);
    uLast.value = state.drawnLevels - 1;
    uCamHeight.value = h;
    setRanges();
    updateAnchors();
  }

  // ---------------------------------------------------------------------------------- tier change (no stall)
  // 1) build: the new tier's arrays are generated in BUILD_BUDGET_MS slices; 2) upload: copied into the idle half in
  // UPLOAD_CHUNK-element update ranges (one per attribute per frame); 3) swap: params, morph uniforms, active half
  // (drawRange) in one update. The old tier stays drawn throughout. Above maxTier: reallocation (one-frame hitch).
  const BUILD_BUDGET_MS = 2;
  const UPLOAD_CHUNK = 1 << 18; // elements (1 MB of float32 / uint32)
  let pending: {
    params: LodParams; it: Generator<void, LodGeometryData, void> | null; data: LodGeometryData | null;
    pos: number; idx: number;
  } | null = null;
  function stepPending() {
    if (!pending) return;
    if (pending.it) {
      const t0 = performance.now();
      let r = pending.it.next();
      while (!r.done && performance.now() - t0 < BUILD_BUDGET_MS) r = pending.it.next();
      if (!r.done) return;
      pending.it = null;
      pending.data = r.value;
      const c = lodGeometryCounts(pending.params);
      if (c.vertexCount > VCAP || c.triangleCount * 3 > ICAP) {
        // Above maxTier: reallocate both halves (hitch), keep the current tier in half 0.
        allocate(TIER_ORDER.indexOf(pending.params.tier));
        active = 0;
        writeHalf(0, data, 0, 0, data.packed.length);
        writeHalf(0, data, 1, 0, data.index.length);
      }
      return; // upload starts next frame
    }
    const d = pending.data!, h = 1 - active;
    if (pending.pos < d.packed.length) {
      const n = writeHalf(h, d, 0, pending.pos, UPLOAD_CHUNK);
      posAttr.addUpdateRange(h * VCAP * 3 + pending.pos, n);
      posAttr.needsUpdate = true;
      pending.pos += n;
    }
    if (pending.idx < d.index.length) {
      const n = writeHalf(h, d, 1, pending.idx, UPLOAD_CHUNK);
      idxAttr.addUpdateRange(h * ICAP + pending.idx, n);
      idxAttr.needsUpdate = true;
      pending.idx += n;
    }
    if (pending.pos < d.packed.length || pending.idx < d.index.length) return;
    // Swap (the last chunk is uploaded with this frame's render, before the draw that uses it).
    params = pending.params;
    data = d;
    active = h;
    pending = null;
    uMorphA.value = params.morphStart;
    uMorphB.value = params.morphEnd;
    uHalfHole.value = params.halfCells / 2;
    state.log2Spacing = NaN;
  }

  function update(camera: any, org?: FloatingOrigin) {
    if (org) origin = org;
    const e = camera.matrixWorld.elements;
    camR.x = e[12]; camR.y = e[13]; camR.z = e[14];
    lastFar = typeof camera.far === 'number' && camera.far > 0 ? camera.far : Infinity;
    stepPending();
    updateFromRender(true);
    cull(camera);
  }

  // Origin shift: the camera's render position moves by -delta; re-place the mesh + anchors right away.
  const onShift = (delta: Readonly<Vec3>) => {
    camR.x -= delta.x; camR.y -= delta.y; camR.z -= delta.z;
    if (state.hasPrev) updateFromRender(false);
  };
  let unsubscribe = origin.onShift(onShift);

  const statsObj = {} as OceanLodStats;
  const self: OceanLod = {
    mesh,
    get closure() { return closureMesh; },
    nodes,
    get params() { return params; },
    state,
    update,
    debug,
    get culling() { return lod.culling; },
    set culling(v: boolean) { lod.culling = v; },
    setMaterial(material: any) {
      mesh.material = [material];
    },
    cullFor(camera: any) { if (state.hasPrev) cull(camera); },
    enableClosure(material: any, o?: { reversedDepth?: boolean }) {
      uFarNdc.value = o?.reversedDepth ? 0 : 1;
      material.positionNode = localPos;
      material.vertexNode = closureClipPosition;
      material.depthWrite = false;
      material.depthTest = true;
      material.transparent = false;
      if (!closureMesh) {
        closureMesh = new Mesh(closureGeom, material);
        closureMesh.name = 'ocean.lod.closure';
        closureMesh.frustumCulled = false;
        closureMesh.renderOrder = -1e9; // first opaque draw: a background everything else overwrites
        closureMesh.castShadow = false;
        closureMesh.receiveShadow = false;
        closureMesh.position.copy(mesh.position);
        closureMesh.quaternion.copy(mesh.quaternion);
      } else closureMesh.material = material;
      if (state.hasPrev) setRanges();
      return closureMesh;
    },
    getFoot(out: { u: number; v: number }) {
      out.u = state.footU; out.v = state.footV;
      return state.hasPrev;
    },
    setTier(tier: LodTier) {
      if (pending) { if (pending.params.tier === tier) return; pending = null; }
      if (tier === params.tier) return;
      const p = lodParams(tier, { coverage: opts.coverage });
      pending = { params: p, it: lodGeometrySteps(p), data: null, pos: 0, idx: 0 };
    },
    get pendingTier() { return pending ? pending.params.tier : null; },
    stats(out?: OceanLodStats): OceanLodStats {
      const o = out ?? statsObj;
      o.tier = params.tier;
      o.vertices = data.vertexCount;
      o.triangles = data.triangleCount;
      o.drawnTriangles = (lodDrawCount(state, data) - data.skirtIndexCount) / 3;
      o.culledDrawTriangles = Array.isArray(mesh.material) ? culledIndexCount / 3 : o.drawnTriangles;
      o.drawRanges = Array.isArray(mesh.material) ? rangeCount : 1;
      o.drawnLevels = state.drawnLevels;
      o.spacing0 = state.spacing0;
      o.ringReach = (params.halfCells + params.margin) * state.spacing0 * 2 ** (state.drawnLevels - 1);
      o.pendingTier = pending ? pending.params.tier : null;
      o.closureEnabled = closureMesh !== null;
      return o;
    },
    dispose() {
      unsubscribe();
      unsubscribe = () => {};
      mesh.removeFromParent();
      if (closureMesh) closureMesh.removeFromParent();
      gridGeom.dispose();
      closureGeom.dispose();
    },
  };
  return self;
}
