// DEMO ONLY: TypeScript port of the reference src/camera/free-camera.js (+ the drag-look / wheel-speed /
// WASD-QE-Shift input wiring from src/ocean.js). Position is world space float64 metres (flat, +Y up).
// Differences: no 150 m roaming radius (the demo must be able to fly far for floating-origin checks),
// and the position can be set directly (URL ?cam=).

const STOPPED = 1e-3;
export const PITCH_LIMIT = 1.52; // rad

export interface FreeCameraOptions {
  speed?: number; // m/s at speed level 2
  boost?: number; // shift multiplier
  minY?: number;
  maxY?: number;
}

export interface FreeCamera {
  x: number; y: number; z: number;
  yaw: number;   // rad, 0 looks towards -Z, positive turns towards +X
  pitch: number; // rad, positive looks up
  speedLevel: number; // 1..5
  /** Integrate velocity; returns true when the camera moved. dt in seconds. */
  update(dt: number): boolean;
  /** Writes the unit forward vector. */
  forward(out: { x: number; y: number; z: number }): void;
  attach(element: HTMLElement): void;
  dispose(): void;
  enabled: boolean;
  onSpeedChange?: (level: number) => void;
}

const KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight']);

export function createFreeCamera(opts: FreeCameraOptions = {}): FreeCamera {
  const speed = opts.speed ?? 6, boost = opts.boost ?? 3;
  const minY = opts.minY ?? -50, maxY = opts.maxY ?? 5000;
  let vx = 0, vy = 0, vz = 0;
  const held = new Set<string>();
  const pointers = new Map<number, { x: number; y: number }>();
  let wheelRemainder = 0;
  let element: HTMLElement | null = null;

  const onKeyDown = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement)?.matches?.('input,textarea,select,[contenteditable="true"]')) return;
    if (KEYS.has(e.code)) { held.add(e.code); e.preventDefault(); }
  };
  const onKeyUp = (e: KeyboardEvent) => { held.delete(e.code); };
  const onBlur = () => held.clear();
  const onDown = (e: PointerEvent) => {
    if (!cam.enabled) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    element?.setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent) => {
    const prev = pointers.get(e.pointerId);
    if (!prev || !cam.enabled) return;
    if (pointers.size === 1) {
      cam.yaw += (e.clientX - prev.x) * 0.0025;
      cam.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, cam.pitch - (e.clientY - prev.y) * 0.002));
    }
    prev.x = e.clientX; prev.y = e.clientY;
  };
  const onUp = (e: PointerEvent) => { pointers.delete(e.pointerId); };
  const onWheel = (e: WheelEvent) => {
    if (!cam.enabled) return;
    e.preventDefault();
    const pixels = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;
    wheelRemainder += e.deltaY * pixels;
    while (wheelRemainder >= 100) { wheelRemainder -= 100; setLevel(cam.speedLevel - 1); }
    while (wheelRemainder <= -100) { wheelRemainder += 100; setLevel(cam.speedLevel + 1); }
  };
  function setLevel(l: number) {
    const next = Math.min(5, Math.max(1, Math.round(l)));
    if (next !== cam.speedLevel) { cam.speedLevel = next; cam.onSpeedChange?.(next); }
  }

  const cam: FreeCamera = {
    x: 0, y: 4.8, z: 8, yaw: 0, pitch: -0.28, speedLevel: 2, enabled: true,
    forward(out) {
      const cp = Math.cos(cam.pitch);
      out.x = Math.sin(cam.yaw) * cp; out.y = Math.sin(cam.pitch); out.z = -Math.cos(cam.yaw) * cp;
    },
    update(dt) {
      if (!(dt > 0) || !Number.isFinite(dt)) return false;
      const on = cam.enabled;
      const cp = Math.cos(cam.pitch);
      const fx = Math.sin(cam.yaw) * cp, fy = Math.sin(cam.pitch), fz = -Math.cos(cam.yaw) * cp;
      const rx = Math.cos(cam.yaw), rz = Math.sin(cam.yaw);
      let wx = 0, wy = 0, wz = 0;
      if (on && held.has('KeyW')) { wx += fx; wy += fy; wz += fz; }
      if (on && held.has('KeyS')) { wx -= fx; wy -= fy; wz -= fz; }
      if (on && held.has('KeyD')) { wx += rx; wz += rz; }
      if (on && held.has('KeyA')) { wx -= rx; wz -= rz; }
      if (on && held.has('KeyE')) wy += 1;
      if (on && held.has('KeyQ')) wy -= 1;
      const len = Math.hypot(wx, wy, wz);
      if (len > 0) { wx /= len; wy /= len; wz /= len; }
      const target = speed * (cam.speedLevel / 2) * (on && (held.has('ShiftLeft') || held.has('ShiftRight')) ? boost : 1);
      const smooth = 1 - Math.exp(-dt * 8);
      vx += (wx * target - vx) * smooth;
      vy += (wy * target - vy) * smooth;
      vz += (wz * target - vz) * smooth;
      if (wx === 0 && wy === 0 && wz === 0 && Math.hypot(vx, vy, vz) < STOPPED) { vx = 0; vy = 0; vz = 0; }
      cam.x += vx * dt; cam.y += vy * dt; cam.z += vz * dt;
      if (cam.y > maxY) { cam.y = maxY; if (vy > 0) vy = 0; } else if (cam.y < minY) { cam.y = minY; if (vy < 0) vy = 0; }
      return vx !== 0 || vy !== 0 || vz !== 0;
    },
    attach(el) {
      element = el;
      el.addEventListener('pointerdown', onDown);
      el.addEventListener('pointermove', onMove);
      for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) el.addEventListener(ev, onUp);
      addEventListener('wheel', onWheel, { passive: false, capture: true });
      addEventListener('keydown', onKeyDown);
      addEventListener('keyup', onKeyUp);
      addEventListener('blur', onBlur);
    },
    dispose() {
      if (element) {
        element.removeEventListener('pointerdown', onDown);
        element.removeEventListener('pointermove', onMove);
        for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture'] as const) element.removeEventListener(ev, onUp);
      }
      removeEventListener('wheel', onWheel, { capture: true });
      removeEventListener('keydown', onKeyDown);
      removeEventListener('keyup', onKeyUp);
      removeEventListener('blur', onBlur);
    },
  };
  return cam;
}
