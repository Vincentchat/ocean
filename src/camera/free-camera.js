const STOPPED = 1e-3;

function unit3(x, y, z) {
  const length = Math.hypot(x, y, z);
  return length > 0 ? [x / length, y / length, z / length] : [0, 0, 0];
}

export function createFreeCamera({
  radius = 150,
  speed = 6,
  boost = 3,
  eyeY = 4.8,
  minY = -50,
  maxY = 400,
} = {}) {
  let x = 0,
    y = eyeY,
    z = 0,
    vx = 0,
    vy = 0,
    vz = 0,
    dirty = false;
  let speedLevel = 2;

  function reset() {
    x = 0;
    y = eyeY;
    z = 0;
    vx = 0;
    vy = 0;
    vz = 0;
    dirty = true;
  }

  function setSpeedLevel(level) {
    const next = Math.min(5, Math.max(1, Math.round(level)));
    if (next === speedLevel) return speedLevel;
    speedLevel = next;
    return speedLevel;
  }

  function nudgeY(delta) {
    if (!Number.isFinite(delta) || delta === 0) return false;
    const next = Math.min(maxY, Math.max(minY, y + delta));
    if (next === y) return false;
    y = next;
    dirty = true;
    return true;
  }

  function update(dt, yaw, input = {}, pitch = 0) {
    const wasDirty = dirty;
    dirty = false;
    if (
      !(dt > 0) ||
      !Number.isFinite(dt) ||
      !Number.isFinite(yaw) ||
      !Number.isFinite(pitch)
    )
      return wasDirty;
    const cosPitch = Math.cos(pitch),
      forwardX = Math.sin(yaw) * cosPitch,
      forwardY = Math.sin(pitch),
      forwardZ = -Math.cos(yaw) * cosPitch,
      rightX = Math.cos(yaw),
      rightZ = Math.sin(yaw);
    let wishX = 0,
      wishY = 0,
      wishZ = 0;
    if (input.forward) {
      wishX += forwardX;
      wishY += forwardY;
      wishZ += forwardZ;
    }
    if (input.back) {
      wishX -= forwardX;
      wishY -= forwardY;
      wishZ -= forwardZ;
    }
    if (input.right) {
      wishX += rightX;
      wishZ += rightZ;
    }
    if (input.left) {
      wishX -= rightX;
      wishZ -= rightZ;
    }
    if (input.up) wishY += 1;
    if (input.down) wishY -= 1;
    [wishX, wishY, wishZ] = unit3(wishX, wishY, wishZ);
    const targetSpeed = speed * (speedLevel / 2) * (input.boost ? boost : 1);
    const smooth = 1 - Math.exp(-dt * 8);
    vx += (wishX * targetSpeed - vx) * smooth;
    vy += (wishY * targetSpeed - vy) * smooth;
    vz += (wishZ * targetSpeed - vz) * smooth;
    if (
      wishX === 0 &&
      wishY === 0 &&
      wishZ === 0 &&
      Math.hypot(vx, vy, vz) < STOPPED
    ) {
      vx = 0;
      vy = 0;
      vz = 0;
    }
    x += vx * dt;
    y += vy * dt;
    z += vz * dt;
    const distance = Math.hypot(x, z);
    if (distance > radius) {
      const scale = radius / distance;
      x *= scale;
      z *= scale;
      const outward = vx * x + vz * z;
      if (outward > 0) {
        vx -= (outward / (radius * radius)) * x;
        vz -= (outward / (radius * radius)) * z;
      }
    }
    if (y > maxY) {
      y = maxY;
      if (vy > 0) vy = 0;
    } else if (y < minY) {
      y = minY;
      if (vy < 0) vy = 0;
    }
    return wasDirty || Math.hypot(vx, vy, vz) > 0;
  }

  return {
    update,
    reset,
    nudgeY,
    setSpeedLevel,
    get speedLevel() {
      return speedLevel;
    },
    get x() {
      return x;
    },
    get y() {
      return y;
    },
    get z() {
      return z;
    },
  };
}
