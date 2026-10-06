import { test } from "node:test";
import assert from "node:assert/strict";
import { createFreeCamera } from "../src/camera/free-camera.js";

function step(camera, frames, yaw, input, dt = 1 / 60, pitch = 0) {
  let moving = false;
  for (let i = 0; i < frames; i++)
    moving = camera.update(dt, yaw, input, pitch);
  return moving;
}

test("movement follows the look direction and strafes to the right", () => {
  const ahead = createFreeCamera({ radius: 1000, speed: 10, boost: 1 });
  step(ahead, 180, 0, { forward: true });
  assert(ahead.z < -8);
  assert(Math.abs(ahead.x) < 0.5);

  const east = createFreeCamera({ radius: 1000, speed: 10, boost: 1 });
  step(east, 180, Math.PI / 2, { forward: true });
  assert(east.x > 8);
  assert(Math.abs(east.z) < 0.5);

  const strafe = createFreeCamera({ radius: 1000, speed: 10, boost: 1 });
  step(strafe, 180, 0, { right: true });
  assert(strafe.x > 8);
  assert(Math.abs(strafe.z) < 0.5);
});

test("diagonal movement is not faster than moving forward", () => {
  const straight = createFreeCamera({ radius: 5000, speed: 10, boost: 1 });
  const diagonal = createFreeCamera({ radius: 5000, speed: 10, boost: 1 });
  step(straight, 180, 0, { forward: true });
  step(diagonal, 180, 0, { forward: true, right: true });
  const straightDistance = Math.hypot(straight.x, straight.z);
  const diagonalDistance = Math.hypot(diagonal.x, diagonal.z);
  assert(Math.abs(straightDistance - diagonalDistance) / straightDistance < 0.05);
});

test("shift boost covers more ground and the radius is a hard limit", () => {
  const walk = createFreeCamera({ radius: 1000, speed: 6, boost: 3 });
  const run = createFreeCamera({ radius: 1000, speed: 6, boost: 3 });
  step(walk, 90, 0, { forward: true });
  step(run, 90, 0, { forward: true, boost: true });
  assert(Math.abs(run.z) > Math.abs(walk.z) * 2);

  const limited = createFreeCamera({ radius: 10, speed: 100, boost: 1 });
  step(limited, 120, 0, { forward: true });
  assert(Math.hypot(limited.x, limited.z) <= 10 + 1e-6);
});

test("Q and E move straight down and up, and speed 2 matches the base pace", () => {
  const down = createFreeCamera({
    radius: 500,
    speed: 10,
    boost: 1,
    eyeY: 10,
    minY: -50,
    maxY: 80,
  });
  step(down, 180, 0, { down: true });
  assert(down.y < 8);
  assert(Math.abs(down.x) < 0.5);
  assert(Math.abs(down.z) < 0.5);

  const slow = createFreeCamera({
    radius: 5000,
    speed: 10,
    boost: 1,
    eyeY: 0,
  });
  const fast = createFreeCamera({
    radius: 5000,
    speed: 10,
    boost: 1,
    eyeY: 0,
  });
  assert.equal(slow.speedLevel, 2);
  assert.equal(slow.setSpeedLevel(1), 1);
  assert.equal(fast.setSpeedLevel(5), 5);
  assert.equal(fast.setSpeedLevel(9), 5);
  step(slow, 180, 0, { up: true });
  step(fast, 180, 0, { up: true });
  assert(fast.y > slow.y * 2);
});

test("looking up flies upward and altitude stays inside its limits", () => {
  const rising = createFreeCamera({
    radius: 5000,
    speed: 10,
    boost: 1,
    eyeY: 0,
    minY: -20,
    maxY: 80,
  });
  step(rising, 180, 0, { forward: true }, 1 / 60, Math.PI / 6);
  assert(rising.y > 2);
  assert(Math.abs(rising.y / -rising.z - Math.tan(Math.PI / 6)) < 0.08);

  const capped = createFreeCamera({
    radius: 5000,
    speed: 100,
    boost: 1,
    eyeY: 0,
    minY: -20,
    maxY: 30,
  });
  step(capped, 120, 0, { forward: true }, 1 / 60, Math.PI / 2);
  assert.equal(capped.y, 30);
  step(capped, 120, 0, { forward: true }, 1 / 60, -Math.PI / 2);
  assert.equal(capped.y, -20);
});

test("reset returns to the origin and requests one more update", () => {
  const camera = createFreeCamera({ radius: 1000, speed: 10, boost: 1 });
  step(camera, 60, 0, { forward: true });
  assert(camera.z < 0);
  camera.reset();
  assert.equal(camera.x, 0);
  assert.equal(camera.y, 4.8);
  assert.equal(camera.z, 0);
  assert.equal(camera.update(1 / 60, 0, {}), true);
  assert.equal(camera.update(1 / 60, 0, {}), false);
});
