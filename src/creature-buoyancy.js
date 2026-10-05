const TAU = Math.PI * 2;

const clamp = (value, minimum, maximum) =>
  Math.max(minimum, Math.min(maximum, value));

export function createBuoyancyState(initial = {}) {
  return {
    heave: Number.isFinite(initial.heave) ? initial.heave : 0,
    pitch: Number.isFinite(initial.pitch) ? initial.pitch : 0,
    roll: Number.isFinite(initial.roll) ? initial.roll : 0,
    heaveVelocity: Number.isFinite(initial.heaveVelocity)
      ? initial.heaveVelocity
      : 0,
    pitchVelocity: Number.isFinite(initial.pitchVelocity)
      ? initial.pitchVelocity
      : 0,
    rollVelocity: Number.isFinite(initial.rollVelocity)
      ? initial.rollVelocity
      : 0,
    valid: initial.valid ?? true,
  };
}

export function deriveBuoyancyTarget(samples) {
  const average =
      samples.center * 0.3 +
      (samples.bow + samples.stern) * 0.2 +
      (samples.port + samples.starboard) * 0.15,
    pitch = clamp((samples.bow - samples.stern) / 16, -0.08, 0.08),
    roll = clamp((samples.starboard - samples.port) / 4, -0.06, 0.06);
  return { heave: average * 0.75, pitch, roll };
}

export function applyCreatureFramePoint(point, origin, heading, frame) {
  const headingLength = Math.hypot(heading.x, heading.y) || 1,
    hx = heading.x / headingLength,
    hz = heading.y / headingLength,
    sx = -hz,
    sz = hx,
    dx = point.x - origin.x,
    dy = point.y - origin.y,
    dz = point.z - origin.z,
    localX = dx * hx + dz * hz,
    localZ = dx * sx + dz * sz,
    cp = Math.cos(frame.pitch),
    sp = Math.sin(frame.pitch),
    cr = Math.cos(frame.roll),
    sr = Math.sin(frame.roll),
    pitchedX = cp * localX - sp * dy,
    pitchedY = sp * localX + cp * dy,
    rolledY = cr * pitchedY - sr * localZ,
    rolledZ = sr * pitchedY + cr * localZ;
  return {
    x: origin.x + hx * pitchedX + sx * rolledZ,
    y: origin.y + rolledY + frame.heave,
    z: origin.z + hz * pitchedX + sz * rolledZ,
  };
}

function integrateCritical(position, velocity, target, dt, omega) {
  const displacement = position - target,
    combination = velocity + omega * displacement,
    decay = Math.exp(-omega * dt);
  return {
    position: target + (displacement + combination * dt) * decay,
    velocity: (velocity - omega * combination * dt) * decay,
  };
}

export function stepBuoyancyState(state, target, dt) {
  if (!(dt > 0) || !Number.isFinite(dt)) return state;
  const safeDt = Math.min(dt, 0.1),
    heave = integrateCritical(
      state.heave,
      state.heaveVelocity,
      target.heave,
      safeDt,
      2.15,
    ),
    pitch = integrateCritical(
      state.pitch,
      state.pitchVelocity,
      target.pitch,
      safeDt,
      1.85,
    ),
    roll = integrateCritical(
      state.roll,
      state.rollVelocity,
      target.roll,
      safeDt,
      1.7,
    );
  state.heave = heave.position;
  state.pitch = pitch.position;
  state.roll = roll.position;
  state.heaveVelocity = heave.velocity;
  state.pitchVelocity = pitch.velocity;
  state.rollVelocity = roll.velocity;
  state.valid = true;
  return state;
}

const FALLBACK_WAVES = [
  [[0.38, 0.92], 31, 0.25, 0.5],
  [[-0.14, 0.99], 17.3, 0.2, 1.7],
  [[0.73, 0.69], 9.8, 0.16, 2.9],
  [[-0.65, 0.76], 6.1, 0.1, 0.3],
  [[0.21, 0.98], 3.7, 0.075, 4.2],
  [[-0.81, -0.58], 2.2, 0.045, 2],
  [[0.9, 0.44], 1.37, 0.025, 1],
].map(([direction, wavelength, steepness, phase]) => {
  const length = Math.hypot(...direction);
  return {
    x: direction[0] / length,
    y: direction[1] / length,
    wavelength,
    steepness,
    phase,
  };
});

export function sampleFallbackSurface(x, z, uniforms) {
  const wave = uniforms.uWave?.value ?? 1,
    wind = uniforms.uWind?.value ?? 1,
    phases = uniforms.uWavePhase?.value ?? [],
    radialStep = uniforms.uGridRadialStep?.value ?? Math.log(9601) / 384,
    angularStep = uniforms.uGridAngularStep?.value ?? TAU / 512;
  let restX = x,
    restZ = z,
    height = 0;
  for (let iteration = 0; iteration < 3; iteration++) {
    let displacedX = restX,
      displacedZ = restZ;
    height = 0;
    const radius = Math.hypot(restX, restZ - 8),
      spacing = Math.max((radius + 0.25) * radialStep, radius * angularStep);
    for (let i = 0; i < FALLBACK_WAVES.length; i++) {
      const item = FALLBACK_WAVES[i],
        k = TAU / item.wavelength,
        lodT = clamp((spacing / item.wavelength - 0.18) / 0.3, 0, 1),
        lod = 1 - lodT * lodT * (3 - 2 * lodT),
        vertical = (item.steepness / k) * wave * lod,
        horizontal =
          (item.steepness / k) *
          Math.min(wave, (1.04 * 1.15) / Math.max(1.15, 0.35 + wind)) *
          lod,
        angle =
          k * (item.x * restX + item.y * restZ) -
          (phases[i] ?? 0) +
          item.phase;
      displacedX += item.x * horizontal * Math.cos(angle);
      displacedZ += item.y * horizontal * Math.cos(angle);
      height += vertical * Math.sin(angle);
    }
    restX -= displacedX - x;
    restZ -= displacedZ - z;
  }
  return height;
}

export const CREATURE_FRAME_GLSL = `
uniform sampler2D uCreatureBuoyancy;
uniform float uCreatureBuoyancyReady;
vec3 creatureFrameValues(){
  return uCreatureBuoyancyReady>.5
    ? texture2D(uCreatureBuoyancy,vec2(.25,.5)).xyz
    : vec3(0.);
}
mat3 creatureFrameRotation(vec2 heading){
  vec3 frame=creatureFrameValues();
  float cp=cos(frame.y),sp=sin(frame.y),cr=cos(frame.z),sr=sin(frame.z);
  mat3 pitch=mat3(cp,sp,0.,-sp,cp,0.,0.,0.,1.);
  mat3 roll=mat3(1.,0.,0.,0.,cr,sr,0.,-sr,cr);
  vec2 h=normalize(heading+vec2(1e-6,0.));
  mat3 basis=mat3(vec3(h.x,0.,h.y),vec3(0.,1.,0.),vec3(-h.y,0.,h.x));
  return basis*roll*pitch*transpose(basis);
}
vec3 creatureApplyFramePoint(vec3 world,vec3 origin,vec2 heading){
  vec3 framed=origin+creatureFrameRotation(heading)*(world-origin);
  framed.y+=creatureFrameValues().x;
  return framed;
}
vec3 creatureApplyFrameVector(vec3 vector,vec2 heading){
  return creatureFrameRotation(heading)*vector;
}
`;

export const CREATURE_SURFACE_GLSL = `
float creatureSurfaceHeight(vec2 world){
  vec2 rest=world;
  float height=0.;
  for(int iteration=0;iteration<3;iteration++){
    if(uSpectral>.5){
      vec4 field=spectralField(rest,4.);
      vec2 displaced=rest+field.yz;
      rest-=displaced-world;
      height=field.x;
    }else{
      vec3 point,normal;float compression,variance;
      ocean(rest,point,normal,compression,variance);
      rest-=point.xz-world;
      height=point.y;
    }
  }
  return height;
}
`;

const FALLBACK_SURFACE_GLSL = `
uniform float uTime;
uniform vec4 uCreaturePose;
uniform vec2 uCreatureHeading;
float creatureSurfaceHeight(vec2 world){
  return sin(dot(world,vec2(.055,.031))+uTime*.72)*.12;
}
`;

const BUOYANCY_FRAGMENT = `
uniform sampler2D uPreviousBuoyancy;
uniform float uBuoyancyDt,uBuoyancyInitialized;
vec2 spring(float position,float velocity,float target,float dt,float omega){
  float displacement=position-target;
  float combination=velocity+omega*displacement;
  float decay=exp(-omega*dt);
  return vec2(target+(displacement+combination*dt)*decay,(velocity-omega*combination*dt)*decay);
}
void main(){
  vec2 heading=normalize(uCreatureHeading+vec2(1e-6,0.));
  vec2 side=vec2(-heading.y,heading.x);
  float center=creatureSurfaceHeight(uCreaturePose.xz);
  float bow=creatureSurfaceHeight(uCreaturePose.xz+heading*8.);
  float stern=creatureSurfaceHeight(uCreaturePose.xz-heading*8.);
  float port=creatureSurfaceHeight(uCreaturePose.xz+side*2.);
  float starboard=creatureSurfaceHeight(uCreaturePose.xz-side*2.);
  vec3 target=vec3(
    .75*(center*.3+(bow+stern)*.2+(port+starboard)*.15),
    clamp((bow-stern)/16.,-.08,.08),
    clamp((starboard-port)/4.,-.06,.06)
  );
  vec3 position=texture2D(uPreviousBuoyancy,vec2(.25,.5)).xyz;
  vec3 velocity=texture2D(uPreviousBuoyancy,vec2(.75,.5)).xyz;
  if(uBuoyancyInitialized<.5){position=target;velocity=vec3(0.);}
  else if(uBuoyancyDt>0.){
    vec2 h=spring(position.x,velocity.x,target.x,uBuoyancyDt,2.15);
    vec2 p=spring(position.y,velocity.y,target.y,uBuoyancyDt,1.85);
    vec2 r=spring(position.z,velocity.z,target.z,uBuoyancyDt,1.7);
    position=vec3(h.x,p.x,r.x);velocity=vec3(h.y,p.y,r.y);
  }
  gl_FragColor=gl_FragCoord.x<1.?vec4(position,target.x):vec4(velocity,target.y);
}`;

const FULLSCREEN_VERTEX = `
void main(){gl_Position=vec4(position.xy,0.,1.);}
`;

function installBuoyancyUniforms(THREE, uniforms) {
  let ownsTexture = false;
  if (!uniforms.uCreatureBuoyancy) {
    const data = new Float32Array(8),
      texture = new THREE.DataTexture(
        data,
        2,
        1,
        THREE.RGBAFormat,
        THREE.FloatType,
      );
    texture.needsUpdate = true;
    texture.magFilter = texture.minFilter = THREE.NearestFilter;
    uniforms.uCreatureBuoyancy = { value: texture };
    ownsTexture = true;
  }
  if (!uniforms.uCreatureBuoyancyReady)
    uniforms.uCreatureBuoyancyReady = { value: 0 };
  return ownsTexture;
}

function cpuTargets(uniforms) {
  const pose = uniforms.uCreaturePose.value,
    heading = uniforms.uCreatureHeading.value,
    sideX = -heading.y,
    sideZ = heading.x,
    height = (x, z) => sampleFallbackSurface(x, z, uniforms);
  return {
    center: height(pose.x, pose.z),
    bow: height(pose.x + heading.x * 8, pose.z + heading.y * 8),
    stern: height(pose.x - heading.x * 8, pose.z - heading.y * 8),
    port: height(pose.x + sideX * 2, pose.z + sideZ * 2),
    starboard: height(pose.x - sideX * 2, pose.z - sideZ * 2),
  };
}

export function createCreatureBuoyancy(
  THREE,
  renderer,
  sharedUniforms,
  surfaceShader = "",
  options = {},
) {
  const ownsTexture = installBuoyancyUniforms(THREE, sharedUniforms),
    fallbackTexture = sharedUniforms.uCreatureBuoyancy.value,
    cpuState = createBuoyancyState(),
    gpuSupported =
      options.forceGpu === true ||
      (options.forceGpu !== false &&
        renderer.extensions?.has?.("EXT_color_buffer_float")),
    state = {
      mode: gpuSupported ? "gpu" : "cpu",
      initialized: false,
      read: 0,
      disposed: false,
      validated: false,
      lastTargets: null,
    };
  let targets = [], scene = null, camera = null, material = null, geometry = null;

  if (gpuSupported) try {
    targets = [0, 1].map(() => {
      const target = new THREE.WebGLRenderTarget(2, 1, {
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        depthBuffer: false,
        stencilBuffer: false,
      });
      target.texture.generateMipmaps = false;
      return target;
    });
    const passUniforms = {
      ...sharedUniforms,
      uPreviousBuoyancy: { value: fallbackTexture },
      uBuoyancyDt: { value: 0 },
      uBuoyancyInitialized: { value: 0 },
    };
    material = new THREE.ShaderMaterial({
      uniforms: passUniforms,
      vertexShader: FULLSCREEN_VERTEX,
      fragmentShader:
        (surfaceShader || FALLBACK_SURFACE_GLSL) + BUOYANCY_FRAGMENT,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
    geometry = new THREE.PlaneGeometry(2, 2);
    scene = new THREE.Scene();
    camera = new THREE.Camera();
    scene.add(new THREE.Mesh(geometry, material));
  } catch (error) {
    for (const target of targets) target.dispose();
    material?.dispose();
    geometry?.dispose();
    targets = [];
    material = geometry = scene = camera = null;
    state.mode = "cpu";
    options.onError?.(error);
  }

  function releaseGpu() {
    for (const target of targets) target.dispose();
    targets = [];
    material?.dispose();
    geometry?.dispose();
    material = geometry = scene = camera = null;
  }

  function updateCpu(dt) {
    const samples = cpuTargets(sharedUniforms),
      target = deriveBuoyancyTarget(samples);
    state.lastTargets = samples;
    if (!state.initialized) {
      Object.assign(cpuState, target, {
        heaveVelocity: 0,
        pitchVelocity: 0,
        rollVelocity: 0,
      });
      state.initialized = true;
    } else stepBuoyancyState(cpuState, target, dt);
    const data = fallbackTexture.image.data;
    data.set(
      [
        cpuState.heave,
        cpuState.pitch,
        cpuState.roll,
        target.heave,
        cpuState.heaveVelocity,
        cpuState.pitchVelocity,
        cpuState.rollVelocity,
        target.pitch,
      ],
      0,
    );
    fallbackTexture.needsUpdate = true;
    sharedUniforms.uCreatureBuoyancy.value = fallbackTexture;
    sharedUniforms.uCreatureBuoyancyReady.value = 1;
  }

  function update(_time, dt) {
    if (state.disposed) return;
    if (state.mode === "cpu") {
      updateCpu(dt);
      return;
    }
    if (state.initialized && !(dt > 0)) return;
    const previous = renderer.getRenderTarget(),
      face = renderer.getActiveCubeFace?.() ?? 0,
      level = renderer.getActiveMipmapLevel?.() ?? 0,
      write = 1 - state.read;
    try {
      material.uniforms.uPreviousBuoyancy.value = state.initialized
        ? targets[state.read].texture
        : fallbackTexture;
      material.uniforms.uBuoyancyDt.value = Math.min(Math.max(dt || 0, 0), 0.1);
      material.uniforms.uBuoyancyInitialized.value = state.initialized ? 1 : 0;
      renderer.setRenderTarget(targets[write]);
      renderer.render(scene, camera);
      if (!state.validated && renderer.getContext) {
        const gl = renderer.getContext();
        if (
          gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE
        )
          throw new Error("Whale buoyancy framebuffer incomplete");
        state.validated = true;
      }
      state.read = write;
      state.initialized = true;
      sharedUniforms.uCreatureBuoyancy.value = targets[state.read].texture;
      sharedUniforms.uCreatureBuoyancyReady.value = 1;
    } catch (error) {
      renderer.setRenderTarget(previous, face, level);
      releaseGpu();
      state.mode = "cpu";
      state.initialized = false;
      sharedUniforms.uCreatureBuoyancy.value = fallbackTexture;
      sharedUniforms.uCreatureBuoyancyReady.value = 0;
      options.onError?.(error);
      updateCpu(dt);
      return;
    } finally {
      renderer.setRenderTarget(previous, face, level);
    }
  }

  async function debugRead(time = sharedUniforms.uTime?.value ?? 0) {
    let data;
    if (state.mode === "gpu" && state.initialized) {
      data = new Float32Array(8);
      if (renderer.readRenderTargetPixelsAsync)
        await renderer.readRenderTargetPixelsAsync(
          targets[state.read],
          0,
          0,
          2,
          1,
          data,
        );
      else renderer.readRenderTargetPixels(targets[state.read], 0, 0, 2, 1, data);
    } else data = Float32Array.from(fallbackTexture.image.data);
    return {
      time,
      mode: state.mode,
      offset: Array.from(data.slice(0, 3)),
      velocity: Array.from(data.slice(4, 7)),
      targetHeave: data[3],
      targetPitch: data[7],
      waterTargets: state.lastTargets,
    };
  }

  return {
    update,
    debugRead,
    get mode() {
      return state.mode;
    },
    dispose() {
      if (state.disposed) return;
      state.disposed = true;
      releaseGpu();
      if (ownsTexture) fallbackTexture.dispose();
      sharedUniforms.uCreatureBuoyancyReady.value = 0;
    },
  };
}
