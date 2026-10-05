import {
  CREATURE_FRAME_GLSL,
  CREATURE_SURFACE_GLSL,
} from "./creature-buoyancy.js";

const MIST_COUNT = 28;
const DROPLET_COUNT = 10;

function installUniform(THREE, uniforms, name, value) {
  if (!uniforms[name]) uniforms[name] = { value };
  return uniforms[name];
}

function makeSeedData(count) {
  const values = new Float32Array(count * 4);
  let state = 0x6d2b79f5;
  const random = () => {
    state = Math.imul(state ^ (state >>> 15), 1 | state);
    state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
    return ((state ^ (state >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < values.length; i++) values[i] = random();
  return values;
}

export function backtrackEmissionPosition(position, velocity, elapsed, output = {}) {
  const seconds = Number.isFinite(elapsed) ? Math.max(elapsed, 0) : 0;
  output.x = position.x - velocity.x * seconds;
  output.y = position.y - velocity.y * seconds;
  output.z = position.z - velocity.z * seconds;
  return output;
}

const FALLBACK_SURFACE = `
uniform float uTime;
float creatureSurfaceHeight(vec2 world){
  return sin(dot(world,vec2(.055,.031))+uTime*.72)*.12;
}
`;

function missingUniform(source, type, name) {
  return new RegExp(`uniform[^;]*\\b${name}\\b`).test(source)
    ? ""
    : `uniform ${type} ${name};\n`;
}

function vertexShader(surfaceShader) {
  const surface = surfaceShader
    ? `${surfaceShader}\n${CREATURE_SURFACE_GLSL}`
    : FALLBACK_SURFACE;
  return `${surface}
${CREATURE_FRAME_GLSL}
${missingUniform(surfaceShader, "float", "uCreatureWake")}${missingUniform(surfaceShader, "float", "uCreatureBreath")}${missingUniform(surfaceShader, "float", "uCreatureBreathPhase")}${missingUniform(surfaceShader, "float", "uBreathEnabled")}${missingUniform(surfaceShader, "vec4", "uCreaturePose")}${missingUniform(surfaceShader, "vec2", "uCreatureHeading")}${missingUniform(surfaceShader, "vec3", "uCreatureVelocity")}
attribute vec4 aBreathSeed;
attribute float aBreathKind;
varying vec2 vBreathUv;
varying float vBreathAlpha,vBreathKind,vBreathSeed;

vec3 backtrackCreatureMotion(vec3 point,vec3 localOffset,vec2 heading,float elapsed){
  vec3 velocity=uCreatureVelocity;
  if(uCreatureBuoyancyReady>.5){
    vec3 frameVelocity=texture2D(uCreatureBuoyancy,vec2(.75,.5)).xyz;
    vec3 angularLocal=vec3(frameVelocity.z,0.,-frameVelocity.y);
    vec3 angularWorld=creatureApplyFrameVector(angularLocal,heading);
    vec3 framedOffset=creatureApplyFrameVector(localOffset,heading);
    velocity+=cross(angularWorld,framedOffset);
    velocity.y+=frameVelocity.x;
  }
  return point-velocity*elapsed;
}

void main() {
  vec2 heading=normalize(uCreatureHeading+vec2(1e-5,0.));
  vec3 origin=uCreaturePose.xyz;
  vec3 unframedAnchor=origin+vec3(heading.x*6.,2.1,heading.y*6.);
  vec3 anchor=creatureApplyFramePoint(unframedAnchor,origin,heading);
  float clearance=anchor.y-creatureSurfaceHeight(anchor.xz);
  float aboveWater=smoothstep(.025,.16,clearance);
  float gate=clamp(uCreatureBreath,0.,1.)*uCreatureWake*uCreaturePose.w*uBreathEnabled*aboveWater;

  vec2 crossWind=vec2(-heading.y,heading.x);
  vec2 downwind=normalize(vec2(-.63,.32)+heading*.18);
  vec3 world=anchor;
  float width,height,alpha;
  if (aBreathKind < .5) {
    float birth=aBreathSeed.w*.25;
    float age=clamp((uCreatureBreathPhase-birth)/.75,0.,1.);
    world=backtrackCreatureMotion(anchor,unframedAnchor-origin,heading,age*2.25);
    float life=smoothstep(0.,.10,age)*(1.-smoothstep(.72,1.,age));
    float expansion=smoothstep(.04,.68,age);
    float rise=.10+age*2.30+aBreathSeed.y*.10;
    float drift=age*(.48+.72*aBreathSeed.z);
    world.xz+=downwind*drift+crossWind*(aBreathSeed.x-.5)*(.15+rise*.28);
    world.y+=rise+sin(uTime*(.31+.11*aBreathSeed.w)+aBreathSeed.x*6.283)*.025*life;
    width=mix(.24,.56,aBreathSeed.z)*mix(.72,1.75,expansion);
    height=mix(.40,.82,aBreathSeed.w)*mix(.82,1.25,expansion);
    alpha=gate*life*mix(.065,.14,aBreathSeed.x);
  } else {
    float birth=aBreathSeed.w*.22;
    float age=clamp((uCreatureBreathPhase-birth)/.58,0.,1.);
    world=backtrackCreatureMotion(anchor,unframedAnchor-origin,heading,age*1.74);
    float life=smoothstep(0.,.08,age)*(1.-smoothstep(.76,1.,age));
    float speed=mix(.32,.66,aBreathSeed.y);
    world.xz+=(downwind*.48+crossWind*(aBreathSeed.x-.5)*.72)*speed*age;
    world.y+=.13+mix(.55,1.05,aBreathSeed.z)*age-1.28*age*age;
    width=mix(.018,.038,aBreathSeed.x);
    height=width*mix(2.1,3.8,aBreathSeed.y);
    alpha=gate*life*.34;
  }

  vec4 viewCenter=viewMatrix*vec4(world,1.);
  viewCenter.xy+=position.xy*vec2(width,height);
  gl_Position=projectionMatrix*viewCenter;
  vBreathUv=uv;
  vBreathAlpha=alpha;
  vBreathKind=aBreathKind;
  vBreathSeed=aBreathSeed.z;
}`;
}

const FALLBACK_LIGHTING = `
vec3 horizonColor(){return vec3(.16,.20,.22);}
vec3 sunlight(){return vec3(.72,.76,.78);}
vec3 tonemap(vec3 x){x=max(x,vec3(0));return pow(clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.),vec3(1./2.2));}
`;

function fragmentShader(lightingShader) {
  return `${lightingShader || FALLBACK_LIGHTING}
varying vec2 vBreathUv;
varying float vBreathAlpha,vBreathKind,vBreathSeed;
void main() {
  vec2 p=vBreathUv*2.-1.;
  float alpha;
  if (vBreathKind < .5) {
    float bend=p.x+(.13+.12*vBreathSeed)*p.y*p.y-.08*sin(p.y*5.7+vBreathSeed*8.);
    float shape=length(vec2(bend*1.42,p.y*.72));
    float torn=.77+.07*sin(p.y*13.7+p.x*7.1+vBreathSeed*19.)
      +.045*sin(p.y*25.1-p.x*5.3+vBreathSeed*31.);
    alpha=1.-smoothstep(torn-.18,torn,shape);
    alpha*=.76+.24*smoothstep(.18,.76,.5+.5*sin(p.y*10.3+vBreathSeed*23.));
  } else {
    alpha=1.-smoothstep(.22,.82,length(vec2(p.x*2.4,p.y*.68)));
  }
  vec3 mist=mix(horizonColor()*.72,vec3(.62,.65,.65),.58)+sunlight()*.035;
  gl_FragColor=vec4(tonemap(mist),alpha*vBreathAlpha);
}`;
}

export function createWhaleBreath(THREE, sharedUniforms = {}, lightingShader = "") {
  const ownsLargeField = !sharedUniforms.uFieldLarge,
    ownsSmallField = !sharedUniforms.uFieldSmall,
    ownsFineField = !sharedUniforms.uFieldFine,
    fallbackField =
      ownsLargeField || ownsSmallField || ownsFineField
        ? new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1)
        : null;
  if (fallbackField) fallbackField.needsUpdate = true;
  installUniform(THREE, sharedUniforms, "uTime", 0);
  installUniform(THREE, sharedUniforms, "uSpectral", 0);
  installUniform(THREE, sharedUniforms, "uCreaturePose", new THREE.Vector4(0, -2, 0, 0));
  installUniform(THREE, sharedUniforms, "uCreatureHeading", new THREE.Vector2(1, 0));
  installUniform(THREE, sharedUniforms, "uCreatureWake", 0);
  installUniform(THREE, sharedUniforms, "uCreatureBreath", 0);
  installUniform(THREE, sharedUniforms, "uCreatureBreathPhase", 0);
  installUniform(THREE, sharedUniforms, "uCreatureBuoyancy", null);
  installUniform(THREE, sharedUniforms, "uCreatureBuoyancyReady", 0);
  installUniform(THREE, sharedUniforms, "uCreatureVelocity", new THREE.Vector3());
  installUniform(THREE, sharedUniforms, "uFieldLarge", fallbackField);
  installUniform(THREE, sharedUniforms, "uFieldSmall", fallbackField);
  installUniform(THREE, sharedUniforms, "uFieldFine", fallbackField);

  const base = new THREE.PlaneGeometry(1, 1),
    geometry = new THREE.InstancedBufferGeometry();
  geometry.index = base.index;
  for (const [name, attribute] of Object.entries(base.attributes))
    geometry.setAttribute(name, attribute);
  geometry.setAttribute(
    "aBreathSeed",
    new THREE.InstancedBufferAttribute(makeSeedData(MIST_COUNT + DROPLET_COUNT), 4),
  );
  const kinds = new Float32Array(MIST_COUNT + DROPLET_COUNT);
  kinds.fill(1, MIST_COUNT);
  geometry.setAttribute("aBreathKind", new THREE.InstancedBufferAttribute(kinds, 1));
  geometry.instanceCount = MIST_COUNT + DROPLET_COUNT;
  base.dispose();

  const enabledUniform = { value: 1 },
    material = new THREE.ShaderMaterial({
      uniforms: { ...sharedUniforms, uBreathEnabled: enabledUniform },
      vertexShader: vertexShader(lightingShader),
      fragmentShader: fragmentShader(lightingShader),
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      toneMapped: false,
    }),
    object = new THREE.Mesh(geometry, material),
    state = { disposed: false };
  object.name = "Blue whale breath mist and droplets";
  object.frustumCulled = false;
  object.renderOrder = 4;

  return {
    object,
    update(time) {
      sharedUniforms.uTime.value = Number.isFinite(time) ? time : 0;
    },
    setEnabled(enabled) {
      enabledUniform.value = enabled ? 1 : 0;
      object.visible = Boolean(enabled);
    },
    dispose() {
      if (state.disposed) return;
      state.disposed = true;
      geometry.dispose();
      material.dispose();
      fallbackField?.dispose();
      object.removeFromParent();
    },
  };
}
