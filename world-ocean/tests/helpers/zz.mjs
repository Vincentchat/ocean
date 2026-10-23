import v8 from 'node:v8';
import { createOcean } from '../../index.ts';
import { createFlatShape, createStubWorld } from '../../stubs/impl.ts';
const gc = globalThis.gc;
const shape = createFlatShape();
const world = createStubWorld({ shape });
const ocean = createOcean({ preset: 'windy', worker: false }); world.use(ocean); world.start();
for (let i = 0; i < 6000; i++) world.stepOnce();
const ns = () => v8.getHeapSpaceStatistics().find((s) => s.space_name === 'new_space').space_used_size;
const empty = createStubWorld({ shape });
function per(n, fn) { for (let i = 0; i < n; i++) fn(); gc(); const a = ns(); for (let i = 0; i < n; i++) fn(); return (ns() - a) / n; }
console.log('tick', (per(600, () => world.stepOnce()) - per(600, () => empty.stepOnce())).toFixed(2), 'B');
