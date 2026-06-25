import type { Mat, Vec } from "./linalg";

/**
 * Deterministic linear-congruential RNG with a Box–Muller normal generator.
 * Seeded so every render of an explorable shows the same cloud.
 */
export class Rng {
  private state: number;

  constructor(seed = 1) {
    this.state = seed >>> 0 || 1;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 4294967296;
  }

  /** Standard normal sample. */
  gauss(): number {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** `n` independent standard-normal samples. */
  gaussVec(n: number): Vec {
    const out: Vec = [];
    for (let i = 0; i < n; i++) out.push(this.gauss());
    return out;
  }
}

/**
 * Sample `n` points from a correlated Gaussian by pushing white noise through a
 * mixing matrix: `x = A z`. The resulting covariance is `A Aᵀ`, which lets the
 * caller dial in the exact principal structure they want to teach.
 */
export function correlatedCloud(rng: Rng, n: number, mix: Mat): Vec[] {
  const rows = mix.length;
  const cols = mix[0].length;
  const pts: Vec[] = [];
  for (let i = 0; i < n; i++) {
    const z = rng.gaussVec(cols);
    const x: Vec = new Array(rows).fill(0);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let c = 0; c < cols; c++) s += mix[r][c] * z[c];
      x[r] = s;
    }
    pts.push(x);
  }
  return pts;
}
