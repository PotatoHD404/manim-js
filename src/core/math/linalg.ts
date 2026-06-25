export type Vec = number[];
export type Mat = number[][];

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function scale(a: Vec, k: number): Vec {
  return a.map((v) => v * k);
}

export function sub(a: Vec, b: Vec): Vec {
  return a.map((v, i) => v - b[i]);
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a));
}

/** Column-wise mean of a set of row vectors. */
export function mean(rows: Vec[]): Vec {
  const d = rows[0].length;
  const m: Vec = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) m[j] += r[j];
  return m.map((v) => v / rows.length);
}

/** Subtract the column mean from every row. */
export function center(rows: Vec[]): { centered: Vec[]; mean: Vec } {
  const m = mean(rows);
  return { centered: rows.map((r) => sub(r, m)), mean: m };
}

/** Sample covariance `Σ = (1/n) Σ xᵢ xᵢᵀ` of already-centered rows. */
export function covariance(centered: Vec[]): Mat {
  const n = centered.length;
  const d = centered[0].length;
  const cov: Mat = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const x of centered) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        cov[i][j] += x[i] * x[j];
      }
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      const v = cov[i][j] / n;
      cov[i][j] = v;
      cov[j][i] = v;
    }
  }
  return cov;
}

export interface Eigen {
  /** Eigenvalues, sorted descending. */
  values: number[];
  /** Eigenvectors as rows, aligned with `values` (so `vectors[k]` is axis k). */
  vectors: Vec[];
}

/**
 * Jacobi eigenvalue algorithm for a real symmetric matrix. Converges for any
 * size the explorables need (2×2, 3×3, low-dim covariance). Returns eigenpairs
 * ordered by descending eigenvalue — i.e. principal axes first.
 */
export function jacobiEigen(symmetric: Mat, sweeps = 100, eps = 1e-12): Eigen {
  const n = symmetric.length;
  const a: Mat = symmetric.map((r) => r.slice());
  const v: Mat = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    }
    if (off < eps) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < eps) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        rotate(a, v, p, q, c, s, n);
      }
    }
  }

  const pairs = a
    .map((row, i) => ({ value: row[i], vector: v.map((vr) => vr[i]) }))
    .sort((x, y) => y.value - x.value);

  return {
    values: pairs.map((p) => p.value),
    vectors: pairs.map((p) => normalizeSign(p.vector)),
  };
}

function rotate(a: Mat, v: Mat, p: number, q: number, c: number, s: number, n: number): void {
  for (let i = 0; i < n; i++) {
    const aip = a[i][p];
    const aiq = a[i][q];
    a[i][p] = c * aip - s * aiq;
    a[i][q] = s * aip + c * aiq;
  }
  for (let i = 0; i < n; i++) {
    const api = a[p][i];
    const aqi = a[q][i];
    a[p][i] = c * api - s * aqi;
    a[q][i] = s * api + c * aqi;
  }
  for (let i = 0; i < n; i++) {
    const vip = v[i][p];
    const viq = v[i][q];
    v[i][p] = c * vip - s * viq;
    v[i][q] = s * vip + c * viq;
  }
}

/** Fix the sign so the largest-magnitude component is positive (stable axes). */
function normalizeSign(vec: Vec): Vec {
  let idx = 0;
  for (let i = 1; i < vec.length; i++) {
    if (Math.abs(vec[i]) > Math.abs(vec[idx])) idx = i;
  }
  return vec[idx] < 0 ? vec.map((x) => -x) : vec;
}

export interface Pca {
  mean: Vec;
  centered: Vec[];
  covariance: Mat;
  eigen: Eigen;
  /** Fraction of total variance carried by each axis. */
  explained: number[];
}

/** Full PCA of raw rows: center, covariance, eigendecomposition, explained variance. */
export function pca(rows: Vec[]): Pca {
  const { centered, mean: mu } = center(rows);
  const cov = covariance(centered);
  const eigen = jacobiEigen(cov);
  const total = eigen.values.reduce((a, b) => a + b, 0) || 1;
  return {
    mean: mu,
    centered,
    covariance: cov,
    eigen,
    explained: eigen.values.map((v) => v / total),
  };
}

/** Variance captured by projecting centered data onto a unit direction `u`. */
export function varianceAlong(cov: Mat, u: Vec): number {
  const d = u.length;
  let s = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) s += u[i] * cov[i][j] * u[j];
  }
  return s;
}
