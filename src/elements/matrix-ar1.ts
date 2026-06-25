import { createMatrixAr1, type MatrixAr1Api } from "../matrix/ar1";

/**
 * `<matrix-ar1 size="8" rho="0.6" delta="0.6">` — drop-in element for the AR(1)
 * covariance figure: the banded Toeplitz matrix Σ_ij = ρ^|i-j|, its eigenvalue
 * spectrum, and the quantized-Gaussian completion capacity C(ρ). All attributes
 * are optional and default to the post's example (k=8, ρ=0.6, Δ=0.6).
 */
export class MatrixAr1Element extends HTMLElement {
  private api?: MatrixAr1Api;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixAr1(this, {
      size: num(this.getAttribute("size")),
      rho: num(this.getAttribute("rho")),
      delta: num(this.getAttribute("delta")),
    });
  }

  disconnectedCallback(): void {
    this.api?.destroy();
    this.api = undefined;
  }
}

function num(v: string | null): number | undefined {
  if (v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
