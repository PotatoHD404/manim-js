import { createMatrixEquicorrelated, type EquicorrelatedApi } from "../matrix/equicorrelated";

/**
 * `<matrix-equicorrelated k="8" rho="0.6" delta="0.01">` — drop-in element for
 * the equicorrelated-covariance figure (heatmap + eigenvalue spectrum + the
 * quantized completion capacity and effective rank). All attributes are
 * optional and default to the post's example.
 */
export class MatrixEquicorrelatedElement extends HTMLElement {
  private api?: EquicorrelatedApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixEquicorrelated(this, {
      k: num(this.getAttribute("k")),
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
