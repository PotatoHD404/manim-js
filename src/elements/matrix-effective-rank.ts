import { createMatrixEffectiveRank, type EffectiveRankApi } from "../matrix/effective-rank";

/**
 * `<matrix-effective-rank k="12" rho="0.6" model="equicorr|ar1">` — drop-in
 * element for the effective-rank figure: a covariance spectrum normalized to a
 * probability distribution, its effective rank exp(H(p)) marked against the
 * nominal rank k, and the effective rank traced across the correlation ρ. All
 * attributes are optional and default to the post's example.
 */
export class MatrixEffectiveRankElement extends HTMLElement {
  private api?: EffectiveRankApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixEffectiveRank(this, {
      k: num(this.getAttribute("k")),
      rho: num(this.getAttribute("rho")),
      model: this.getAttribute("model") === "ar1" ? "ar1" : "equicorr",
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
