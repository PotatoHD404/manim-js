import { createEigenwarp, type EigenwarpApi } from "../pca/eigenwarp";

/**
 * `<pca-eigenwarp seed="11" points="240">` — the covariance-as-transformation
 * eigenvector reveal.
 */
export class PcaEigenwarpElement extends HTMLElement {
  private api?: EigenwarpApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createEigenwarp(this, {
      seed: num(this.getAttribute("seed")),
      points: num(this.getAttribute("points")),
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
