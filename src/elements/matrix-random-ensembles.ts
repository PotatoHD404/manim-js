import { createMatrixRandomEnsembles, type RandomEnsemblesApi } from "../matrix/random-ensembles";

/**
 * `<matrix-random-ensembles k="4" alpha="1.0" samples="4000" seed="12345">` —
 * drop-in element for the random-ensembles figure: a histogram of the completion
 * capacity C = S / H_joint over random Dirichlet joints on {0,1}^k.
 */
export class MatrixRandomEnsemblesElement extends HTMLElement {
  private api?: RandomEnsemblesApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixRandomEnsembles(this, {
      k: num(this.getAttribute("k")),
      alpha: num(this.getAttribute("alpha")),
      samples: num(this.getAttribute("samples")),
      seed: num(this.getAttribute("seed")),
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
