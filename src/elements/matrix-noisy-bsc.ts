import { createMatrixNoisyBsc, type NoisyBscApi } from "../matrix/noisy-bsc";

/**
 * `<matrix-noisy-bsc k="6" r="2" delta="0.1" seed="0">` — drop-in element for the
 * binary-symmetric-channel capacity figure: C(δ) = k(1 − h(δ))/r over the
 * crossover probability δ.
 */
export class MatrixNoisyBscElement extends HTMLElement {
  private api?: NoisyBscApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixNoisyBsc(this, {
      k: num(this.getAttribute("k")),
      r: num(this.getAttribute("r")),
      delta: num(this.getAttribute("delta")),
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
