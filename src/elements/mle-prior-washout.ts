import { createPriorWashout, type PriorWashoutApi } from "../mle/prior-washout";

/**
 * `<mle-prior-washout alpha="2" beta="6" truth="0.7" max-n="500" n="5">` —
 * drop-in element for the prior-washout figure: scrub the sample size and watch
 * the Beta posterior concentrate onto the truth as MAP converges to the MLE.
 */
export class MlePriorWashoutElement extends HTMLElement {
  private api?: PriorWashoutApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createPriorWashout(this, {
      alpha: num(this.getAttribute("alpha")),
      beta: num(this.getAttribute("beta")),
      truth: num(this.getAttribute("truth")),
      maxN: num(this.getAttribute("max-n")),
      n: num(this.getAttribute("n")),
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
