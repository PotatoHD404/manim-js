import { createGaussianShrinkage, type GaussianShrinkageApi } from "../mle/gaussian-shrinkage";

/**
 * `<mle-gaussian-shrinkage prior-mean="-1.4" prior-var="1" sample-mean="1.8" sigma2="1" n="4">`
 * — drop-in element for the Normal-Normal shrinkage figure: the MAP estimate of
 * the mean as a precision-weighted blend of the prior mean and the sample mean.
 */
export class MleGaussianShrinkageElement extends HTMLElement {
  private api?: GaussianShrinkageApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createGaussianShrinkage(this, {
      priorMean: num(this.getAttribute("prior-mean")),
      priorVar: num(this.getAttribute("prior-var")),
      sampleMean: num(this.getAttribute("sample-mean")),
      sigma2: num(this.getAttribute("sigma2")),
      n: num(this.getAttribute("n")),
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
