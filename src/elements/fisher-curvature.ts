import { createFisherCurvature, type FisherCurvatureApi } from "../fisher/curvature";

/**
 * `<fisher-curvature n="6" min-n="3" max-n="120" lambda-true="1" seed="3">` —
 * drop-in element for the observed-information curvature figure: as n grows the
 * exponential log-likelihood's peak sharpens, the second-order Taylor parabola
 * tightens, and the osculating circle of radius r = 1/J shrinks.
 */
export class FisherCurvatureElement extends HTMLElement {
  private api?: FisherCurvatureApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherCurvature(this, {
      n: num(this.getAttribute("n")),
      minN: num(this.getAttribute("min-n")),
      maxN: num(this.getAttribute("max-n")),
      lambdaTrue: num(this.getAttribute("lambda-true")),
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
