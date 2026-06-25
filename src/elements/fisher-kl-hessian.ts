import { createFisherKlHessian, type FisherKlHessianApi } from "../fisher/kl-hessian";

/**
 * `<fisher-kl-hessian p0="0.35" mu0="0" v0="1" delta-max="0.22">` — drop-in
 * element for the "Fisher is the Hessian of the KL divergence" figure: the
 * Bernoulli exact KL against its Fisher parabola ½ I(p₀) δ² (left), and the
 * Gaussian KL contours against the Fisher-ellipse contours ½ δᵀ I δ (right),
 * coinciding near the truth and peeling apart outward.
 */
export class FisherKlHessianElement extends HTMLElement {
  private api?: FisherKlHessianApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherKlHessian(this, {
      p0: num(this.getAttribute("p0")),
      mu0: num(this.getAttribute("mu0")),
      v0: num(this.getAttribute("v0")),
      deltaMax: num(this.getAttribute("delta-max")),
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
