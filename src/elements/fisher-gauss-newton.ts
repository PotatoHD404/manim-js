import { createFisherGaussNewton, type GaussNewtonApi } from "../fisher/gauss-newton";

/**
 * `<fisher-gauss-newton seed="2" points="200" steps="15">` — drop-in element for
 * the "Fisher matrix as a metric" figure: vanilla gradient descent racing a
 * Fisher-preconditioned Gauss-Newton (natural-gradient) path down a nonlinear
 * least-squares loss surface, with the post's ‖H−Fisher‖/‖H‖ gap inset.
 */
export class FisherGaussNewtonElement extends HTMLElement {
  private api?: GaussNewtonApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherGaussNewton(this, {
      seed: num(this.getAttribute("seed")),
      points: num(this.getAttribute("points")),
      sigma: num(this.getAttribute("sigma")),
      damping: num(this.getAttribute("damping")),
      lr: num(this.getAttribute("lr")),
      steps: num(this.getAttribute("steps")),
      iter: num(this.getAttribute("iter")),
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
