import { createFisherCrb, type FisherCrbApi } from "../fisher/crb";

/**
 * `<fisher-crb lambda="3" n="400" seed="11">` — drop-in element for the
 * Cramér–Rao / asymptotic-normality figure: the empirical variance of the
 * Poisson MLE riding on the floor 1/(n I₁), and the rescaled error √n(λ̂−λ)
 * collapsing onto 𝒩(0, 1/I₁) as n grows. Both panels are driven by live,
 * seeded in-browser Poisson sampling.
 */
export class FisherCrbElement extends HTMLElement {
  private api?: FisherCrbApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherCrb(this, {
      lambda: num(this.getAttribute("lambda")),
      n: num(this.getAttribute("n")),
      seed: num(this.getAttribute("seed")),
      reps: num(this.getAttribute("reps")),
      histReps: num(this.getAttribute("hist-reps")),
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
