import { createFisherTwoForms, type FisherTwoFormsApi } from "../fisher/two-forms";

/**
 * `<fisher-two-forms lambda="3" seed="7" n="1000">` — drop-in element for the
 * "two faces of one matrix" figure: the score-variance and negative-Hessian
 * Monte-Carlo estimators of the Poisson Fisher information converging to 1/λ.
 */
export class FisherTwoFormsElement extends HTMLElement {
  private api?: FisherTwoFormsApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherTwoForms(this, {
      lambda: num(this.getAttribute("lambda")),
      seed: num(this.getAttribute("seed")),
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
