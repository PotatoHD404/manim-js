import { createFisherReparam, type FisherReparamApi } from "../fisher/reparam";

/**
 * `<fisher-reparam p="0.30">` — drop-in element for the "a tensor, not a number"
 * figure: Bernoulli Fisher information in the probability coordinate vs the
 * natural (logit) parameter, with the squared-Jacobian rescaling law shown live.
 * The attribute is optional and defaults to the post's worked point p = 0.30.
 */
export class FisherReparamElement extends HTMLElement {
  private api?: FisherReparamApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherReparam(this, {
      p: num(this.getAttribute("p")),
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
