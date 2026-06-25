import { type BetaPosteriorApi, createBetaPosterior } from "../mle/beta-posterior";

/**
 * `<mle-beta-bernoulli alpha="2" beta="2" heads="7" tails="3">` — drop-in
 * element for the Beta-Bernoulli MLE/MAP figure. All attributes are optional and
 * default to the post's example (k=7, n=10, Beta(2,2)).
 */
export class MleBetaBernoulliElement extends HTMLElement {
  private api?: BetaPosteriorApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createBetaPosterior(this, {
      alpha: num(this.getAttribute("alpha")),
      beta: num(this.getAttribute("beta")),
      heads: num(this.getAttribute("heads")),
      tails: num(this.getAttribute("tails")),
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
