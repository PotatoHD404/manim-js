import { type CeDecompositionApi, createCeDecomposition } from "../ce/ce-decomposition";

/**
 * `<ce-decomposition p="0.7" q="0.4">` — drop-in element for the
 * cross-entropy = entropy + KL decomposition figure. All attributes are optional
 * and default to the post's example (Bernoulli p = 0.7, cursor starting at q = p).
 */
export class CeDecompositionElement extends HTMLElement {
  private api?: CeDecompositionApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createCeDecomposition(this, {
      p: num(this.getAttribute("p")),
      q: num(this.getAttribute("q")),
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
