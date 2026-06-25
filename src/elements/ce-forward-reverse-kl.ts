import { createForwardReverseKl, type ForwardReverseKlApi } from "../ce/forward-reverse-kl";

/**
 * `<ce-forward-reverse-kl mu="0" sigma="2" seed="0" grid="2000">` — drop-in
 * element for the forward-vs-reverse-KL figure: a fixed bimodal target p and a
 * draggable single Gaussian q, with live forward KL(p‖q) (mode-covering) and
 * reverse KL(q‖p) (mode-seeking) and snap buttons onto each optimum.
 */
export class CeForwardReverseKlElement extends HTMLElement {
  private api?: ForwardReverseKlApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createForwardReverseKl(this, {
      mu: num(this.getAttribute("mu")),
      sigma: num(this.getAttribute("sigma")),
      seed: num(this.getAttribute("seed")),
      grid: num(this.getAttribute("grid")),
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
