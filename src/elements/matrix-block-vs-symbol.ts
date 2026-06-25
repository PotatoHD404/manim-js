import { type BlockVsSymbolApi, createBlockVsSymbol } from "../matrix/block-vs-symbol";

/**
 * `<matrix-block-vs-symbol n="12" k="8" p="0.35" derive="true">` — drop-in
 * element for the block-versus-symbol amplification figure: a fixed per-column
 * (symbol) failure q, derived from the seeded GF(2) decoder for (k, p), amplifies
 * to a whole-matrix block error 1 − (1−q)^n that climbs to 1 as the block widens.
 */
export class MatrixBlockVsSymbolElement extends HTMLElement {
  private api?: BlockVsSymbolApi;

  connectedCallback(): void {
    if (this.api) return;
    const derive = this.getAttribute("derive");
    this.api = createBlockVsSymbol(this, {
      q: num(this.getAttribute("q")),
      n: num(this.getAttribute("n")),
      k: num(this.getAttribute("k")),
      p: num(this.getAttribute("p")),
      seed: num(this.getAttribute("seed")),
      deriveFromDecoder: derive === null ? undefined : derive !== "false",
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
