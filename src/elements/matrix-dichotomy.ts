import { createMatrixDichotomy, type DichotomyApi } from "../matrix/dichotomy";

/**
 * `<matrix-dichotomy k="16" k-min="2" k-max="128">` — drop-in element for the
 * structure-dichotomy figure: completion capacity C = N/dim𝓕 versus matrix size
 * k, with the families splitting about the C = 2 boundary into linear-redundancy
 * (diverging) and involutive (bounded) regimes.
 */
export class MatrixDichotomyElement extends HTMLElement {
  private api?: DichotomyApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixDichotomy(this, {
      k: num(this.getAttribute("k")),
      kMin: num(this.getAttribute("k-min")),
      kMax: num(this.getAttribute("k-max")),
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
