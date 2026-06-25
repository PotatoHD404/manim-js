import { createSymbolError, type SymbolErrorApi } from "../matrix/symbol-error";

/**
 * `<matrix-symbol-error p="0.25" ks="[[4,1],[8,2],[16,4],[32,8]]" trials="220">`
 * — the operational-decoder figure: expected GF(2) symbol-error rate versus the
 * observation rate p, with the converse-threshold pivot at p★ = r/k.
 */
export class MatrixSymbolErrorElement extends HTMLElement {
  private api?: SymbolErrorApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createSymbolError(this, {
      p: num(this.getAttribute("p")),
      ks: pairs(this.getAttribute("ks")),
      trials: num(this.getAttribute("trials")),
      generators: num(this.getAttribute("generators")),
      grid: num(this.getAttribute("grid")),
      seed: num(this.getAttribute("seed")),
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

function pairs(v: string | null): Array<[number, number]> | undefined {
  if (!v) return undefined;
  try {
    const arr = JSON.parse(v);
    if (
      Array.isArray(arr) &&
      arr.length > 0 &&
      arr.every(
        (x) => Array.isArray(x) && x.length === 2 && typeof x[0] === "number" && typeof x[1] === "number",
      )
    ) {
      return arr.map((x) => [x[0], x[1]] as [number, number]);
    }
    return undefined;
  } catch {
    return undefined;
  }
}
