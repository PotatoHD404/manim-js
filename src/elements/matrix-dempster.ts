import { createMatrixDempster, type DempsterApi } from "../matrix/dempster";

/**
 * `<matrix-dempster dim="6" rho="0.6" entry="1,4">` — drop-in element for the
 * Dempster covariance-selection figure: sweep an unobserved entry and watch the
 * entropy `log₂ det M` peak exactly where the precision entry `(M⁻¹)₁,₄` vanishes.
 */
export class MatrixDempsterElement extends HTMLElement {
  private api?: DempsterApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMatrixDempster(this, {
      dim: num(this.getAttribute("dim")),
      rho: num(this.getAttribute("rho")),
      entry: entryPair(this.getAttribute("entry")),
      start: num(this.getAttribute("start")),
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

function entryPair(v: string | null): [number, number] | undefined {
  if (!v) return undefined;
  const parts = v.split(/[,\s]+/).map((s) => Number(s.trim()));
  if (parts.length === 2 && parts.every((n) => Number.isFinite(n))) return [parts[0], parts[1]];
  return undefined;
}
