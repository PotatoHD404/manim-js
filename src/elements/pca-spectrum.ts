import { createSpectrum, type SpectrumApi } from "../pca/spectrum";

/**
 * `<pca-spectrum view="scree|reconstruction" threshold="0.95">` — drop-in
 * element for the scree plot and reconstruction-error figures.
 */
export class PcaSpectrumElement extends HTMLElement {
  private api?: SpectrumApi;

  connectedCallback(): void {
    if (this.api) return;
    const view = this.getAttribute("view");
    this.api = createSpectrum(this, {
      view: view === "reconstruction" ? "reconstruction" : "scree",
      threshold: num(this.getAttribute("threshold")),
      q: num(this.getAttribute("q")),
      spectrum: jsonNums(this.getAttribute("spectrum")),
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

function jsonNums(v: string | null): number[] | undefined {
  if (!v) return undefined;
  try {
    const arr = JSON.parse(v);
    return Array.isArray(arr) && arr.every((x) => typeof x === "number") ? arr : undefined;
  } catch {
    return undefined;
  }
}
