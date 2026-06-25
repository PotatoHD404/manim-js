import { createSpectralFlatness, type FlatnessModel, type SpectralFlatnessApi } from "../matrix/spectral-flatness";

/**
 * `<matrix-spectral-flatness model="equi|ar1" size="8" rho="0.6">` — drop-in
 * element for the spectral-flatness figure. All attributes are optional and
 * default to the post's example (equicorrelated, k = 8, ρ = 0.6).
 */
export class MatrixSpectralFlatnessElement extends HTMLElement {
  private api?: SpectralFlatnessApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createSpectralFlatness(this, {
      model: model(this.getAttribute("model")),
      size: num(this.getAttribute("size")),
      rho: num(this.getAttribute("rho")),
    });
  }

  disconnectedCallback(): void {
    this.api?.destroy();
    this.api = undefined;
  }
}

function model(v: string | null): FlatnessModel | undefined {
  return v === "ar1" ? "ar1" : v === "equi" ? "equi" : undefined;
}

function num(v: string | null): number | undefined {
  if (v === null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
