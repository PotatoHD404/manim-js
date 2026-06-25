import { createFisherMatrixEllipse, type FisherMatrixEllipseApi } from "../fisher/matrix-ellipse";

/**
 * `<fisher-matrix-ellipse i11="0.25" i22="0.03125" i12="0" n="200" c="2.9957">` —
 * drop-in element for the matrix-case figure: the 2×2 Fisher information as the
 * quadratic form ½δᵀIδ = c, drawn as a gold confidence ellipse with eigen-axis
 * arrows over the seeded asymptotic-MLE scatter (covariance I⁻¹/n). Sliders set
 * the matrix entries; a draggable cursor / arrow keys grow the level c; live
 * eigenvalue readouts track the matrix.
 */
export class FisherMatrixEllipseElement extends HTMLElement {
  private api?: FisherMatrixEllipseApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createFisherMatrixEllipse(this, {
      i11: num(this.getAttribute("i11")),
      i22: num(this.getAttribute("i22")),
      i12: num(this.getAttribute("i12")),
      n: num(this.getAttribute("n")),
      c: num(this.getAttribute("c")),
      points: num(this.getAttribute("points")),
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
