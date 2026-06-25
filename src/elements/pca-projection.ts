import { createProjection, type ProjectionApi } from "../pca/projection";

/**
 * `<pca-projection seed="42" points="200" angle="15">` — drop-in custom element.
 * Numeric attributes are optional; sensible defaults match the published demo.
 */
export class PcaProjectionElement extends HTMLElement {
  private api?: ProjectionApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createProjection(this, {
      seed: num(this.getAttribute("seed")),
      points: num(this.getAttribute("points")),
      angle: num(this.getAttribute("angle")),
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
