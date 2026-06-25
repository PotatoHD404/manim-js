import { createMseVsBceGradient, type MseVsBceGradientApi } from "../ce/mse-vs-bce-gradient";

/**
 * `<ce-mse-vs-bce-gradient z="-6" target="1" zrange="8">` — drop-in element for
 * the "why not MSE?" gradient figure. All attributes are optional and default to
 * the post's example (positive target y = 1, z ∈ [−8, 8], starting confidently
 * wrong at z = −6).
 */
export class CeMseVsBceGradientElement extends HTMLElement {
  private api?: MseVsBceGradientApi;

  connectedCallback(): void {
    if (this.api) return;
    const t = num(this.getAttribute("target"));
    this.api = createMseVsBceGradient(this, {
      z: num(this.getAttribute("z")),
      zRange: num(this.getAttribute("zrange")),
      target: t === 0 ? 0 : t === 1 ? 1 : undefined,
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
