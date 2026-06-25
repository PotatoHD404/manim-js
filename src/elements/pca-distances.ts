import { createDistances, type DistancesApi } from "../pca/distances";

/**
 * `<pca-distances points="120" tilt="0" spectrum="[6,3,0.15]">` — drop-in
 * element for the metric-MDS distance-preservation figure: tilt the projection
 * plane off the principal plane and watch the original-vs-projected distance
 * scatter peel off the identity line as the correlation falls.
 */
export class PcaDistancesElement extends HTMLElement {
  private api?: DistancesApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createDistances(this, {
      points: num(this.getAttribute("points")),
      seed: num(this.getAttribute("seed")),
      tilt: num(this.getAttribute("tilt")),
      spectrum: triple(this.getAttribute("spectrum")),
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

function triple(v: string | null): [number, number, number] | undefined {
  if (!v) return undefined;
  try {
    const arr = JSON.parse(v);
    return Array.isArray(arr) && arr.length === 3 && arr.every((x) => typeof x === "number")
      ? [arr[0], arr[1], arr[2]]
      : undefined;
  } catch {
    return undefined;
  }
}
