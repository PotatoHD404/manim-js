import { createRankQuantization, type RankQuantizationApi } from "../matrix/rank-quantization";

/**
 * `<matrix-rank-quantization k="6" ranks="[1,2,3]" rho="0.6" delta="0.1">` —
 * drop-in element for the quantized-Gaussian capacity figure: completion
 * capacity C = S/H_joint versus observation resolution Δ, with full-rank
 * (C → 1) and exactly rank-r (C → k/r) covariances.
 */
export class MatrixRankQuantizationElement extends HTMLElement {
  private api?: RankQuantizationApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createRankQuantization(this, {
      k: num(this.getAttribute("k")),
      ranks: jsonNums(this.getAttribute("ranks")),
      rho: num(this.getAttribute("rho")),
      seed: num(this.getAttribute("seed")),
      delta: num(this.getAttribute("delta")),
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
