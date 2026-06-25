import { createZeroCount, type ZeroCountApi } from "../mle/zero-count";

/**
 * `<mle-zero-count counts="[8,5,0,3]" labels='["A","B","C","D"]' alpha="2">` —
 * the categorical zero-count figure: MLE (zero on the unseen category) versus
 * the Dirichlet/Laplace-smoothed MAP estimate.
 */
export class MleZeroCountElement extends HTMLElement {
  private api?: ZeroCountApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createZeroCount(this, {
      counts: jsonNums(this.getAttribute("counts")),
      labels: jsonStrings(this.getAttribute("labels")),
      alpha: num(this.getAttribute("alpha")),
      maxCount: num(this.getAttribute("max-count")),
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

function jsonStrings(v: string | null): string[] | undefined {
  if (!v) return undefined;
  try {
    const arr = JSON.parse(v);
    return Array.isArray(arr) && arr.every((x) => typeof x === "string") ? arr : undefined;
  } catch {
    return undefined;
  }
}
