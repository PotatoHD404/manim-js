import { type CommonBitApi, createCommonBit } from "../matrix/common-bit";

/**
 * `<matrix-common-bit ks="[2,3,4,6]" active="4" q="0.1">` — drop-in element for
 * the common-bit completion-capacity figure. All attributes are optional and
 * default to the post's example (k ∈ {2,3,4,6}, active k = 4).
 */
export class MatrixCommonBitElement extends HTMLElement {
  private api?: CommonBitApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createCommonBit(this, {
      ks: jsonNums(this.getAttribute("ks")),
      active: num(this.getAttribute("active")),
      q: num(this.getAttribute("q")),
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
