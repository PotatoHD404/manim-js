import { createMseVsBceLoss, type MseVsBceLossApi } from "../ce/mse-vs-bce-loss";

/**
 * `<ce-mse-vs-bce-loss label="1" space="prob|logit" prob="0.18">` — drop-in
 * element for the "why not MSE?" loss-shape figure: cross-entropy −log p versus
 * squared error (p−1)² over the predicted probability, with a movable read-off
 * point.
 */
export class CeMseVsBceLossElement extends HTMLElement {
  private api?: MseVsBceLossApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createMseVsBceLoss(this, {
      label: this.getAttribute("label") === "0" ? 0 : 1,
      space: this.getAttribute("space") === "logit" ? "logit" : "prob",
      prob: num(this.getAttribute("prob")),
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
