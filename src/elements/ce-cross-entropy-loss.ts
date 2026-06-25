import { createCrossEntropyLoss, type CrossEntropyLossApi } from "../ce/cross-entropy-loss";

/**
 * `<ce-cross-entropy-loss p="0.9" loss-max="5">` — drop-in element for the
 * per-example cross-entropy loss curve −log q(y): drag the predicted probability
 * and read the loss off the curve.
 */
export class CeCrossEntropyLossElement extends HTMLElement {
  private api?: CrossEntropyLossApi;

  connectedCallback(): void {
    if (this.api) return;
    this.api = createCrossEntropyLoss(this, {
      p: num(this.getAttribute("p")),
      lossMax: num(this.getAttribute("loss-max")),
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
