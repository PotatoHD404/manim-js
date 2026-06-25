import { el } from "./dom";
import { ROMAN_FONT } from "./draw";
import { tex } from "./katex";
import type { Theme } from "./theme";

export interface LiveEquation {
  node: HTMLElement;
  /** Update only the trailing value (no KaTeX re-render, cheap per frame). */
  set(value: string): void;
}

/**
 * A scene label: a KaTeX-rendered static formula followed by a live value, e.g.
 * `Var(u) = uᵀΣu = ` · `3.42`. Only the value span is rewritten per frame, so
 * this is cheap to animate.
 */
export function liveEquation(formulaTex: string, theme: Theme, valueColor?: string): LiveEquation {
  const node = el("div", {
    style: `display:flex;align-items:baseline;gap:8px;color:${theme.fg};font-size:15px;line-height:1.9;`,
  });
  const lhs = el("span");
  lhs.innerHTML = tex(formulaTex);
  const value = el("span", {
    style: `font:16px ${ROMAN_FONT};color:${valueColor ?? theme.fg};min-width:3ch;`,
  });
  node.append(lhs, value);
  return { node, set: (v) => (value.textContent = v) };
}

/** A static KaTeX line (no live value). */
export function staticEquation(formulaTex: string, theme: Theme, color?: string): HTMLElement {
  const node = el("div", {
    style: `color:${color ?? theme.muted};font-size:14px;line-height:1.9;`,
  });
  node.innerHTML = tex(formulaTex);
  return node;
}

/** A floating overlay container pinned to a corner of a scene panel. */
export function overlayLayer(corner: "tl" | "tr" | "bl" = "tl"): HTMLElement {
  const pos =
    corner === "tl"
      ? "top:12px;left:14px;"
      : corner === "tr"
        ? "top:12px;right:14px;text-align:right;"
        : "bottom:12px;left:14px;";
  return el("div", {
    style: `position:absolute;${pos}pointer-events:none;z-index:2;`,
  });
}
