import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface CrossEntropyLossOptions {
  /** Initial predicted probability of the true class, p ∈ (0,1]. Default 0.9. */
  p?: number;
  /** Top of the loss axis; the curve is clamped here as p→0 (the post uses 5). */
  lossMax?: number;
  theme?: Partial<Theme>;
}

export interface CrossEntropyLossApi {
  /** Move the read-off cursor to a predicted probability p (eased). */
  setP(p: number, animate?: boolean): void;
  destroy(): void;
}

/** A reference point on the curve, as printed in the post. */
interface Marker {
  p: number;
  label: string;
}

/**
 * Scene — "the loss punishes confident wrong predictions". Over the predicted
 * probability of the true class p ∈ (0,1], the per-example cross-entropy loss is
 * the single curve −log p (natural log): it is ≈0 when the model is confident
 * and right, and diverges as p→0. Drag the cream cursor along p and read the
 * loss −log p off the curve; two gold reference points reproduce the post's
 * numbers (p=0.9 → 0.11, p=0.01 → 4.61), the whole point being that the penalty
 * for a confident mistake is enormous.
 *
 * Ported from the post's notebook cell verbatim: `-np.log(pp)` over
 * `pp = np.linspace(1e-3, 1-1e-3, 600)`, with the loss axis clamped to [0, 5].
 */
export function createCrossEntropyLoss(
  target: HTMLElement,
  options: CrossEntropyLossOptions = {},
): CrossEntropyLossApi {
  const theme = withTheme(options.theme);
  const lossMax = options.lossMax ?? 5;

  // The loss is unbounded as p→0; the smallest probability the post plots is
  // 1e-3, matching np.linspace(1e-3, 1-1e-3, 600). Cursor p lives in [pMin, 1].
  const pMin = 1e-3;
  const pMax = 1 - 1e-3;

  // The curve grid, exactly the notebook's 600 points on the open interval.
  const G = 600;
  const grid = new Float64Array(G);
  const loss = new Float64Array(G);
  for (let i = 0; i < G; i++) {
    const p = pMin + (pMax - pMin) * (i / (G - 1));
    grid[i] = p;
    loss[i] = -Math.log(p); // natural log — per-example cross-entropy −log q(y)
  }

  // The two reference predictions the post calls out.
  const markers: Marker[] = [
    { p: 0.9, label: "0.9" },
    { p: 0.01, label: "0.01" },
  ];

  // The per-example cross-entropy loss at a probability p (the figure's value).
  const ceLoss = (p: number): number => -Math.log(clamp(p, pMin, pMax));

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the live loss read off at the cursor
  const overlay = overlayLayer("tl");
  const eqLoss = liveEquation("L=-\\log q(y)\\;=", theme, theme.green);
  const eqP = liveEquation("q(y)\\;=", theme, theme.cream);
  overlay.append(eqLoss.node, eqP.node);
  panel.append(overlay);

  // top-right: the definition, restated
  const defOverlay = overlayLayer("tr");
  defOverlay.append(staticEquation("H(p,q)=-\\sum_k p(k)\\log q(k)=-\\log q(y)", theme, theme.muted));
  panel.append(defOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "predicted prob  q(y)";
  const slider = el("input", {
    type: "range",
    min: "1",
    max: "999",
    step: "1",
    value: String(Math.round(clamp(options.p ?? 0.9, pMin, pMax) * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "predicted probability of the true class",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag the curve · ←→ scrub";
  controls.append(label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the read-off cursor over p --------------------
  let cursor = clamp(options.p ?? 0.9, pMin, pMax);
  let destroyed = false;
  const cursorTracker = valueTracker(cursor, (v) => {
    cursor = clamp(v, pMin, pMax);
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    draw();
    eqLoss.set(ceLoss(cursor).toFixed(2));
    eqP.set(cursor.toFixed(3));
    if (document.activeElement !== slider) slider.value = String(Math.round(cursor * 1000));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 40;
    const mr = 18;
    const mt = 16;
    const mb = 32;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const x = (p: number) => ml + clamp(p, 0, 1) * plotW;
    const y = (l: number) => mt + plotH - (clamp(l, 0, lossMax) / lossMax) * plotH;

    // faint number plane in BLUE_D, receding: vertical lines at probabilities,
    // horizontal lines at integer loss values.
    for (let g = 0; g <= 1.0001; g += 0.2) {
      strokeLine(ctx, [x(g), mt], [x(g), mt + plotH], theme.grid, 1, g === 0 ? 0.28 : 0.16);
    }
    for (let l = 0; l <= lossMax + 1e-6; l += 1) {
      strokeLine(ctx, [ml, y(l)], [W - mr, y(l)], theme.grid, 1, l === 0 ? 0.28 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    // the loss curve −log p (the data object: bright BLUE_C, faintly glowing).
    // Built from the same 600-point grid as the notebook; clamped at lossMax so
    // the divergence as p→0 reads as a wall at the top of the frame.
    const curve: Pt[] = new Array(G);
    for (let i = 0; i < G; i++) curve[i] = [x(grid[i]), y(loss[i])];
    glow(ctx, theme.blue, 6, () => {
      polyline(ctx, curve, { color: theme.blue, width: 2.6, alpha: 1 });
    });

    // gold reference points — the post's two callouts. The "answer" colour.
    for (const m of markers) {
      const px = x(m.p);
      const l = ceLoss(m.p);
      const py = y(l);
      strokeLine(ctx, [px, py], [px, mt + plotH], theme.gold, 1, 0.45);
      glow(ctx, theme.gold, 5, () => disc(ctx, [px, py], 3.4, theme.gold, 1));
      const right = m.p < 0.6;
      mathLabel(ctx, "L", [px + (right ? 9 : -9), py + (l > lossMax - 0.7 ? 16 : -7)], {
        color: theme.gold,
        size: 13,
        align: right ? "left" : "right",
        glow: 4,
      });
      ctx.fillStyle = theme.gold;
      ctx.font = `11px ${ROMAN_FONT}`;
      ctx.textAlign = right ? "left" : "right";
      ctx.textBaseline = l > lossMax - 0.7 ? "top" : "bottom";
      ctx.fillText(`${m.label} → ${l.toFixed(2)}`, px + (right ? 9 : -9), py + (l > lossMax - 0.7 ? 30 : -22));
    }

    // the read-off cursor (the movable variable): a vertical rule, the dropped
    // GREEN trace of the loss value, and a glowing cream dot on the curve.
    const cx = x(cursor);
    const cl = ceLoss(cursor);
    const cy = y(cl);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.5);
    // GREEN traced quantity: the loss height, drawn from the axis to the curve.
    glow(ctx, theme.green, 5, () => strokeLine(ctx, [ml, cy], [cx, cy], theme.green, 2, 0.85));
    disc(ctx, [ml, cy], 3, theme.green, 1);
    glow(ctx, theme.cream, 8, () => disc(ctx, [cx, cy], 4.4, theme.cream, 1));
    mathLabel(ctx, "q", [cx + (cursor < 0.85 ? 7 : -7), mt + 14], {
      color: theme.cream,
      size: 15,
      sub: "y",
      align: cursor < 0.85 ? "left" : "right",
      glow: 5,
    });

    // x ticks (predicted probability) and y ticks (loss), roman numerals.
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = 0; g <= 1.0001; g += 0.2) ctx.fillText(g.toFixed(1), x(g), mt + plotH + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let l = 0; l <= lossMax + 1e-6; l += 1) ctx.fillText(String(l), ml - 6, y(l));

    // axis titles, KaTeX serif
    mathLabel(ctx, "q", [W - mr, mt + plotH + 26], { color: theme.tick, size: 15, sub: "y", align: "right" });
    mathLabel(ctx, "L", [ml - 26, mt + 6], { color: theme.tick, size: 15, align: "left" });
  }

  // --- interaction ---------------------------------------------------------
  function setP(p: number, animate = false): void {
    if (animate) cursorTracker.set(clamp(p, pMin, pMax), true);
    else cursorTracker.jump(clamp(p, pMin, pMax));
  }

  slider.addEventListener("input", () => cursorTracker.jump(clamp(Number(slider.value) / 1000, pMin, pMax)));

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 40;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      cursorTracker.jump(clamp((px - ml) / plotW, pMin, pMax));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Cross-entropy loss versus predicted probability of the true class: the curve negative-log-q is near zero when the prediction is confident and correct and diverges as the probability approaches zero. Drag to read the loss off the curve at a chosen probability.",
  );

  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setP(cursor - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setP(cursor + step, true);
    else if (e.key === "Home") setP(0.01, true);
    else if (e.key === "End") setP(0.9, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the confident-and-right reference immediately.
  if (prefersReducedMotion()) cursor = clamp(options.p ?? 0.9, pMin, pMax);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setP,
    destroy() {
      destroyed = true;
      cursorTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}
