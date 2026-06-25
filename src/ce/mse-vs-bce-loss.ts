import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface MseVsBceLossOptions {
  /** Fixed binary label the losses are scored against. The notebook uses y = 1. */
  label?: 0 | 1;
  /** x-axis: "prob" plots loss vs predicted probability p; "logit" vs pre-activation z (p = σ(z)). */
  space?: "prob" | "logit";
  /** Initial predicted probability of the positive class. */
  prob?: number;
  theme?: Partial<Theme>;
}

export interface MseVsBceLossApi {
  /** Move the read-off point to predicted probability p (eased). */
  setProb(p: number, animate?: boolean): void;
  /** Set the fixed binary label y ∈ {0, 1}. */
  setLabel(y: 0 | 1): void;
  /** Swap the horizontal axis between probability p and logit z. */
  setSpace(space: "prob" | "logit"): void;
  destroy(): void;
}

/**
 * Scene — "why not MSE?". For a fixed binary label y, two losses are drawn over
 * the model's predicted probability p of the positive class: cross-entropy
 * −log p (the answer, gold, unbounded) and squared error (p−1)² (cream-warm,
 * bounded above by 1). A movable point reads both losses off at the chosen p,
 * and the gap between them is shaded — small near a correct, confident
 * prediction, but exploding as p heads toward a confident mistake, where
 * cross-entropy diverges while squared error flattens against its ceiling of 1.
 *
 * Ported from the post's notebook cell (axL of the "Why not MSE?" figure):
 * cross-entropy = −log p, squared error = (p−1)² for y = 1, with the loss = 1
 * reference line; x ∈ [1e−3, 1−1e−3], y ∈ [0, 5]. The logit view re-plots the
 * same losses against z with p = σ(z), making the saturated tail explicit.
 */
export function createMseVsBceLoss(
  target: HTMLElement,
  options: MseVsBceLossOptions = {},
): MseVsBceLossApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let y: 0 | 1 = options.label === 0 ? 0 : 1;
  let space: "prob" | "logit" = options.space === "logit" ? "logit" : "prob";
  let prob = clamp(options.prob ?? (y === 1 ? 0.18 : 0.82), PMIN, PMAX);
  let destroyed = false;

  // The losses, as a function of predicted probability p of the positive class,
  // for a fixed label y (the notebook's y = 1 by default).
  const ceLoss = (p: number): number => (y === 1 ? -Math.log(p) : -Math.log(1 - p));
  const mseLoss = (p: number): number => (p - y) * (p - y);

  // The notebook's grid: 600 points on the open probability interval.
  const G = 600;
  const pGrid = new Float64Array(G);
  for (let i = 0; i < G; i++) pGrid[i] = PMIN + (PMAX - PMIN) * (i / (G - 1));
  // The logit grid mirrors the notebook's z ∈ [−8, 8] view.
  const zGrid = new Float64Array(G);
  for (let i = 0; i < G; i++) zGrid[i] = ZMIN + (ZMAX - ZMIN) * (i / (G - 1));

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the two losses read off live at the cursor
  const overlay = overlayLayer("tl");
  const eqCe = liveEquation("L_{\\text{CE}}=-\\log p\\;=", theme, theme.gold);
  const eqMse = liveEquation("L_{\\text{MSE}}=(p-y)^{2}\\;=", theme, theme.cream);
  overlay.append(eqCe.node, eqMse.node);
  panel.append(overlay);

  // top-right: the fixed label and the bound MSE never crosses
  const factOverlay = overlayLayer("tr");
  const eqLabel = staticEquation("", theme, theme.blue);
  const eqBound = staticEquation("(p-1)^{2}\\le 1", theme, theme.muted);
  factOverlay.append(eqLabel, eqBound);
  panel.append(factOverlay);

  const rc = responsiveCanvas(panel, 1.5, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const labelBtn = el("button", { type: "button", style: btnStyle(theme) });
  const spaceBtn = el("button", { type: "button", style: btnStyle(theme) });
  const sliderLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  sliderLabel.textContent = "predicted probability  p";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1000",
    step: "1",
    value: String(probToSlider(prob)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "predicted probability of the positive class",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · y · logit · ←→";
  controls.append(labelBtn, spaceBtn, sliderLabel, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the read-off point's predicted probability ----
  const probTracker = valueTracker(prob, (v) => {
    prob = clamp(v, PMIN, PMAX);
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    draw();

    eqCe.set(ceLoss(prob).toFixed(3));
    eqMse.set(mseLoss(prob).toFixed(3));
    eqLabel.innerHTML = staticEquation(`y=${y}`, theme, theme.blue).innerHTML;

    if (document.activeElement !== slider) slider.value = String(probToSlider(prob));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 44;
    const mr = 18;
    const mt = 16;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // shared loss axis: the notebook clamps the view at loss = 5
    const x = (u: number) =>
      ml + ((u - (space === "prob" ? 0 : ZMIN)) / (space === "prob" ? 1 : ZMAX - ZMIN)) * plotW;
    const yOf = (loss: number) => mt + plotH - (clamp(loss, 0, LMAX) / LMAX) * plotH;

    const grid = space === "prob" ? pGrid : zGrid;
    const probOf = (g: number): number => (space === "prob" ? g : clamp(sigmoid(g), PMIN, PMAX));

    // faint receding number plane: vertical rules at the readable fractions
    const verts = space === "prob" ? [0, 0.25, 0.5, 0.75, 1] : [-8, -4, 0, 4, 8];
    for (const g of verts) {
      const edge = g === verts[0] || g === verts[verts.length - 1];
      strokeLine(ctx, [x(g), mt], [x(g), mt + plotH], theme.grid, 1, edge ? 0.28 : 0.16);
    }
    for (let lv = 1; lv <= LMAX - 0.5; lv++) {
      strokeLine(ctx, [ml, yOf(lv)], [ml + plotW, yOf(lv)], theme.grid, 1, 0.1);
    }

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    // the loss = 1 reference line: the ceiling squared error never exceeds
    polyline(ctx, [[ml, yOf(1)], [ml + plotW, yOf(1)]], {
      color: theme.muted,
      width: 1.2,
      alpha: 0.75,
      dash: [2, 4],
    });
    ctx.fillStyle = theme.muted;
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("loss = 1", ml + 6, yOf(1) - 3);

    // sample both losses across the grid
    const ceCurve: Pt[] = [];
    const mseCurve: Pt[] = [];
    const gap: Pt[] = [];
    for (let i = 0; i < G; i++) {
      const g = grid[i];
      const p = probOf(g);
      const sx = x(g);
      ceCurve.push([sx, yOf(ceLoss(p))]);
      mseCurve.push([sx, yOf(mseLoss(p))]);
    }
    // shaded gap between the two penalties (how much harder CE punishes)
    for (let i = 0; i < G; i++) gap.push(mseCurve[i]);
    for (let i = G - 1; i >= 0; i--) gap.push(ceCurve[i]);
    polyline(ctx, gap, { color: "transparent", width: 0, alpha: 1, closed: true, fill: hexA(theme.gold, 0.07) });

    // squared error (cream-warm, dashed, bounded) — "not wrong, just flat"
    polyline(ctx, mseCurve, { color: theme.cream, width: 2.2, alpha: 0.9, dash: [7, 5] });
    // cross-entropy (gold, solid, faintly glowing) — the answer, unbounded
    glow(ctx, theme.gold, 5, () => polyline(ctx, ceCurve, { color: theme.gold, width: 2.4, alpha: 1 }));

    // the movable read-off point at the chosen p
    const gNow = space === "prob" ? prob : logit(prob);
    const cx = clamp(x(gNow), ml, ml + plotW);
    const ceY = yOf(ceLoss(prob));
    const mseY = yOf(mseLoss(prob));
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.5);
    // connect the two read-offs so the live gap is legible
    strokeLine(ctx, [cx, ceY], [cx, mseY], theme.red, 1.6, 0.8);
    disc(ctx, [cx, mseY], 3.4, theme.cream, 1);
    glow(ctx, theme.gold, 7, () => disc(ctx, [cx, ceY], 4, theme.gold, 1));

    // movable-variable tick label
    const labLeft = gNow < (space === "prob" ? 0.86 : 4);
    mathLabel(ctx, space === "prob" ? "p" : "z", [cx + (labLeft ? 6 : -6), mt + plotH - 6], {
      color: theme.cream,
      size: 16,
      align: labLeft ? "left" : "right",
      glow: 5,
    });

    // x ticks (roman numerals)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const g of verts) {
      ctx.fillText(space === "prob" ? g.toFixed(2) : String(g), x(g), mt + plotH + 5);
    }
    // y ticks
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let lv = 0; lv <= LMAX; lv++) ctx.fillText(String(lv), ml - 6, yOf(lv));

    // axis captions in KaTeX serif
    mathLabel(ctx, space === "prob" ? "p" : "z", [ml + plotW, mt + plotH + 20], {
      color: theme.tick,
      size: 15,
      align: "right",
    });
    mathLabel(ctx, "L", [ml - 28, mt + 6], { color: theme.tick, size: 15 });

    // legend
    drawLegend(ctx, W - mr, mt + plotH);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, bottom: number): void {
    const rows: [string, string, number[]][] = [
      ["cross-entropy", theme.gold, []],
      ["squared error", theme.cream, [7, 5]],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = bottom - 32;
    for (const [name, color, dash] of rows) {
      const lx = right - 116;
      ctx.strokeStyle = color;
      ctx.lineWidth = dash.length ? 2.2 : 2.4;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(lx, yy);
      ctx.lineTo(lx + lineW, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = theme.fg;
      ctx.fillText(name, lx + lineW + 6, yy + 0.5);
      yy += 16;
    }
    ctx.restore();
  }

  // --- interaction ---------------------------------------------------------
  function setProb(p: number, animate = false): void {
    if (animate) probTracker.set(clamp(p, PMIN, PMAX), true);
    else probTracker.jump(clamp(p, PMIN, PMAX));
  }

  function setLabel(ny: 0 | 1): void {
    y = ny === 0 ? 0 : 1;
    updateLabelBtn();
    render();
  }

  function setSpace(ns: "prob" | "logit"): void {
    space = ns === "logit" ? "logit" : "prob";
    sliderLabel.textContent = space === "prob" ? "predicted probability  p" : "predicted probability  p  (x: logit z)";
    updateSpaceBtn();
    render();
  }

  function updateLabelBtn(): void {
    labelBtn.innerHTML = `y = ${y}`;
  }
  function updateSpaceBtn(): void {
    spaceBtn.textContent = space === "prob" ? "x: p" : "x: z";
  }
  updateLabelBtn();
  updateSpaceBtn();

  labelBtn.addEventListener("click", () => setLabel(y === 1 ? 0 : 1));
  spaceBtn.addEventListener("click", () => setSpace(space === "prob" ? "logit" : "prob"));
  slider.addEventListener("input", () => probTracker.jump(sliderToProb(Number(slider.value))));

  // Canvas drag sets p directly (real-time), mapping the pointer x through the
  // active horizontal axis (probability or logit).
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 44;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      const g = space === "prob" ? f : ZMIN + f * (ZMAX - ZMIN);
      probTracker.jump(clamp(space === "prob" ? g : sigmoid(g), PMIN, PMAX));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Loss versus predicted probability for a fixed binary label: cross-entropy minus-log-p diverges as the probability of the correct class goes to zero, while squared error stays bounded above by one. Drag the point to read both losses at a chosen probability.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "y" || e.key === "Y") {
      e.preventDefault();
      setLabel(y === 1 ? 0 : 1);
      return;
    }
    if (e.key === "z" || e.key === "Z") {
      e.preventDefault();
      setSpace(space === "prob" ? "logit" : "prob");
      return;
    }
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setProb(prob - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setProb(prob + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: no glide, the point lands where it starts.
  if (prefersReducedMotion()) prob = clamp(prob, PMIN, PMAX);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setProb,
    setLabel,
    setSpace,
    destroy() {
      destroyed = true;
      probTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- constants & helpers ----------------------------------------------------

// The notebook's open probability interval p ∈ [1e−3, 1−1e−3] and its viewport
// loss ∈ [0, 5]; the logit view re-uses its z ∈ [−8, 8] window.
const PMIN = 1e-3;
const PMAX = 1 - 1e-3;
const LMAX = 5;
const ZMIN = -8;
const ZMAX = 8;

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function logit(p: number): number {
  const c = clamp(p, PMIN, PMAX);
  return Math.log(c / (1 - c));
}

/** Map p ∈ [PMIN, PMAX] to the integer slider range [0, 1000]. */
function probToSlider(p: number): number {
  return Math.round(((clamp(p, PMIN, PMAX) - PMIN) / (PMAX - PMIN)) * 1000);
}

function sliderToProb(v: number): number {
  return PMIN + clamp(v / 1000, 0, 1) * (PMAX - PMIN);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}

/** Append an alpha byte to a #rrggbb color. */
function hexA(hex: string, alpha: number): string {
  const a = Math.round(clamp(alpha, 0, 1) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}
