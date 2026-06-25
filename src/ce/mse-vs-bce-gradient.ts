import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface MseVsBceGradientOptions {
  /** Pre-activation range, symmetric. The notebook uses z ∈ [−8, 8]. */
  zRange?: number;
  /** Binary target for the example; the post uses a positive example y = 1. */
  target?: 0 | 1;
  /** Initial position of the read-off point along z. Default −6 (confidently wrong). */
  z?: number;
  theme?: Partial<Theme>;
}

export interface MseVsBceGradientApi {
  /** Move the read-off point to a pre-activation z (eased unless animate is false). */
  setZ(z: number, animate?: boolean): void;
  /** Glide to the confidently-wrong regime where MSE's gradient has died. */
  toConfidentlyWrong(): void;
  destroy(): void;
}

/**
 * Scene — "why not MSE?". For a positive example (y = 1) with ŷ = σ(z), the
 * gradient magnitudes with respect to the pre-activation z are
 *
 *     cross-entropy:  |∂L/∂z| = |σ(z) − y|,
 *     squared error:  |∂L/∂z| = |(σ(z) − y) · σ(z)(1 − σ(z))|.
 *
 * The squared-error gradient carries the extra factor σ′(z) = σ(z)(1−σ(z)),
 * which collapses to zero when the sigmoid saturates — so a confidently-wrong
 * neuron (σ(z) ≈ 0) gets almost no gradient under MSE and stops learning, while
 * cross-entropy's gradient stays near one. Drag the cream point along z to read
 * both magnitudes off; the live ratio shows how much gradient MSE throws away.
 *
 * Ported verbatim from the post's notebook cell (z ∈ [−8, 8], 800 points,
 * y = 1): BCE = |s − y|, MSE = |(s − y)·s·(1 − s)|.
 */
export function createMseVsBceGradient(
  target: HTMLElement,
  options: MseVsBceGradientOptions = {},
): MseVsBceGradientApi {
  const theme = withTheme(options.theme);
  const zr = options.zRange ?? 8;
  const y = options.target ?? 1;

  // z grid (matches the notebook: 800 points across the symmetric range).
  const G = 800;
  const zg = new Float64Array(G);
  for (let i = 0; i < G; i++) zg[i] = -zr + (2 * zr * i) / (G - 1);

  // Precomputed gradient-magnitude curves — pure functions of z, so they are
  // fixed; only the read-off point moves.
  const bceMag = new Float64Array(G);
  const mseMag = new Float64Array(G);
  for (let i = 0; i < G; i++) {
    const s = sigmoid(zg[i]);
    bceMag[i] = Math.abs(s - y);
    mseMag[i] = Math.abs((s - y) * s * (1 - s));
  }
  const yMax = 1.05; // the notebook's axG.set_ylim(0, 1.05)
  // The shaded "confidently WRONG" band from the notebook: z ∈ [−8, −3] for y=1.
  const wrongLo = y === 1 ? -zr : 3;
  const wrongHi = y === 1 ? -3 : zr;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the two gradient magnitudes read off at the current z
  const overlay = overlayLayer("tl");
  const eqBce = liveEquation("\\left|\\tfrac{\\partial L_{\\text{BCE}}}{\\partial z}\\right|=", theme, theme.gold);
  const eqMse = liveEquation("\\left|\\tfrac{\\partial L_{\\text{MSE}}}{\\partial z}\\right|=", theme, theme.red);
  const eqRatio = liveEquation("\\text{ratio}\\;\\tfrac{\\text{MSE}}{\\text{BCE}}=", theme, theme.green);
  overlay.append(eqBce.node, eqMse.node, eqRatio.node);
  panel.append(overlay);

  // top-right: the closed forms, restated
  const formOverlay = overlayLayer("tr");
  formOverlay.append(
    staticEquation("\\left|\\partial_z L_{\\text{BCE}}\\right|=|\\sigma(z)-y|", theme, theme.gold),
    staticEquation("\\left|\\partial_z L_{\\text{MSE}}\\right|=|(\\sigma(z)-y)\\,\\sigma'(z)|", theme, theme.red),
    staticEquation(`y=${y}`, theme, theme.muted),
  );
  panel.append(formOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const wrongBtn = el("button", { type: "button", style: btnStyle(theme) });
  wrongBtn.innerHTML = "&#9194;&nbsp; confidently wrong";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "pre-activation  z";
  const slider = el("input", {
    type: "range",
    min: String(-zr),
    max: String(zr),
    step: "0.05",
    value: String(clamp(options.z ?? -6, -zr, zr)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "pre-activation z at which to read off the gradient magnitudes",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · ←→ · jump to the dead zone";
  controls.append(wrongBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the read-off point over z ---------------------
  let z = clamp(options.z ?? -6, -zr, zr);
  let destroyed = false;
  const zTracker = valueTracker(z, (v) => {
    z = clamp(v, -zr, zr);
    render();
  });

  // linear interpolation of a precomputed curve at a continuous z
  function curveAt(arr: Float64Array, zv: number): number {
    const f = ((clamp(zv, -zr, zr) + zr) / (2 * zr)) * (G - 1);
    const i = clamp(Math.floor(f), 0, G - 2);
    const frac = f - i;
    return arr[i] * (1 - frac) + arr[i + 1] * frac;
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    draw();
    const bce = curveAt(bceMag, z);
    const mse = curveAt(mseMag, z);
    eqBce.set(bce.toFixed(3));
    eqMse.set(mse.toFixed(3));
    eqRatio.set(bce > 1e-6 ? (mse / bce).toFixed(3) : "—");
    if (document.activeElement !== slider) slider.value = z.toFixed(2);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 44;
    const mr = 16;
    const mt = 16;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const x = (zv: number) => ml + ((clamp(zv, -zr, zr) + zr) / (2 * zr)) * plotW;
    const yv = (v: number) => mt + plotH - (clamp(v, 0, yMax) / yMax) * plotH;

    // the "confidently WRONG" band — the saturated regime where MSE dies
    ctx.save();
    ctx.fillStyle = theme.muted;
    ctx.globalAlpha = 0.1;
    ctx.fillRect(x(wrongLo), mt, x(wrongHi) - x(wrongLo), plotH);
    ctx.restore();

    // faint vertical grid in BLUE_D (the receded number plane) at every 2 units
    for (let g = -zr; g <= zr + 1e-6; g += 2) {
      strokeLine(ctx, [x(g), mt], [x(g), mt + plotH], theme.grid, 1, g === 0 ? 0.32 : 0.16);
    }
    // a faint horizontal rule at the BCE plateau height (|σ−y| → 1 when wrong)
    strokeLine(ctx, [ml, yv(1)], [W - mr, yv(1)], theme.grid, 1, 0.18);

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    const toScreen = (arr: Float64Array): Pt[] => {
      const out: Pt[] = new Array(G);
      for (let i = 0; i < G; i++) out[i] = [x(zg[i]), yv(arr[i])];
      return out;
    };

    // squared-error gradient (red, dashed) — the failing object: vanishes when
    // the unit is confidently wrong (left) or confidently right (far right).
    polyline(ctx, toScreen(mseMag), { color: theme.red, width: 2.2, alpha: 0.95, dash: [7, 5] });

    // cross-entropy gradient (gold, solid, faintly glowing) — the answer: stays
    // near one exactly where MSE has died.
    glow(ctx, theme.gold, 6, () => {
      polyline(ctx, toScreen(bceMag), { color: theme.gold, width: 2.6, alpha: 1 });
    });

    // the read-off point (the movable variable): a vertical rule, ticks on both
    // curves, and dashed drops to the axis showing each magnitude.
    const bce = curveAt(bceMag, z);
    const mse = curveAt(mseMag, z);
    const cx = x(z);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.55);

    // drop lines from each read-off to the y-axis
    strokeLine(ctx, [ml, yv(bce)], [cx, yv(bce)], theme.gold, 1, 0.4);
    strokeLine(ctx, [ml, yv(mse)], [cx, yv(mse)], theme.red, 1, 0.4);

    glow(ctx, theme.gold, 6, () => disc(ctx, [cx, yv(bce)], 4, theme.gold, 1));
    glow(ctx, theme.red, 6, () => disc(ctx, [cx, yv(mse)], 4, theme.red, 1));

    // the z marker label on the axis, in KaTeX serif
    mathLabel(ctx, "z", [cx + (z < zr - 1 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 16,
      align: z < zr - 1 ? "left" : "right",
      glow: 5,
    });

    // gap bracket between the two read-offs — the gradient MSE throws away
    if (Math.abs(bce - mse) > 0.04) {
      const gx = cx + (z < 0 ? 14 : -14);
      strokeLine(ctx, [gx, yv(bce)], [gx, yv(mse)], theme.green, 1.4, 0.85);
      strokeLine(ctx, [gx - 3, yv(bce)], [gx + 3, yv(bce)], theme.green, 1.4, 0.85);
      strokeLine(ctx, [gx - 3, yv(mse)], [gx + 3, yv(mse)], theme.green, 1.4, 0.85);
      mathLabel(ctx, "Δ", [gx + (z < 0 ? 6 : -6), (yv(bce) + yv(mse)) / 2 + 5], {
        color: theme.green,
        size: 14,
        align: z < 0 ? "left" : "right",
        glow: 4,
      });
    }

    // x ticks (pre-activation z) in roman numerals
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = -zr; g <= zr + 1e-6; g += 2) ctx.fillText(String(g), x(g), mt + plotH + 6);

    // y ticks (gradient magnitude)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(v.toFixed(2), ml - 6, yv(v));

    // axis captions
    mathLabel(ctx, "z", [W - mr - 4, mt + plotH + 24], { color: theme.tick, size: 15, align: "right" });
    mathLabel(ctx, "|∂_z L|", [ml - 30, mt + 4], { color: theme.tick, size: 14, align: "left", italic: false });

    // legend + the "confidently wrong" caption inside the band
    drawLegend(ctx, W - mr, mt + plotH);
    ctx.fillStyle = theme.muted;
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    ctx.textAlign = y === 1 ? "left" : "right";
    ctx.textBaseline = "top";
    ctx.fillText("confidently wrong", y === 1 ? ml + 8 : W - mr - 8, mt + 6);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, bottom: number): void {
    const rows: [string, string, number[]][] = [
      ["cross-entropy  |σ−y|", theme.gold, []],
      ["squared error  |(σ−y)σ′|", theme.red, [7, 5]],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = bottom - 30;
    for (const [name, color, dash] of rows) {
      const lx = right - 168;
      ctx.strokeStyle = color;
      ctx.lineWidth = dash.length ? 2.2 : 2.6;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(lx, yy);
      ctx.lineTo(lx + lineW, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = theme.fg;
      ctx.fillText(name, lx + lineW + 6, yy + 0.5);
      yy += 15;
    }
    ctx.restore();
  }

  // --- interaction ---------------------------------------------------------
  function setZ(zv: number, animate = false): void {
    if (animate) zTracker.set(clamp(zv, -zr, zr), true);
    else zTracker.jump(clamp(zv, -zr, zr));
  }

  function toConfidentlyWrong(): void {
    setZ(y === 1 ? -6 : 6, true);
  }

  slider.addEventListener("input", () => zTracker.jump(Number(slider.value)));
  wrongBtn.addEventListener("click", toConfidentlyWrong);

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 44;
      const mr = 16;
      const plotW = rc.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      zTracker.jump(-zr + f * 2 * zr);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Gradient magnitude versus pre-activation z for a positive example: cross-entropy's gradient |sigma(z) minus y| stays near one, while squared error's gradient carries the extra factor sigma-prime and vanishes when the neuron is confidently wrong. Drag the point along z to read both magnitudes off.",
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      toConfidentlyWrong();
      return;
    }
    const step = e.shiftKey ? 1 : 0.2;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setZ(z - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setZ(z + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the confidently-wrong point with no glide.
  if (prefersReducedMotion()) z = y === 1 ? -6 : 6;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setZ,
    toConfidentlyWrong,
    destroy() {
      destroyed = true;
      zTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- helpers ----------------------------------------------------------------

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
