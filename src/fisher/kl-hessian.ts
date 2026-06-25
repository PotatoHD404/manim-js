import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface FisherKlHessianOptions {
  /** Bernoulli base rate p₀ for the left panel (the post uses 0.35). */
  p0?: number;
  /** Gaussian base mean μ₀ for the right panel (the post uses 0). */
  mu0?: number;
  /** Gaussian base variance σ₀² for the right panel (the post uses 1). */
  v0?: number;
  /** Half-width of the δ sweep on the left panel (the post uses 0.22). */
  deltaMax?: number;
  /** Initial read-off shift δ on the left panel. */
  delta?: number;
  theme?: Partial<Theme>;
}

export interface FisherKlHessianApi {
  /** Move the left-panel read-off cursor to δ (eased glide when `animate`). */
  setDelta(delta: number, animate?: boolean): void;
  /** Set the Bernoulli base rate p₀ (left panel). */
  setP0(p0: number): void;
  /** Set the Gaussian base mean μ₀ (right panel). */
  setMu0(mu0: number): void;
  /** Set the Gaussian base variance σ₀² (right panel). */
  setV0(v0: number): void;
  /** Sweep δ out from the truth and back, watching KL peel off the quadratic. */
  play(): void;
  destroy(): void;
}

/**
 * Scene — "the Fisher matrix is the Hessian of the KL divergence". Expand the KL
 * divergence to a nearby distribution: the constant and linear terms vanish (KL
 * is minimized at δ = 0) and the quadratic term is exactly ½ δᵀ I δ. So near the
 * truth the exact KL and the Fisher quadratic coincide, and they peel apart only
 * at third order outward.
 *
 * Left — 1-D Bernoulli. Over δ ∈ [−Δ, Δ] the blue exact KL D(p₀ ∥ p₀+δ) is drawn
 * against the gold parabola ½ I(p₀) δ², with I(p₀) = 1/(p₀(1−p₀)). A draggable
 * cream cursor reads both curves off at the chosen δ and traces the red residual
 * KL − ½Iδ² — zero at the origin, growing as |δ| grows. A slider sets p₀.
 *
 * Right — 2-D Gaussian N(μ, σ²). The exact KL contours (blue, solid) sit against
 * the Fisher-ellipse contours ½ δᵀ I δ = c (gold, dashed) for I = diag(1/σ²,
 * 1/2σ⁴). Inner rings hug; outer rings separate. Sliders dial the base (μ₀, σ₀²).
 *
 * Ported verbatim from the post's notebook cell (p₀ = 0.35, μ₀ = 0, σ₀² = 1,
 * δ ∈ [−0.22, 0.22], contour levels [0.02, 0.08, 0.2, 0.4]).
 */
export function createFisherKlHessian(
  target: HTMLElement,
  options: FisherKlHessianOptions = {},
): FisherKlHessianApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  const dMax = clamp(options.deltaMax ?? 0.22, 0.05, 0.4);
  let p0 = clamp(options.p0 ?? 0.35, 0.05, 0.95);
  let mu0 = clamp(options.mu0 ?? 0.0, -2, 2);
  let v0 = clamp(options.v0 ?? 1.0, 0.4, 3);
  let destroyed = false;

  // contour levels and grid extent for the right panel (matches the notebook)
  const levels = [0.02, 0.08, 0.2, 0.4];
  const gHalf = 0.9;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;display:grid;grid-template-columns:1fr 1fr;gap:1px;width:100%;border-radius:14px;overflow:hidden;background:${theme.grid};`,
  });
  const leftHost = el("div", { style: `position:relative;background:${theme.bg};` });
  const rightHost = el("div", { style: `position:relative;background:${theme.bg};` });
  panel.append(leftHost, rightHost);
  root.append(panel);

  // left overlay: the live KL vs quadratic at the cursor, and their residual
  const leftOverlay = overlayLayer("tl");
  const eqKl = liveEquation("D_{\\mathrm{KL}}\\;=", theme, theme.blue);
  const eqQuad = liveEquation("\\tfrac12 I(p_0)\\,\\delta^2\\;=", theme, theme.gold);
  const eqGap = liveEquation("\\text{residual}\\;=", theme, theme.red);
  leftOverlay.append(eqKl.node, eqQuad.node, eqGap.node);
  leftHost.append(leftOverlay);

  const leftTag = el("div", {
    style: `position:absolute;top:12px;right:14px;pointer-events:none;color:${theme.muted};font:12px ${theme.mono};text-align:right;`,
  });
  leftTag.textContent = "Bernoulli";
  leftHost.append(leftTag);

  // right overlay: the Fisher matrix, restated for the current σ₀²
  const rightOverlay = overlayLayer("tr");
  const eqFisher = staticEquation("", theme, theme.gold);
  rightOverlay.append(eqFisher);
  rightHost.append(rightOverlay);

  const rightTag = el("div", {
    style: `position:absolute;top:12px;left:14px;pointer-events:none;color:${theme.muted};font:12px ${theme.mono};`,
  });
  rightTag.textContent = "Gaussian  N(μ, σ²)";
  rightHost.append(rightTag);

  const rcL = responsiveCanvas(leftHost, 1.15, () => render());
  const rcR = responsiveCanvas(rightHost, 1.15, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });

  const p0Row = sliderRow(theme, "p₀", 5, 95, Math.round(p0 * 100), "Bernoulli base rate p0, in percent");
  const muRow = sliderRow(theme, "μ₀", -20, 20, Math.round(mu0 * 10), "Gaussian base mean mu0, in tenths");
  const vRow = sliderRow(theme, "σ₀²", 4, 30, Math.round(v0 * 10), "Gaussian base variance sigma0 squared, in tenths");
  controls.append(playBtn, p0Row.row, muRow.row, vRow.row);
  root.append(controls);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag left ↔ to read δ  ·  ←→ scrub  ·  sweep δ  ·  sliders set the base point";
  root.append(hint);

  target.append(root);

  // --- the one eased scalar: the read-off shift δ on the left panel --------
  let delta = clamp(options.delta ?? 0.12, -dMax, dMax);
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  const deltaTracker = valueTracker(delta, (v) => {
    delta = clamp(v, -dMax, dMax);
    render();
  });

  // --- math (ported verbatim from the notebook) ----------------------------
  /** KL( Bern(p) ‖ Bern(q) ). */
  function klBern(p: number, q: number): number {
    const qc = clamp(q, 1e-9, 1 - 1e-9);
    return p * Math.log(p / qc) + (1 - p) * Math.log((1 - p) / (1 - qc));
  }
  /** ½ I(p₀) δ² with I(p₀) = 1/(p₀(1−p₀)). */
  function quadBern(d: number): number {
    return (0.5 * d * d) / (p0 * (1 - p0));
  }
  /** KL( N(m0,vv0) ‖ N(m1,vv1) ). */
  function klGauss(m0: number, vv0: number, m1: number, vv1: number): number {
    return 0.5 * (Math.log(vv1 / vv0) + (vv0 + (m0 - m1) ** 2) / vv1 - 1.0);
  }
  /** ½ δᵀ I δ for the Gaussian, I = diag(1/v₀, 1/2v₀²) over (Δμ, Δσ²). */
  function quadGauss(dm: number, dv: number): number {
    const i00 = 1 / v0;
    const i11 = 1 / (2 * v0 * v0);
    return 0.5 * (i00 * dm * dm + i11 * dv * dv);
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    drawLeft();
    drawRight();

    const klNow = klBern(p0, p0 + delta);
    const quadNow = quadBern(delta);
    eqKl.set(klNow.toFixed(4));
    eqQuad.set(quadNow.toFixed(4));
    eqGap.set((klNow - quadNow).toFixed(4));

    eqFisher.innerHTML = staticEquation(
      `I=\\operatorname{diag}\\!\\big(\\tfrac1{\\sigma_0^2},\\tfrac1{2\\sigma_0^4}\\big)=\\operatorname{diag}(${(1 / v0).toFixed(2)},\\,${(1 / (2 * v0 * v0)).toFixed(2)})`,
      theme,
      theme.gold,
    ).innerHTML;

    if (document.activeElement !== p0Row.input) p0Row.set(Math.round(p0 * 100));
    if (document.activeElement !== muRow.input) muRow.set(Math.round(mu0 * 10));
    if (document.activeElement !== vRow.input) vRow.set(Math.round(v0 * 10));
  }

  // --- left panel: 1-D Bernoulli KL vs Fisher parabola ---------------------
  function drawLeft(): void {
    const { ctx, width: W, height: H } = rcL;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 16;
    const mt = 20;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // sample both curves on a shared grid
    const N = 240;
    const ds = new Float64Array(N);
    const kls = new Float64Array(N);
    const qs = new Float64Array(N);
    let yMax = 1e-6;
    for (let i = 0; i < N; i++) {
      const d = -dMax + (2 * dMax) * (i / (N - 1));
      ds[i] = d;
      kls[i] = klBern(p0, p0 + d);
      qs[i] = quadBern(d);
      if (kls[i] > yMax) yMax = kls[i];
      if (qs[i] > yMax) yMax = qs[i];
    }
    yMax *= 1.12;

    const x = (d: number) => ml + ((d + dMax) / (2 * dMax)) * plotW;
    const y = (vv: number) => mt + plotH - (clamp(vv, 0, yMax) / yMax) * plotH;

    // faint number-plane backdrop (the cyan grid receding)
    for (const gd of [-0.2, -0.1, 0, 0.1, 0.2]) {
      if (Math.abs(gd) > dMax + 1e-9) continue;
      strokeLine(ctx, [x(gd), mt], [x(gd), mt + plotH], theme.grid, 1, gd === 0 ? 0.34 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // residual band KL − quadratic (red, faint fill) — the third-order gap
    const top: Pt[] = [];
    const bot: Pt[] = [];
    for (let i = 0; i < N; i++) {
      top.push([x(ds[i]), y(kls[i])]);
      bot.push([x(ds[i]), y(qs[i])]);
    }
    const band: Pt[] = [...top, ...[...bot].reverse()];
    polyline(ctx, band, { color: "transparent", width: 0, alpha: 1, closed: true, fill: hexA(theme.red, 0.12) });

    // gold Fisher parabola (the local approximation)
    polyline(ctx, bot, { color: theme.gold, width: 1.8, alpha: 0.9, dash: [7, 5] });
    // blue exact KL (the answer object), faintly glowing
    glow(ctx, theme.blue, 6, () => polyline(ctx, top, { color: theme.blue, width: 2.4, alpha: 1 }));

    // the read-off cursor (cream): vertical rule + a dot on each curve + the gap
    const klC = klBern(p0, p0 + delta);
    const qC = quadBern(delta);
    const cx = x(delta);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.55);
    // the residual segment between the two curves at the cursor
    glow(ctx, theme.red, 5, () => strokeLine(ctx, [cx, y(klC)], [cx, y(qC)], theme.red, 2.4, 0.95));
    disc(ctx, [cx, y(qC)], 2.8, theme.gold, 0.95);
    glow(ctx, theme.cream, 7, () => disc(ctx, [cx, y(klC)], 3.6, theme.cream, 1));
    mathLabel(ctx, "δ", [cx + (delta < dMax * 0.8 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 15,
      align: delta < dMax * 0.8 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "D", [ml - 8, mt + 4], { color: theme.tick, size: 14, sub: "KL", align: "right" });
    mathLabel(ctx, "δ", [W - mr - 2, mt + plotH + 24], { color: theme.tick, size: 15, align: "right" });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const gd of [-0.2, -0.1, 0, 0.1, 0.2]) {
      if (Math.abs(gd) > dMax + 1e-9) continue;
      ctx.fillText(gd.toFixed(1), x(gd), mt + plotH + 5);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let k = 1; k <= 3; k++) {
      const vv = (yMax / 1.12) * (k / 3);
      ctx.fillText(vv.toFixed(2), ml - 6, y(vv));
    }

    drawLegend(ctx, ml + 8, mt + 2, [
      ["exact KL", theme.blue, []],
      ["½ I(p₀) δ²", theme.gold, [7, 5]],
    ]);
  }

  // --- right panel: 2-D Gaussian KL contours vs Fisher ellipses ------------
  function drawRight(): void {
    const { ctx, width: W, height: H } = rcR;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    // equal-aspect square mapping of (Δμ, Δσ²) ∈ [−gHalf, gHalf]² to the panel
    const pad = 30;
    const side = Math.min(W, H) - 2 * pad;
    const ox = (W - side) / 2;
    const oy = (H - side) / 2;
    const X = (dm: number) => ox + ((dm + gHalf) / (2 * gHalf)) * side;
    const Y = (dv: number) => oy + (1 - (dv + gHalf) / (2 * gHalf)) * side;

    // faint number plane
    for (const g of [-0.5, 0, 0.5]) {
      strokeLine(ctx, [X(g), Y(-gHalf)], [X(g), Y(gHalf)], theme.grid, 1, g === 0 ? 0.34 : 0.18);
      strokeLine(ctx, [X(-gHalf), Y(g)], [X(gHalf), Y(g)], theme.grid, 1, g === 0 ? 0.34 : 0.18);
    }
    strokeLine(ctx, [X(-gHalf), Y(0)], [X(gHalf), Y(0)], theme.axis, 1.3, 0.9);
    strokeLine(ctx, [X(0), Y(-gHalf)], [X(0), Y(gHalf)], theme.axis, 1.3, 0.9);

    // sample both scalar fields on a grid (marching squares per level)
    const M = 96;
    const klField = new Float64Array((M + 1) * (M + 1));
    const quadField = new Float64Array((M + 1) * (M + 1));
    const at = (i: number) => -gHalf + (2 * gHalf) * (i / M);
    for (let j = 0; j <= M; j++) {
      const dv = at(j);
      for (let i = 0; i <= M; i++) {
        const dm = at(i);
        const idx = j * (M + 1) + i;
        klField[idx] = klGauss(mu0, v0, mu0 + dm, v0 + dv);
        quadField[idx] = quadGauss(dm, dv);
      }
    }

    const toScreen = (i: number, j: number): Pt => [X(at(i)), Y(at(j))];

    // gold dashed Fisher ellipses (the local approximation): one dashed batched
    // path per level so the dashes read as in the source figure
    ctx.save();
    ctx.setLineDash([7, 5]);
    ctx.strokeStyle = theme.gold;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.85;
    ctx.lineCap = "round";
    for (const lv of levels) {
      const segs = marchingSquares(quadField, M, lv);
      ctx.beginPath();
      for (const s of segs) {
        const a = toScreen(s[0], s[1]);
        const b = toScreen(s[2], s[3]);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.restore();

    // blue solid exact-KL contours over them, faintly glowing (the answer): the
    // inner rings coincide with the gold ellipses, the outer rings separate
    for (const lv of levels) {
      const segs = marchingSquares(klField, M, lv);
      glow(ctx, theme.blue, 4, () => {
        for (const s of segs) {
          const a = toScreen(s[0], s[1]);
          const b = toScreen(s[2], s[3]);
          strokeLine(ctx, a, b, theme.blue, 2, 0.95);
        }
      });
    }

    // the truth at the origin (the minimum of both surfaces)
    glow(ctx, theme.cream, 6, () => disc(ctx, [X(0), Y(0)], 3.4, theme.cream, 1));

    // axis labels (KaTeX serif) and ticks
    mathLabel(ctx, "Δμ", [X(gHalf) - 2, Y(0) + 18], { color: theme.tick, size: 14, align: "right" });
    mathLabel(ctx, "Δσ", [X(0) + 8, Y(gHalf) + 12], { color: theme.tick, size: 14, sub: "2", align: "left" });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const g of [-0.5, 0.5]) ctx.fillText(g.toFixed(1), X(g), Y(0) + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const g of [-0.5, 0.5]) ctx.fillText(g.toFixed(1), X(0) - 6, Y(g));

    drawLegend(ctx, ox + 8, oy + 2, [
      ["exact KL", theme.blue, []],
      ["½ δᵀ I δ", theme.gold, [7, 5]],
    ]);
  }

  // --- shared legend -------------------------------------------------------
  function drawLegend(
    ctx: CanvasRenderingContext2D,
    x0: number,
    y0: number,
    rows: [string, string, number[]][],
  ): void {
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = y0 + 8;
    for (const [name, color, dash] of rows) {
      ctx.strokeStyle = color;
      ctx.lineWidth = dash.length ? 1.4 : 2.2;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(x0, yy);
      ctx.lineTo(x0 + lineW, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = theme.fg;
      ctx.fillText(name, x0 + lineW + 6, yy + 0.5);
      yy += 16;
    }
    ctx.restore();
  }

  // --- interaction ---------------------------------------------------------
  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function setDelta(d: number, animate = false): void {
    stopSweep();
    const t = clamp(d, -dMax, dMax);
    if (animate) deltaTracker.set(t, true);
    else deltaTracker.jump(t);
  }
  function setP0(v: number): void {
    p0 = clamp(v, 0.05, 0.95);
    render();
  }
  function setMu0(v: number): void {
    mu0 = clamp(v, -2, 2);
    render();
  }
  function setV0(v: number): void {
    v0 = clamp(v, 0.4, 3);
    render();
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setDelta(dMax, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    let t = 0;
    sweepCancel = ticker((dt) => {
      t += dt / 3.2;
      if (t >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        deltaTracker.set(dMax, true);
        return false;
      }
      // a full there-and-back sweep across the δ range
      deltaTracker.jump(Math.sin(t * Math.PI * 2) * dMax);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; sweep δ";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  p0Row.input.addEventListener("input", () => setP0(Number(p0Row.input.value) / 100));
  muRow.input.addEventListener("input", () => setMu0(Number(muRow.input.value) / 10));
  vRow.input.addEventListener("input", () => setV0(Number(vRow.input.value) / 10));

  // drag on the left canvas scrubs δ
  const stopDrag = draggable(
    rcL.canvas,
    (px) => {
      stopSweep();
      const ml = 46;
      const mr = 16;
      const plotW = rcL.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      deltaTracker.jump(-dMax + 2 * dMax * f);
    },
    { onStart: () => (rcL.canvas.style.cursor = "grabbing"), onEnd: () => (rcL.canvas.style.cursor = "grab") },
  );
  rcL.canvas.style.cursor = "grab";

  for (const c of [rcL.canvas, rcR.canvas]) {
    c.tabIndex = 0;
    c.setAttribute("role", "img");
  }
  rcL.canvas.setAttribute(
    "aria-label",
    "Bernoulli panel: the exact KL divergence from the base rate plotted against its Fisher parabola one-half I times delta squared. Drag to read both off at a chosen shift delta; the red residual between them is zero at the truth and grows outward.",
  );
  rcR.canvas.setAttribute(
    "aria-label",
    "Gaussian panel: exact KL divergence contours plotted against the Fisher-ellipse contours one-half delta-transpose I delta. Inner rings near the truth coincide; outer rings separate.",
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const step = e.shiftKey ? dMax / 5 : dMax / 22;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setDelta(delta - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setDelta(delta + step, true);
    else if (e.key === "Home") setDelta(0, true);
    else return;
    e.preventDefault();
  };
  rcL.canvas.addEventListener("keydown", onKey);
  rcR.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on a static informative δ with no glide.
  if (prefersReducedMotion()) delta = clamp(options.delta ?? 0.12, -dMax, dMax);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setDelta,
    setP0,
    setMu0,
    setV0,
    play,
    destroy() {
      destroyed = true;
      deltaTracker.stop();
      sweepCancel?.();
      stopDrag();
      rcL.canvas.removeEventListener("keydown", onKey);
      rcR.canvas.removeEventListener("keydown", onKey);
      rcL.destroy();
      rcR.destroy();
      root.remove();
    },
  };
}

// --- small UI builder ------------------------------------------------------
interface Row {
  row: HTMLElement;
  input: HTMLInputElement;
  set(v: number): void;
}

function sliderRow(theme: Theme, label: string, min: number, max: number, value: number, aria: string): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:8px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:2.6ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: "1",
    value: String(value),
    style: "flex:1;min-width:84px;max-width:150px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  row.append(lab, input);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
    },
  };
}

/**
 * Marching squares on a scalar field sampled on an (M+1)×(M+1) grid, returning
 * the line segments of the iso-contour at `level` in grid coordinates
 * `[i0, j0, i1, j1]` (fractional cell indices). Linear interpolation along each
 * crossed edge keeps the contours smooth. The deterministic stand-in for
 * matplotlib's `contour`.
 */
function marchingSquares(field: Float64Array, M: number, level: number): Array<[number, number, number, number]> {
  const segs: Array<[number, number, number, number]> = [];
  const w = M + 1;
  const v = (i: number, j: number) => field[j * w + i];
  // interpolate the crossing fraction along an edge between two corner values
  const frac = (a: number, b: number) => {
    const dd = b - a;
    return Math.abs(dd) < 1e-12 ? 0.5 : (level - a) / dd;
  };
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const tl = v(i, j);
      const tr = v(i + 1, j);
      const br = v(i + 1, j + 1);
      const bl = v(i, j + 1);
      let code = 0;
      if (tl > level) code |= 8;
      if (tr > level) code |= 4;
      if (br > level) code |= 2;
      if (bl > level) code |= 1;
      if (code === 0 || code === 15) continue;
      // edge crossing points in grid coordinates
      const eTop: Pt = [i + frac(tl, tr), j];
      const eRight: Pt = [i + 1, j + frac(tr, br)];
      const eBottom: Pt = [i + frac(bl, br), j + 1];
      const eLeft: Pt = [i, j + frac(tl, bl)];
      const push = (a: Pt, b: Pt) => segs.push([a[0], a[1], b[0], b[1]]);
      switch (code) {
        case 1:
        case 14:
          push(eLeft, eBottom);
          break;
        case 2:
        case 13:
          push(eBottom, eRight);
          break;
        case 3:
        case 12:
          push(eLeft, eRight);
          break;
        case 4:
        case 11:
          push(eTop, eRight);
          break;
        case 5:
          push(eLeft, eTop);
          push(eBottom, eRight);
          break;
        case 6:
        case 9:
          push(eTop, eBottom);
          break;
        case 7:
        case 8:
          push(eLeft, eTop);
          break;
        case 10:
          push(eTop, eRight);
          push(eLeft, eBottom);
          break;
        default:
          break;
      }
    }
  }
  return segs;
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
