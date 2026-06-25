import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { type Theme, withTheme } from "../core/theme";

export interface MatrixAr1Options {
  /** Matrix size k (number of coordinates). Default 8, clamped to [2, 12]. */
  size?: number;
  /** Initial correlation ρ ∈ [0, 0.98]. Default 0.6. */
  rho?: number;
  /**
   * Quantization resolution Δ for the completion-capacity functional. The post's
   * demo uses 0.6 — coarse enough that the weak directions freeze within the
   * draggable ρ range so the capacity climb is visible.
   */
  delta?: number;
  theme?: Partial<Theme>;
}

export interface MatrixAr1Api {
  /** Set the correlation ρ (eased when `animate`, direct otherwise). */
  setRho(rho: number, animate?: boolean): void;
  /** Set the matrix size k (rebuilds the spectrum). */
  setSize(k: number): void;
  destroy(): void;
}

const RHO_MAX = 0.98;
const K_MIN = 2;
const K_MAX = 12;

/**
 * Scene — "banded correlation, slowly accrued redundancy". The AR(1)/Toeplitz
 * covariance Σ_ij = ρ^|i-j| only couples nearby coordinates, so its k×k heatmap
 * is a band that thickens as ρ grows. Its eigenvalue spectrum (Jacobi) spreads
 * from flat-at-1 (ρ=0, independent, full rank) toward one dominant direction
 * plus a frozen tail as ρ→1. The quantized-Gaussian completion capacity
 * C = S / H_joint, computed at a fixed resolution Δ from those eigenvalues,
 * rises with ρ — but more slowly than the all-to-all equicorrelated case,
 * because only neighbours are strongly dependent. A cream handle drags ρ along
 * the C(ρ) curve; the heatmap, spectrum, and the closed-form log-determinant
 * (k−1)·log₂(1−ρ²) update together.
 *
 * Ported verbatim from the post's demo library: buildAR1(k, ρ), jacobiEigvals,
 * and quantizedCapacity(eigs, diagVars=1, Δ=0.6).
 */
export function createMatrixAr1(target: HTMLElement, options: MatrixAr1Options = {}): MatrixAr1Api {
  const theme = withTheme(options.theme);
  const delta = options.delta ?? 0.6;

  // --- model state ---------------------------------------------------------
  let k = clamp(Math.round(options.size ?? 8), K_MIN, K_MAX);
  let rho = clamp(options.rho ?? 0.6, 0, RHO_MAX); // the eased, displayed value
  let destroyed = false;

  // --- the model (ported from demos-core.js) -------------------------------
  // Σ_ij = ρ^|i-j| (AR(1)/Toeplitz). Unit variances on the diagonal.
  function buildAr1(kk: number, rr: number): Mat {
    const M: Mat = [];
    for (let i = 0; i < kk; i++) {
      const row: number[] = [];
      for (let j = 0; j < kk; j++) row.push(Math.pow(rr, Math.abs(i - j)));
      M.push(row);
    }
    return M;
  }

  // Quantized-Gaussian completion capacity C = S / H_joint at resolution Δ.
  // Each direction with variance λ contributes max(0, ½log₂(2πe λ) − log₂Δ) bits;
  // diagonal (marginal) variances are all 1, eigenvalues carry the joint entropy.
  function bitsOf(lam: number): number {
    return Math.max(0, 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(lam, 1e-300)) - Math.log2(delta));
  }
  function capacityFromEigs(eigs: number[]): number {
    // diagVars are all 1 (unit marginal variances), so S = k · bits(1).
    const S = eigs.length * bitsOf(1);
    let Hj = 0;
    for (const l of eigs) Hj += bitsOf(l);
    return Hj < 1e-12 ? Infinity : S / Hj;
  }
  function eigsAt(kk: number, rr: number): number[] {
    return jacobiEigen(buildAr1(kk, clamp(rr, 0, RHO_MAX))).values;
  }
  function capAt(kk: number, rr: number): number {
    return capacityFromEigs(eigsAt(kk, rr));
  }

  // Cached per-k: the C(ρ) sweep and the y-scale. Rebuilt on size change.
  const SWEEP = 161;
  let curve: number[] = [];
  let curEig: number[] = [];
  let curMat: Mat = [];
  let yMax = 1;

  function rebuild(): void {
    curve = Array.from({ length: SWEEP }, (_, i) => {
      const rr = (i / (SWEEP - 1)) * RHO_MAX;
      const c = capAt(k, rr);
      return Number.isFinite(c) ? c : k;
    });
    yMax = Math.max(k + 0.5, ...curve.map((c) => (Number.isFinite(c) ? c : 0))) * 1.04;
    recomputeCurrent();
  }
  function recomputeCurrent(): void {
    curMat = buildAr1(k, rho);
    curEig = jacobiEigen(curMat).values;
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const topHost = el("div", { style: "position:relative;" });
  const traceHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(topHost, traceHost);
  root.append(panel);

  // top-left: live capacity + sampling fraction
  const overlay = overlayLayer("tl");
  const eqCap = liveEquation("C=\\dfrac{S}{H_{\\text{joint}}}\\;=", theme, theme.green);
  const eqFrac = liveEquation("\\tfrac{1}{C}\\;=", theme, theme.cream);
  overlay.append(eqCap.node, eqFrac.node);
  topHost.append(overlay);

  // top-right: the AR(1) covariance + its closed-form log-determinant
  const idOverlay = overlayLayer("tr");
  const eqSigma = staticEquation("\\Sigma_{ij}=\\rho^{\\,|i-j|}", theme, theme.gold);
  const eqLogdet = liveEquation("\\log_2\\det\\Sigma=", theme, theme.muted);
  idOverlay.append(eqSigma, eqLogdet.node);
  topHost.append(idOverlay);

  const traceTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.green};font:12px ${theme.mono};`,
  });
  traceTag.textContent = "capacity  C  vs.  ρ";
  traceHost.append(traceTag);

  // --- controls ------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(2,minmax(170px,1fr));gap:8px 18px;align-items:center;",
  });
  const rhoRow = sliderRow(theme, "correlation  ρ", 0, RHO_MAX, rho, 0.01, "correlation rho", (v) => v.toFixed(2));
  const kRow = sliderRow(theme, "size  k", K_MIN, K_MAX, k, 1, "matrix size k", (v) => String(Math.round(v)));
  controls.append(rhoRow.row, kRow.row);
  root.append(controls);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ρ along the curve  ·  ←→ scrub  ·  sliders set ρ & k";
  root.append(hint);

  target.append(root);

  // Precompute the C(ρ) sweep and the current matrix/eigenvalues before any
  // canvas exists, so the first synchronous paint has real data to draw.
  rebuild();

  // --- canvases ------------------------------------------------------------
  const rcTop = responsiveCanvas(topHost, 2.05, () => render());
  const rcTrace = responsiveCanvas(traceHost, 4.6, () => render());

  // --- the one eased scalar: ρ --------------------------------------------
  const rhoTracker = valueTracker(rho, (v) => {
    rho = clamp(v, 0, RHO_MAX);
    recomputeCurrent();
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    drawTop();
    drawTrace();

    const c = capAt(k, rho);
    eqCap.set(Number.isFinite(c) ? c.toFixed(2) : "∞");
    eqFrac.set(Number.isFinite(c) && c > 0 ? (1 / c).toFixed(2) : "0");
    // closed form: log2 det Σ_AR(1) = (k-1) log2(1 - ρ²)
    const logdet = (k - 1) * Math.log2(Math.max(1 - rho * rho, 1e-300));
    eqLogdet.set(logdet.toFixed(2));

    if (document.activeElement !== rhoRow.input) rhoRow.set(rho);
    if (document.activeElement !== kRow.input) kRow.set(k);
  }

  // --- top panel: heatmap (left) + eigenvalue spectrum (right) -------------
  function drawTop(): void {
    const { ctx, width: W, height: H } = rcTop;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const mt = 16;
    const mb = 18;
    const innerH = H - mt - mb;
    // square heatmap on the left, the spectrum chart fills the rest.
    const hmSize = Math.min(innerH, (W - 60) * 0.42);
    const hmX = 18;
    const hmY = mt + (innerH - hmSize) / 2;
    drawHeatmap(ctx, hmX, hmY, hmSize);

    const specL = hmX + hmSize + 64;
    drawSpectrum(ctx, specL, mt, W - specL - 18, innerH);
  }

  /** AR(1) covariance heatmap: a BLUE_C ramp on black; the band thickens with ρ. */
  function drawHeatmap(ctx: CanvasRenderingContext2D, x0: number, y0: number, size: number): void {
    const cell = size / k;
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const v = clamp(curMat[i][j], 0, 1);
        // black → BLUE_C ramp; the diagonal is 1 (brightest), off-diagonals fade.
        ctx.fillStyle = mixHex(theme.bg, theme.blue, Math.pow(v, 0.8));
        ctx.globalAlpha = 1;
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.6, cell + 0.6);
      }
    }
    // grid + frame (faint cyan, like the receded number plane)
    ctx.globalAlpha = 1;
    for (let i = 0; i <= k; i++) {
      strokeLine(ctx, [x0, y0 + i * cell], [x0 + size, y0 + i * cell], theme.grid, 1, 0.18);
      strokeLine(ctx, [x0 + i * cell, y0], [x0 + i * cell, y0 + size], theme.grid, 1, 0.18);
    }
    strokeLine(ctx, [x0, y0], [x0 + size, y0], theme.axis, 1.2, 0.7);
    strokeLine(ctx, [x0, y0], [x0, y0 + size], theme.axis, 1.2, 0.7);
    strokeLine(ctx, [x0 + size, y0], [x0 + size, y0 + size], theme.axis, 1.2, 0.7);
    strokeLine(ctx, [x0, y0 + size], [x0 + size, y0 + size], theme.axis, 1.2, 0.7);
    mathLabel(ctx, "Σ", [x0 + size / 2, y0 + size + 14], {
      color: theme.gold,
      size: 15,
      align: "center",
      baseline: "alphabetic",
    });
  }

  /**
   * Eigenvalue spectrum bars (Jacobi), descending. The largest is GOLD — the
   * dominant direction that grows as ρ rises; the rest are BLUE_C; the freezing
   * threshold √λ = Δ (directions below it carry no bits) is a dashed RED line.
   */
  function drawSpectrum(ctx: CanvasRenderingContext2D, x0: number, y0: number, w: number, h: number): void {
    const mb = 18;
    const plotH = h - mb;
    const baseY = y0 + plotH;
    const eMax = Math.max(curEig[0], 1) * 1.08;
    const bx = (i: number) => x0 + ((i + 0.5) / k) * w;
    const by = (v: number) => baseY - (clamp(v, 0, eMax) / eMax) * plotH;
    const bw = (w / k) * 0.6;

    // baseline + left axis
    strokeLine(ctx, [x0, baseY], [x0 + w, baseY], theme.axis, 1.4, 1);
    strokeLine(ctx, [x0, y0], [x0, baseY], theme.axis, 1.4, 1);

    // freezing threshold: a direction with variance λ contributes bits only when
    // √λ > Δ, i.e. λ > Δ². Below this the direction is frozen (carries nothing).
    const lamFreeze = delta * delta;
    if (lamFreeze < eMax) {
      const yf = by(lamFreeze);
      polyline(ctx, [[x0, yf], [x0 + w, yf]], { color: theme.red, width: 1.2, alpha: 0.75, dash: [6, 6] });
      ctx.fillStyle = theme.red;
      ctx.font = `11px ${ROMAN_FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("frozen below √λ = Δ", x0 + 6, yf - 3);
    }

    // λ = 1 reference (the ρ = 0 / independent spectrum sits flat here)
    if (1 < eMax) {
      const y1 = by(1);
      polyline(ctx, [[x0, y1], [x0 + w, y1]], { color: theme.muted, width: 1, alpha: 0.45, dash: [3, 5] });
    }

    // bars
    for (let i = 0; i < k; i++) {
      const cx = bx(i);
      const top = by(curEig[i]);
      const dominant = i === 0;
      const frozen = curEig[i] <= lamFreeze;
      const col = dominant ? theme.gold : frozen ? theme.muted : theme.blue;
      ctx.fillStyle = col;
      ctx.globalAlpha = dominant ? 0.95 : frozen ? 0.4 : 0.82;
      ctx.fillRect(cx - bw / 2, top, bw, baseY - top);
      ctx.globalAlpha = 1;
    }
    // glow the dominant eigenvalue's cap
    glow(ctx, theme.gold, 6, () => disc(ctx, [bx(0), by(curEig[0])], 3.2, theme.gold, 1));

    // axis labels
    mathLabel(ctx, "λ", [x0 - 6, y0 + 4], { color: theme.tick, size: 16, align: "right" });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < k; i++) ctx.fillText(String(i + 1), bx(i), baseY + 4);
  }

  // --- bottom panel: the C(ρ) trace ---------------------------------------
  function drawTrace(): void {
    const { ctx, width: W, height: H } = rcTrace;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 30;
    const mr = 16;
    const mt = 26;
    const mb = 22;
    const x = (rr: number) => ml + (rr / RHO_MAX) * (W - ml - mr);
    const y = (cc: number) => H - mb - (clamp(cc, 0, yMax) / yMax) * (H - mt - mb);

    // axes
    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.2, 1);
    strokeLine(ctx, [ml, H - mb], [ml, mt], theme.axis, 1.2, 1);

    // C = 1 reference (independent rows)
    if (1 <= yMax) {
      polyline(ctx, [[ml, y(1)], [W - mr, y(1)]], { color: theme.muted, width: 1, alpha: 0.5, dash: [4, 4] });
      ctx.fillStyle = theme.muted;
      ctx.font = `11px ${ROMAN_FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("C = 1", ml + 4, y(1) - 2);
    }

    // the full C(ρ) curve (faint), and the swept-in part up to current ρ (bright)
    const full: Pt[] = curve.map((cc, i) => [x((i / (SWEEP - 1)) * RHO_MAX), y(cc)]);
    polyline(ctx, full, { color: theme.green, width: 1.6, alpha: 0.32 });
    const upto = full.filter((_, i) => (i / (SWEEP - 1)) * RHO_MAX <= rho);
    if (upto.length > 1) {
      glow(ctx, theme.green, 4, () => polyline(ctx, upto, { color: theme.green, width: 2.4, alpha: 1 }));
    }

    // current ρ marker (the movable cream variable)
    const cNow = clamp(capAt(k, rho), 0, yMax);
    strokeLine(ctx, [x(rho), H - mb], [x(rho), y(cNow)], theme.cream, 1.2, 0.6);
    glow(ctx, theme.cream, 7, () => disc(ctx, [x(rho), y(cNow)], 4, theme.cream, 1));
    mathLabel(ctx, "ρ", [x(rho) + (rho < RHO_MAX * 0.9 ? 7 : -7), mt + 12], {
      color: theme.cream,
      size: 15,
      align: rho < RHO_MAX * 0.9 ? "left" : "right",
      glow: 5,
    });

    // x ticks
    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const r of [0, 0.25, 0.5, 0.75, RHO_MAX]) ctx.fillText(r.toFixed(2), x(r), H - mb + 4);
  }

  // --- interaction ---------------------------------------------------------
  function setRho(r: number, animate = false): void {
    const t = clamp(r, 0, RHO_MAX);
    if (animate) rhoTracker.set(t, true);
    else rhoTracker.jump(t);
  }

  function setSize(nk: number): void {
    k = clamp(Math.round(nk), K_MIN, K_MAX);
    rebuild();
    render();
  }

  rhoRow.input.addEventListener("input", () => setRho(Number(rhoRow.input.value), false));
  kRow.input.addEventListener("input", () => setSize(Number(kRow.input.value)));

  // Drag along the C(ρ) trace canvas → set ρ from the horizontal position.
  const stopDrag = draggable(
    rcTrace.canvas,
    (px) => {
      const ml = 30;
      const mr = 16;
      const plotW = rcTrace.width - ml - mr;
      rhoTracker.jump(clamp(((px - ml) / plotW) * RHO_MAX, 0, RHO_MAX));
    },
    { onStart: () => (rcTrace.canvas.style.cursor = "grabbing"), onEnd: () => (rcTrace.canvas.style.cursor = "grab") },
  );
  rcTrace.canvas.style.cursor = "grab";

  rcTrace.canvas.tabIndex = 0;
  rcTrace.canvas.setAttribute("role", "img");
  rcTrace.canvas.setAttribute(
    "aria-label",
    "AR(1) covariance: the banded matrix Sigma with entries rho to the power of the index distance, its eigenvalue spectrum, and the quantized-Gaussian completion capacity as a function of the correlation rho. Drag along the curve to set rho.",
  );
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setRho(rho - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setRho(rho + step, true);
    else if (e.key === "Home") setRho(0, true);
    else if (e.key === "End") setRho(RHO_MAX, true);
    else return;
    e.preventDefault();
  };
  rcTrace.canvas.addEventListener("keydown", onKey);

  // Reduced motion: settle on the initial ρ with no glide.
  if (prefersReducedMotion()) rho = clamp(options.rho ?? 0.6, 0, RHO_MAX);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setRho,
    setSize,
    destroy() {
      destroyed = true;
      rhoTracker.stop();
      stopDrag();
      rcTrace.canvas.removeEventListener("keydown", onKey);
      rcTop.destroy();
      rcTrace.destroy();
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

function sliderRow(
  theme: Theme,
  label: string,
  min: number,
  max: number,
  value: number,
  step: number,
  aria: string,
  fmt: (v: number) => string,
): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:8.5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(step),
    value: String(value),
    style: "flex:1;min-width:90px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:3.5ch;text-align:right;` });
  out.textContent = fmt(value);
  input.addEventListener("input", () => (out.textContent = fmt(Number(input.value))));
  row.append(lab, input, out);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
      out.textContent = fmt(v);
    },
  };
}

// --- helpers ----------------------------------------------------------------

/** Linear blend between two #rrggbb colors at fraction `t`, returned as #rrggbb. */
function mixHex(c0: string, c1: string, t: number): string {
  const a = parseHex(c0);
  const b = parseHex(c1);
  const f = clamp(t, 0, 1);
  const m = (i: number) => Math.round(a[i] + (b[i] - a[i]) * f);
  return `#${toHex2(m(0))}${toHex2(m(1))}${toHex2(m(2))}`;
}

function parseHex(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function toHex2(v: number): string {
  return clamp(v, 0, 255).toString(16).padStart(2, "0");
}
