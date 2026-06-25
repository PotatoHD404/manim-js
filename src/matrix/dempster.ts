import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import type { Mat } from "../core/math/linalg";
import { type Theme, withTheme } from "../core/theme";

export interface DempsterOptions {
  /** Matrix dimension k of the base covariance (the chain graph). Default 6. */
  dim?: number;
  /** AR(1) correlation parameter ρ ∈ (0,1) — the chain coupling. Default 0.6. */
  rho?: number;
  /**
   * The unobserved off-diagonal entry (i, j), 1-indexed as in the post (M₁,₄).
   * Defaults to [1, 4]; i and j must be non-adjacent on the chain so that the
   * conditional-independence (max-entropy) completion is well defined.
   */
  entry?: [number, number];
  /** Initial value at the unobserved entry, as a fraction of the feasible window. */
  start?: number;
  theme?: Partial<Theme>;
}

export interface DempsterApi {
  /** Set the value at the unobserved entry M_ij (eased when animate). */
  setEntry(value: number, animate?: boolean): void;
  /** Re-parameterize the chain (AR(1) ρ), rebuilding the base covariance. */
  setRho(rho: number): void;
  /** Snap the swept entry to the max-entropy optimum (precision = 0). */
  snapToOptimum(): void;
  /** Sweep the entry across the feasible window once. */
  play(): void;
  destroy(): void;
}

/**
 * Scene — "max-entropy completion = sparse precision" (Dempster's covariance
 * selection). The base is an AR(1) chain covariance `M`, M_ab = ρ^|a−b| — the
 * same `buildAR1` the post's demo library uses. One off-diagonal entry (the
 * unobserved pair M₁,₄, two hops apart on the chain) is left free and swept by
 * the user. For every value we compute, in real time and exactly, two scalars:
 * the entropy `log₂ det M` (blue, left axis) and the precision entry `(M⁻¹)₁,₄`
 * (gold, right axis). Among all positive-definite completions the entropy peaks
 * precisely where the precision entry crosses zero — the KKT condition of
 * Dempster's theorem made visible: `d/dθ log det M = 2 (M⁻¹)₁,₄`, so the maximum
 * of the blue curve sits exactly under the zero-crossing of the gold line, at
 * the conditional-independence value ρ³. A second slider re-parameterizes the
 * chain; the coincidence is invariant.
 */
export function createMatrixDempster(target: HTMLElement, options: DempsterOptions = {}): DempsterApi {
  const theme = withTheme(options.theme);
  const k = clamp(Math.round(options.dim ?? 6), 4, 10);
  const entry1 = options.entry?.[0] ?? 1;
  const entry2 = options.entry?.[1] ?? 4;
  // 0-indexed positions of the unobserved entry (i < j by construction).
  const ei = clamp(Math.min(entry1, entry2) - 1, 0, k - 1);
  const ej = clamp(Math.max(entry1, entry2) - 1, 0, k - 1);
  const labelIJ = `${entry1},${entry2}`;

  // --- model state ---------------------------------------------------------
  let rho = clamp(options.rho ?? 0.6, 0.1, 0.92);
  let destroyed = false;

  // The base AR(1) chain covariance, rebuilt when ρ changes. The (ei, ej) entry
  // is the free one; everything else is "observed".
  let base: Mat = buildAR1(k, rho);
  // Feasible interval [lo, hi] of the swept entry keeping M positive-definite,
  // the optimum x* (= the natural AR(1) value, the conditional-independence
  // completion), and the displayed sweep window centred on x*.
  let feasLo = -1;
  let feasHi = 1;
  let xStar = 0;
  let winLo = -0.5;
  let winHi = 0.5;
  let x = 0; // current value at the unobserved entry (displayed, eased)

  // Precomputed curves over the display window, refreshed on ρ change / resize.
  const CURVE_N = 241;
  let curveX = new Float64Array(CURVE_N);
  let curveDet = new Float64Array(CURVE_N); // log₂ det M
  let curvePrec = new Float64Array(CURVE_N); // (M⁻¹)_{ij}
  let detPeak = 1;
  let detFloor = 0;
  let precMax = 1; // symmetric extent of the precision axis

  function rebuild(initFrac?: number): void {
    base = buildAR1(k, rho);
    const f = feasibleInterval(base, ei, ej);
    feasLo = f.lo;
    feasHi = f.hi;
    xStar = base[ei][ej]; // AR(1) value ρ^|i−j| — the max-entropy completion
    // A symmetric display window around the optimum, inside the PD interval.
    const halfPossible = Math.min(xStar - feasLo, feasHi - xStar);
    const half = halfPossible * 0.86;
    winLo = xStar - half;
    winHi = xStar + half;

    detPeak = -Infinity;
    detFloor = Infinity;
    precMax = 1e-6;
    for (let i = 0; i < CURVE_N; i++) {
      const xi = winLo + (winHi - winLo) * (i / (CURVE_N - 1));
      curveX[i] = xi;
      const d = log2det(base, ei, ej, xi);
      const pr = precisionEntry(base, ei, ej, xi);
      curveDet[i] = d;
      curvePrec[i] = pr;
      if (d > detPeak) detPeak = d;
      if (d < detFloor) detFloor = d;
      if (Math.abs(pr) > precMax) precMax = Math.abs(pr);
    }
    precMax *= 1.08;
    if (initFrac !== undefined) x = winLo + (winHi - winLo) * clamp(initFrac, 0, 1);
    else x = clamp(x, winLo, winHi);
  }
  rebuild(clamp(options.start ?? 0.5, 0, 1));

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the two live quantities being traced
  const overlay = overlayLayer("tl");
  const eqDet = liveEquation("\\log_2\\det M\\;=", theme, theme.blue);
  const eqPrec = liveEquation("(M^{-1})_{" + labelIJ + "}\\;=", theme, theme.gold);
  const eqVal = liveEquation("M_{" + labelIJ + "}\\;=", theme, theme.cream);
  overlay.append(eqDet.node, eqPrec.node, eqVal.node);
  panel.append(overlay);

  // top-right: the theorem, restated
  const lawOverlay = overlayLayer("tr");
  lawOverlay.append(
    staticEquation("\\arg\\max_{M\\succ0}\\,\\log\\det M \\;\\Leftrightarrow\\; (M^{-1})_{" + labelIJ + "}=0", theme, theme.green),
    staticEquation("\\tfrac{d}{d\\theta}\\log\\det M = 2\\,(M^{-1})_{" + labelIJ + "}", theme, theme.muted),
  );
  panel.append(lawOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => draw());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const snapBtn = el("button", { type: "button", style: btnStyle(theme) });
  snapBtn.innerHTML = "&#9650;&nbsp; snap to optimum";

  const entrySlider = el("input", {
    type: "range",
    min: "0",
    max: "1000",
    step: "1",
    value: String(Math.round(fracOf(x) * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": `value at the unobserved entry M ${labelIJ}`,
  });
  controls.append(playBtn, snapBtn, entrySlider);

  const rhoRow = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const rhoLab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  rhoLab.textContent = "chain ρ";
  const rhoSlider = el("input", {
    type: "range",
    min: "20",
    max: "90",
    step: "1",
    value: String(Math.round(rho * 100)),
    style: "flex:1;min-width:90px;max-width:160px;",
    "aria-label": "AR(1) chain correlation rho",
  });
  const rhoOut = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:3.2ch;text-align:right;` });
  rhoOut.textContent = rho.toFixed(2);
  rhoRow.append(rhoLab, rhoSlider, rhoOut);
  controls.append(rhoRow);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · sweep · ←→  ·  ρ re-parameterizes";
  controls.append(hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the swept entry value -------------------------
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  const xTracker = valueTracker(x, (v) => {
    x = clamp(v, winLo, winHi);
    draw();
    syncReadouts();
  });

  function fracOf(v: number): number {
    return winHi > winLo ? clamp((v - winLo) / (winHi - winLo), 0, 1) : 0;
  }

  // --- render --------------------------------------------------------------
  function syncReadouts(): void {
    const d = log2det(base, ei, ej, x);
    const pr = precisionEntry(base, ei, ej, x);
    eqDet.set(d.toFixed(3));
    eqPrec.set(pr.toFixed(3));
    eqVal.set(x.toFixed(3));
    if (document.activeElement !== entrySlider) entrySlider.value = String(Math.round(fracOf(x) * 1000));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 52; // room for the blue (entropy) axis
    const mr = 56; // room for the gold (precision) axis
    const mt = 18;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const px = (v: number) => ml + clamp((v - winLo) / (winHi - winLo), 0, 1) * plotW;
    // independent vertical scales for the two curves (dual axis)
    const detPad = (detPeak - detFloor) * 0.12 + 1e-6;
    const detLo = detFloor - detPad;
    const detHi = detPeak + detPad;
    const yDet = (v: number) => mt + plotH - ((v - detLo) / (detHi - detLo)) * plotH;
    const yPrec = (v: number) => mt + plotH / 2 - (v / precMax) * (plotH / 2);

    // faint number-plane grid (the cyan plane, receding)
    for (let g = 0; g <= 4; g++) {
      const gx = ml + (g / 4) * plotW;
      strokeLine(ctx, [gx, mt], [gx, mt + plotH], theme.grid, 1, g === 0 || g === 4 ? 0.26 : 0.14);
    }
    // axes frame
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.blue, 1.4, 0.9);
    strokeLine(ctx, [W - mr, mt], [W - mr, mt + plotH], theme.gold, 1.4, 0.9);
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);

    // the precision-zero line (the answer condition): a green dotted rule
    const yZero = yPrec(0);
    polyline(ctx, [[ml, yZero], [W - mr, yZero]], { color: theme.green, width: 1.3, alpha: 0.7, dash: [2, 5] });

    // the optimum: vertical gold guide where entropy peaks == precision = 0
    const optX = px(xStar);
    strokeLine(ctx, [optX, mt], [optX, mt + plotH], theme.gold, 1.2, 0.5);

    // entropy curve (blue, bright, faintly glowing) — the objective
    const detPts: Pt[] = [];
    for (let i = 0; i < CURVE_N; i++) detPts.push([px(curveX[i]), yDet(curveDet[i])]);
    glow(ctx, theme.blue, 5, () => polyline(ctx, detPts, { color: theme.blue, width: 2.6, alpha: 1 }));

    // precision curve (gold, the KKT quantity that must vanish)
    const precPts: Pt[] = [];
    for (let i = 0; i < CURVE_N; i++) precPts.push([px(curveX[i]), yPrec(curvePrec[i])]);
    glow(ctx, theme.gold, 4, () => polyline(ctx, precPts, { color: theme.gold, width: 2.4, alpha: 0.95 }));

    // optimum markers on both curves
    glow(ctx, theme.gold, 7, () => {
      disc(ctx, [optX, yDet(detPeak)], 4, theme.gold, 1);
      disc(ctx, [optX, yZero], 4, theme.gold, 1);
    });

    // the movable variable: the swept entry — a cream vertical line + read-offs
    const cx = px(x);
    const dHere = log2det(base, ei, ej, x);
    const prHere = precisionEntry(base, ei, ej, x);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.3, 0.7);
    glow(ctx, theme.cream, 7, () => disc(ctx, [cx, yDet(dHere)], 3.8, theme.cream, 1));
    disc(ctx, [cx, yPrec(prHere)], 3.2, theme.cream, 0.95);
    mathLabel(ctx, "M", [cx + (x < (winLo + winHi) / 2 ? 7 : -7), mt + 16], {
      color: theme.cream,
      size: 15,
      sub: labelIJ,
      align: x < (winLo + winHi) / 2 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "\\log_2\\det M", [ml - 6, mt + 4], { color: theme.blue, size: 13, align: "right", baseline: "top" });
    mathLabel(ctx, "(M^{-1})", [W - mr + 6, mt + 4], { color: theme.gold, size: 13, align: "left", baseline: "top", sub: labelIJ });

    ctx.font = `11px ${ROMAN_FONT}`;
    // left (entropy) ticks
    ctx.fillStyle = theme.blue;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of niceTicks(detLo, detHi, 4)) ctx.fillText(v.toFixed(2), ml - 6, yDet(v));
    // right (precision) ticks, symmetric about 0
    ctx.fillStyle = theme.gold;
    ctx.textAlign = "left";
    for (const v of symTicks(precMax, 3)) ctx.fillText(v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1), W - mr + 6, yPrec(v));
    // x ticks (the entry value)
    ctx.fillStyle = theme.tick;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of niceTicks(winLo, winHi, 5)) ctx.fillText(v.toFixed(2), px(v), mt + plotH + 6);
    mathLabel(ctx, "M", [W - mr - 6, mt + plotH + 20], { color: theme.tick, size: 14, sub: labelIJ, align: "right" });

    // a small "completion" badge near the optimum
    ctx.fillStyle = theme.gold;
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    ctx.textAlign = xStar < (winLo + winHi) / 2 ? "left" : "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("max-entropy", optX + (xStar < (winLo + winHi) / 2 ? 5 : -5), mt + 14);
  }

  // --- interaction ---------------------------------------------------------
  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function setEntry(value: number, animate = false): void {
    stopSweep();
    const v = clamp(value, winLo, winHi);
    if (animate) xTracker.set(v, true);
    else xTracker.jump(v);
  }

  function setRho(nextRho: number): void {
    stopSweep();
    rho = clamp(nextRho, 0.1, 0.92);
    const keep = fracOf(x);
    rebuild(keep);
    xTracker.jump(x);
    rhoOut.textContent = rho.toFixed(2);
    if (document.activeElement !== rhoSlider) rhoSlider.value = String(Math.round(rho * 100));
    draw();
    syncReadouts();
  }

  function snapToOptimum(): void {
    setEntry(xStar, true);
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setEntry(xStar, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    let s = fracOf(x);
    sweepCancel = ticker((dt) => {
      s += dt / 3.6;
      if (s >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        xTracker.set(xStar, true);
        return false;
      }
      xTracker.jump(winLo + (winHi - winLo) * s);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; sweep entry";
  }
  updatePlayBtn();

  playBtn.addEventListener("click", play);
  snapBtn.addEventListener("click", snapToOptimum);
  entrySlider.addEventListener("input", () => {
    stopSweep();
    xTracker.jump(winLo + (winHi - winLo) * (Number(entrySlider.value) / 1000));
  });
  rhoSlider.addEventListener("input", () => setRho(Number(rhoSlider.value) / 100));

  const stopDrag = draggable(
    rc.canvas,
    (px2) => {
      stopSweep();
      const ml = 52;
      const mr = 56;
      const plotW = rc.width - ml - mr;
      const frac = clamp((px2 - ml) / plotW, 0, 1);
      xTracker.jump(winLo + (winHi - winLo) * frac);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    `Dempster covariance selection: sweeping the unobserved entry M ${labelIJ} of an AR(1) chain covariance, the log-base-2 determinant (entropy) and the precision-matrix entry are plotted together. The entropy peaks exactly where the precision entry crosses zero — the maximum-entropy completion is the one with a sparse precision matrix.`,
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      snapToOptimum();
      return;
    }
    const span = winHi - winLo;
    const step = (e.shiftKey ? 0.06 : 0.015) * span;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setEntry(x - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setEntry(x + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  draw();
  syncReadouts();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) {
        draw();
        syncReadouts();
      }
    });
  }

  return {
    setEntry,
    setRho,
    snapToOptimum,
    play,
    destroy() {
      destroyed = true;
      xTracker.stop();
      sweepCancel?.();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- math --------------------------------------------------------------------

/**
 * AR(1) chain covariance, M_ab = ρ^|a−b| — the post's `buildAR1`. Its precision
 * matrix is tridiagonal, so any pair more than one hop apart (e.g. 1 and 4) is
 * conditionally independent: that entry's precision is already zero, and the
 * AR(1) value ρ^|i−j| is precisely the maximum-entropy completion.
 */
function buildAR1(k: number, rho: number): Mat {
  const m: Mat = [];
  for (let i = 0; i < k; i++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) row.push(Math.pow(rho, Math.abs(i - j)));
    m.push(row);
  }
  return m;
}

/** A copy of `base` with the symmetric (i, j) entry set to `value`. */
function withEntry(base: Mat, i: number, j: number, value: number): Mat {
  const m = base.map((r) => r.slice());
  m[i][j] = value;
  m[j][i] = value;
  return m;
}

/** Base-2 log-determinant of the completed matrix via LU (returns −∞ if not PD). */
function log2det(base: Mat, i: number, j: number, value: number): number {
  const m = withEntry(base, i, j, value);
  const ln = lnDetSym(m);
  return Number.isFinite(ln) ? ln / Math.LN2 : -Infinity;
}

/** The (i, j) entry of the inverse of the completed matrix. */
function precisionEntry(base: Mat, i: number, j: number, value: number): number {
  const inv = inverse(withEntry(base, i, j, value));
  return inv ? inv[i][j] : NaN;
}

/**
 * The feasible interval of the swept entry that keeps the completion positive
 * definite, found by bisecting outward from the natural value until the
 * Cholesky decomposition fails. Symmetric search in both directions.
 */
function feasibleInterval(base: Mat, i: number, j: number): { lo: number; hi: number } {
  const x0 = base[i][j];
  const pd = (v: number) => Number.isFinite(lnDetSym(withEntry(base, i, j, v))) && choleskyOK(withEntry(base, i, j, v));
  const edge = (dir: 1 | -1): number => {
    let good = x0;
    let bad = x0;
    let stepv = 0.05;
    // grow until we leave the PD region (or hit |entry| ≈ 1)
    while (Math.abs(bad) < 0.999) {
      bad = x0 + dir * stepv;
      if (!pd(bad)) break;
      good = bad;
      stepv *= 1.6;
    }
    if (pd(bad)) return good; // never left PD within range
    for (let it = 0; it < 60; it++) {
      const mid = (good + bad) / 2;
      if (pd(mid)) good = mid;
      else bad = mid;
    }
    return good;
  };
  return { lo: edge(-1), hi: edge(1) };
}

/** ln det of a symmetric positive-(semi)definite matrix via Cholesky; −∞ if not PD. */
function lnDetSym(a: Mat): number {
  const n = a.length;
  const l: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  let lnDet = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let p = 0; p < j; p++) s -= l[i][p] * l[j][p];
      if (i === j) {
        if (s <= 0) return -Infinity;
        l[i][j] = Math.sqrt(s);
        lnDet += 2 * Math.log(l[i][j]);
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return lnDet;
}

/** Whether `a` admits a Cholesky factor (i.e. is positive definite). */
function choleskyOK(a: Mat): boolean {
  return Number.isFinite(lnDetSym(a));
}

/** Inverse of a symmetric matrix via Gauss–Jordan; null if singular. */
function inverse(a: Mat): Mat | null {
  const n = a.length;
  const m = a.map((r, i) => [...r, ...Array.from({ length: n }, (_, c) => (c === i ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-14) return null;
    if (piv !== col) {
      const tmp = m[piv];
      m[piv] = m[col];
      m[col] = tmp;
    }
    const inv = 1 / m[col][col];
    for (let c = 0; c < 2 * n; c++) m[col][c] *= inv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let c = 0; c < 2 * n; c++) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row) => row.slice(n));
}

/** ~`count` round-number ticks spanning [lo, hi]. */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  return out;
}

/** Symmetric ticks about 0 out to ±`max`, ~`perSide` per side (excludes 0). */
function symTicks(max: number, perSide: number): number[] {
  const raw = max / perSide;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = step; v <= max + step * 1e-6; v += step) {
    out.push(v);
    out.push(-v);
  }
  return out;
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
