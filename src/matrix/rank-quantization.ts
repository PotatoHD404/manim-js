import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface RankQuantizationOptions {
  /** Ambient dimension k (number of coupled rows). Defaults to the post's k = 6. */
  k?: number;
  /** Exact ranks r of the rank-deficient Gaussians to trace (each gives C → k/r). */
  ranks?: number[];
  /** Correlation ρ of the full-rank equicorrelated reference Σ = (1−ρ)I + ρ11ᵀ. */
  rho?: number;
  /** Seed for the Monte-Carlo rank-r factors F (Σ = F Fᵀ). */
  seed?: number;
  /** Initial quantization resolution Δ (the movable variable). */
  delta?: number;
  theme?: Partial<Theme>;
}

export interface RankQuantizationApi {
  /** Move the resolution Δ (eased when animate). */
  setDelta(delta: number, animate?: boolean): void;
  /** Set the equicorrelated reference's correlation ρ. */
  setRho(rho: number): void;
  /** Sharpen the resolution toward Δ → 0 (eased sweep to the limit). */
  sharpen(): void;
  /** Redraw the Monte-Carlo rank-r factors from the next seed. */
  reseed(): void;
  destroy(): void;
}

/** Δ runs on a log axis from this coarse value (left) to this fine value (right). */
const DELTA_MAX = 1.0;
const DELTA_MIN = 1e-3;
const LOG_DMAX = Math.log10(DELTA_MAX);
const LOG_DMIN = Math.log10(DELTA_MIN);

/**
 * Scene — "quantization recovers the rank". For a k-dimensional Gaussian observed
 * at resolution Δ, each eigen-direction with variance λ carries
 * max(0, ½log₂(2πeλ) − log₂Δ) bits, and the completion capacity is
 * C = S / H_joint — the per-coordinate marginal bits over the joint bits, exactly
 * the post's `gaussian_capacity_quantized(Σ, Δ)`. As Δ → 0 (finer, to the right)
 * an exactly rank-r covariance Σ = F Fᵀ freezes its k − r null directions and the
 * capacity climbs to the gold limit k/r, while a generic full-rank Σ keeps every
 * eigenvalue off zero and C → 1: one observation resolves exactly one entry.
 *
 * The cream Δ cursor sweeps the log axis; a live readout shows the capacity and
 * the implied effective rank k/C for the family at that resolution. Ported from
 * the post's notebook (k = 6, rank-r factors F = randn(k,r), full-rank
 * equicorrelated reference); the bit count matches `quantizedCapacity` verbatim.
 */
export function createRankQuantization(
  target: HTMLElement,
  options: RankQuantizationOptions = {},
): RankQuantizationApi {
  const theme = withTheme(options.theme);
  const k = clamp(Math.round(options.k ?? 6), 2, 12);
  const ranks = (options.ranks ?? [1, 2, 3]).map((r) => clamp(Math.round(r), 1, k));
  let rho = clamp(options.rho ?? 0.6, 0, 0.98);
  let seed = (options.seed ?? 0) >>> 0;
  let destroyed = false;

  // --- model (ported from notebook cell 11) --------------------------------
  // Each family is a fixed covariance Σ; capacity needs only its eigenvalues and
  // its diagonal variances. The rank-r families are Σ = F Fᵀ, F = randn(k, r),
  // so the spectrum has exactly r nonzero eigenvalues. The reference is the
  // full-rank equicorrelated Σ = (1−ρ)I + ρ11ᵀ.
  interface Family {
    label: string;
    tex: string;
    color: string;
    rank: number; // exact rank (k for full-rank); the limit is k/rank
    eigs: number[];
    diag: number[];
  }

  function rankFactorCov(r: number): Mat {
    const rng = new Rng(seed * 131 + r * 977 + 1);
    // F is k×r standard normal; Σ = F Fᵀ is exactly rank min(r, k).
    const F: number[][] = Array.from({ length: k }, () => Array.from({ length: r }, () => rng.gauss()));
    const Sig: Mat = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let c = 0; c < r; c++) s += F[i][c] * F[j][c];
        Sig[i][j] = s;
      }
    }
    return Sig;
  }

  function equicorrCov(): Mat {
    // Σ = (1−ρ)I + ρ11ᵀ — full rank for ρ < 1 (eigenvalues 1+(k−1)ρ once, 1−ρ k−1 times).
    const Sig: Mat = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) Sig[i][j] = i === j ? 1 : rho;
    return Sig;
  }

  function familyFromCov(label: string, texS: string, color: string, rank: number, Sig: Mat): Family {
    const eig = jacobiEigen(Sig);
    const eigs = eig.values.map((v) => Math.max(v, 0));
    const diag = Sig.map((row, i) => Math.max(row[i], 1e-300));
    return { label, tex: texS, color, rank, eigs, diag };
  }

  let families: Family[] = [];
  function rebuild(): void {
    const fams: Family[] = [];
    // The full-rank equicorrelated reference: the gold "answer" is C → 1.
    fams.push(familyFromCov("full rank", "\\text{full rank}", theme.green, k, equicorrCov()));
    // The rank-deficient Gaussians: each converges to its gold limit k/r.
    const palette = [theme.blue, theme.red, theme.cream];
    ranks.forEach((r, i) => {
      fams.push(familyFromCov(`rank ${r}`, `\\text{rank }${r}`, palette[i % palette.length], r, rankFactorCov(r)));
    });
    families = fams;
  }
  rebuild();

  /**
   * The post's quantized-Gaussian capacity, verbatim:
   * each direction carries bits(λ) = max(0, ½log₂(2πeλ) − log₂Δ);
   * C = Σ bits(diagᵢ) / Σ bits(eigⱼ).
   */
  function capacityAt(fam: Family, Delta: number): number {
    const bits = (lam: number) => Math.max(0, 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(lam, 1e-300)) - Math.log2(Delta));
    let S = 0;
    for (const v of fam.diag) S += bits(v);
    let Hj = 0;
    for (const l of fam.eigs) Hj += bits(l);
    return Hj < 1e-12 ? Infinity : S / Hj;
  }

  // Δ ↔ screen on the log axis (finer Δ to the right, matching the figcaption).
  const deltaToFrac = (d: number) => clamp((Math.log10(clamp(d, DELTA_MIN, DELTA_MAX)) - LOG_DMAX) / (LOG_DMIN - LOG_DMAX), 0, 1);
  const fracToDelta = (f: number) => Math.pow(10, LOG_DMAX + clamp(f, 0, 1) * (LOG_DMIN - LOG_DMAX));

  // The capacity ceiling for the plot: k/min(rank) padded a touch.
  const cMax = Math.max(2, k / Math.min(...ranks)) * 1.12;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the capacity functional + its live value at the cursor's Δ
  const overlay = overlayLayer("tl");
  const eqCap = liveEquation("C=\\dfrac{S}{H_{\\text{joint}}}\\;=", theme, theme.cream);
  const eqDelta = liveEquation("\\Delta\\;=", theme, theme.cream);
  overlay.append(eqCap.node, eqDelta.node);
  panel.append(overlay);

  // top-right: the bit count and the limit, restated for these numbers
  const idOverlay = overlayLayer("tr");
  idOverlay.append(
    staticEquation("\\text{bits}(\\lambda)=\\max\\!\\big(0,\\;\\tfrac12\\log_2 2\\pi e\\lambda-\\log_2\\Delta\\big)", theme, theme.muted),
    staticEquation(`C\\to 1\\ \\text{(full rank)},\\quad C\\to k/r\\ (k=${k})`, theme, theme.gold),
  );
  panel.append(idOverlay);

  const rc = responsiveCanvas(panel, 1.7, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const sharpenBtn = el("button", { type: "button", style: btnStyle(theme) });
  sharpenBtn.innerHTML = "&#9654;&nbsp; sharpen Δ→0";
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed F";

  let delta = clamp(options.delta ?? 0.1, DELTA_MIN, DELTA_MAX);
  const deltaLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  deltaLabel.textContent = "resolution  Δ";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1000",
    step: "1",
    value: String(Math.round(deltaToFrac(delta) * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "quantization resolution Delta, on a logarithmic axis (finer to the right)",
  });
  const rhoLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  rhoLabel.textContent = "ρ (full rank)";
  const rhoSlider = el("input", {
    type: "range",
    min: "0",
    max: "98",
    step: "1",
    value: String(Math.round(rho * 100)),
    style: "flex:1;min-width:110px;max-width:180px;",
    "aria-label": "correlation rho of the full-rank equicorrelated reference covariance",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ · sharpen · reseed · ←→";
  controls.append(sharpenBtn, reseedBtn, deltaLabel, slider, rhoLabel, rhoSlider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the Δ cursor, as a [0,1] fraction on the log axis
  let frac = deltaToFrac(delta);
  const fracTracker = valueTracker(frac, (v) => {
    frac = clamp(v, 0, 1);
    delta = fracToDelta(frac);
    render();
  });

  // --- precomputed curves (recomputed on rho / reseed) ---------------------
  const SAMPLES = 220;
  let curves: Pt[][] = []; // per family, in (frac, capacity) units before screen mapping
  function recomputeCurves(): void {
    curves = families.map((fam) => {
      const pts: Pt[] = new Array(SAMPLES);
      for (let i = 0; i < SAMPLES; i++) {
        const f = i / (SAMPLES - 1);
        const d = fracToDelta(f);
        const c = capacityAt(fam, d);
        pts[i] = [f, Number.isFinite(c) ? c : cMax];
      }
      return pts;
    });
  }
  recomputeCurves();

  // --- render --------------------------------------------------------------
  function render(): void {
    draw();

    // live readouts at the cursor's Δ, for the reference family (index 0)
    const refC = capacityAt(families[0], delta);
    eqCap.set(Number.isFinite(refC) ? refC.toFixed(3) : "∞");
    eqDelta.set(delta < 1e-2 ? delta.toExponential(1) : delta.toFixed(3));

    if (document.activeElement !== slider) slider.value = String(Math.round(frac * 1000));
    if (document.activeElement !== rhoSlider) rhoSlider.value = String(Math.round(rho * 100));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 118;
    const mt = 16;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const X = (f: number) => ml + clamp(f, 0, 1) * plotW;
    const Y = (c: number) => mt + plotH - (clamp(c, 0, cMax) / cMax) * plotH;

    // faint BLUE_D number-plane grid: vertical lines at the decade marks of Δ,
    // horizontal lines at the integer capacities.
    for (let p = 0; p <= 3; p++) {
      const d = Math.pow(10, -p); // 1, .1, .01, .001
      const gx = X(deltaToFrac(d));
      strokeLine(ctx, [gx, mt], [gx, mt + plotH], theme.grid, 1, 0.22);
    }
    for (let c = 1; c <= Math.floor(cMax); c++) {
      const gy = Y(c);
      strokeLine(ctx, [ml, gy], [W - mr, gy], theme.grid, 1, c === 1 ? 0.3 : 0.16);
    }

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // the gold "answer" limits: C = 1 (full rank) and C = k/r (rank r), dashed
    glow(ctx, theme.gold, 4, () => {
      polyline(ctx, [[ml, Y(1)], [W - mr, Y(1)]], { color: theme.gold, width: 1.4, alpha: 0.85, dash: [7, 6] });
    });
    for (const r of ranks) {
      const lim = k / r;
      if (lim > cMax) continue;
      polyline(ctx, [[ml, Y(lim)], [W - mr, Y(lim)]], { color: theme.gold, width: 1, alpha: 0.45, dash: [4, 6] });
      mathLabel(ctx, `k/${r}`, [W - mr - 4, Y(lim) - 4], { color: theme.gold, size: 12, align: "right" });
    }

    // each family's capacity curve C(Δ); the swept-in part (Δ ≥ cursor, i.e. to
    // the left of the cursor) is bright, the finer tail is faint.
    families.forEach((fam, fi) => {
      const raw = curves[fi];
      const screen: Pt[] = raw.map(([f, c]) => [X(f), Y(c)]);
      polyline(ctx, screen, { color: fam.color, width: 1.6, alpha: 0.32 });
      const swept = screen.filter((_, i) => raw[i][0] <= frac);
      if (swept.length > 1) {
        glow(ctx, fam.color, fi === 0 ? 6 : 3, () => {
          polyline(ctx, swept, { color: fam.color, width: fi === 0 ? 2.6 : 2.1, alpha: 1 });
        });
      }
    });

    // the cream Δ cursor: a vertical rule reading off every curve
    const cx = X(frac);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.6);
    families.forEach((fam) => {
      const c = capacityAt(fam, delta);
      const cc = Number.isFinite(c) ? c : cMax;
      disc(ctx, [cx, Y(cc)], fam === families[0] ? 4 : 3, fam.color, 1);
    });
    glow(ctx, theme.cream, 6, () => {
      const cRef = capacityAt(families[0], delta);
      disc(ctx, [cx, Y(Number.isFinite(cRef) ? cRef : cMax)], 4.2, theme.cream, 1);
    });
    mathLabel(ctx, "Δ", [cx + (frac < 0.9 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 15,
      align: frac < 0.9 ? "left" : "right",
      italic: false,
      glow: 5,
    });

    // x ticks: Δ decades, finer to the right
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let p = 0; p <= 3; p++) {
      const d = Math.pow(10, -p);
      ctx.fillText(p === 0 ? "1" : `10⁻${p}`, X(deltaToFrac(d)), mt + plotH + 5);
    }
    // y ticks: integer capacities
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let c = 0; c <= Math.floor(cMax); c++) ctx.fillText(String(c), ml - 7, Y(c));

    // axis labels (KaTeX serif)
    mathLabel(ctx, "C", [ml - 30, mt + 10], { color: theme.tick, size: 16 });
    mathLabel(ctx, "Δ", [W - mr - 2, mt + plotH + 20], { color: theme.tick, size: 15, align: "right", italic: false });
    ctx.fillStyle = theme.muted;
    ctx.font = `10px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("coarse", ml, mt + plotH + 19);
    ctx.textAlign = "right";
    ctx.fillText("finer →", W - mr - 30, mt + plotH + 19);

    // the side panel: per-family capacity + effective rank k/C at this Δ
    drawSidebar(ctx, W, mr, mt);
  }

  /** A compact legend with the live capacity and effective-rank read-off. */
  function drawSidebar(ctx: CanvasRenderingContext2D, W: number, mr: number, mt: number): void {
    const x0 = W - mr + 12;
    let yy = mt + 8;
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textBaseline = "middle";
    for (const fam of families) {
      const c = capacityAt(fam, delta);
      const eff = Number.isFinite(c) && c > 1e-9 ? k / c : NaN;
      // colour swatch
      glow(ctx, fam.color, fam === families[0] ? 5 : 0, () => {
        ctx.strokeStyle = fam.color;
        ctx.lineWidth = fam === families[0] ? 2.6 : 1.8;
        ctx.beginPath();
        ctx.moveTo(x0, yy);
        ctx.lineTo(x0 + 16, yy);
        ctx.stroke();
      });
      ctx.fillStyle = theme.fg;
      ctx.textAlign = "left";
      ctx.fillText(fam.label, x0 + 22, yy);
      yy += 15;
      ctx.fillStyle = fam.color;
      const cTxt = Number.isFinite(c) ? c.toFixed(2) : "∞";
      const rTxt = Number.isFinite(eff) ? eff.toFixed(2) : "—";
      ctx.fillText(`C=${cTxt}  k/C=${rTxt}`, x0 + 4, yy);
      yy += 20;
    }
    ctx.restore();
  }

  // --- interaction ---------------------------------------------------------
  function setDeltaFrac(f: number, animate: boolean): void {
    if (animate) fracTracker.set(clamp(f, 0, 1), true);
    else fracTracker.jump(clamp(f, 0, 1));
  }
  function setDelta(d: number, animate = false): void {
    setDeltaFrac(deltaToFrac(d), animate);
  }

  function setRho(r: number): void {
    rho = clamp(r, 0, 0.98);
    rebuild();
    recomputeCurves();
    render();
  }

  function sharpen(): void {
    // Sweep to the finest resolution (Δ → 0, the right edge): the curves climb
    // to their limits. Reduced motion lands there instantly.
    setDeltaFrac(1, !prefersReducedMotion());
  }

  function reseed(): void {
    seed = (seed + 1) >>> 0;
    rebuild();
    recomputeCurves();
    render();
  }

  sharpenBtn.addEventListener("click", sharpen);
  reseedBtn.addEventListener("click", reseed);
  slider.addEventListener("input", () => fracTracker.jump(Number(slider.value) / 1000));
  rhoSlider.addEventListener("input", () => setRho(Number(rhoSlider.value) / 100));

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 46;
      const mr = 118;
      const plotW = rc.width - ml - mr;
      fracTracker.jump(clamp((px - ml) / plotW, 0, 1));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Quantized-Gaussian completion capacity versus observation resolution Delta on a logarithmic axis. A full-rank covariance converges to one as the resolution sharpens, while exactly rank-r covariances converge to k over r. Drag the Delta cursor to read each capacity and the implied effective rank.",
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      sharpen();
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      setDeltaFrac(0, true);
      return;
    }
    const step = e.shiftKey ? 0.05 : 0.012;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setDeltaFrac(frac - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setDeltaFrac(frac + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: settle on the chosen Δ with no glide.
  if (prefersReducedMotion()) {
    frac = deltaToFrac(delta);
  }

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setDelta,
    setRho,
    sharpen,
    reseed,
    destroy() {
      destroyed = true;
      fracTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
