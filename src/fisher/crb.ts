import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface FisherCrbOptions {
  /** True Poisson rate λ. The post uses λ = 3. */
  lambda?: number;
  /** Highlighted / histogram sample size n (slider value). Default 400 (the post). */
  n?: number;
  /** RNG seed for the Monte-Carlo resample. */
  seed?: number;
  /** Replications per n on the variance sweep (post uses 6000). */
  reps?: number;
  /** Datasets in the rescaled-error histogram (post uses 60000). */
  histReps?: number;
  theme?: Partial<Theme>;
}

export interface FisherCrbApi {
  /** Set the sample size n (eased); resamples the histogram and moves the variance marker. */
  setN(n: number, animate?: boolean): void;
  /** Set the true rate λ; recomputes the whole sweep and the floor. */
  setLambda(lambda: number): void;
  /** Draw a fresh Monte-Carlo realization from the next seed. */
  reseed(): void;
  destroy(): void;
}

// The post's logspace(0.7, 3.2, 18), rounded & de-duplicated — the sample sizes
// on the variance sweep. Computed once; identical to np.logspace.
const N_GRID: number[] = (() => {
  const raw: number[] = [];
  for (let i = 0; i < 18; i++) raw.push(Math.round(10 ** (0.7 + (3.2 - 0.7) * (i / 17))));
  const out: number[] = [];
  for (const v of raw) if (out[out.length - 1] !== v) out.push(v);
  return out;
})();
const N_MIN = N_GRID[0];
const N_MAX = N_GRID[N_GRID.length - 1];

/**
 * Scene — "the inverse Fisher is a variance". Ported from the post's notebook
 * cell (Poisson MLE, λ = 3): the MLE is the sample mean, the per-sample
 * information is I₁ = 1/λ, and the Cramér–Rao floor is 1/(n I₁) = λ/n. Two views,
 * driven live by in-browser Poisson sampling:
 *
 * Left (log–log): for a grid of sample sizes n, the empirical Var(λ̂) over many
 * replications (bright blue points) rides exactly on the gold Cramér–Rao floor
 * λ/n. A cream marker tracks the chosen n.
 *
 * Right: the rescaled error √n(λ̂−λ) is histogrammed (blue bars) and collapses
 * onto the gold limit law N(0, 1/I₁ = λ) as n grows — asymptotic normality of
 * the MLE, made out of real samples. The green outline traces the empirical
 * density. Slider for n (live-resamples both), slider for λ, a re-seed control.
 */
export function createFisherCrb(target: HTMLElement, options: FisherCrbOptions = {}): FisherCrbApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let lambda = clamp(options.lambda ?? 3, 0.5, 12);
  let nSel = clamp(Math.round(options.n ?? 400), N_MIN, N_MAX);
  let seed = (options.seed ?? 11) >>> 0 || 1;
  // Replication counts. The notebook uses 6000 / 60000; in the browser those are
  // recomputed live (the sweep once per λ·seed, the histogram on every n change),
  // so the defaults are trimmed for interactivity. Higher counts only smooth the
  // Monte-Carlo estimates — the Cramér–Rao floor and the limit law are exact
  // regardless of how many replications back them.
  const reps = Math.max(200, Math.round(options.reps ?? 1800));
  const histReps = Math.max(1000, Math.round(options.histReps ?? 9000));
  let destroyed = false;

  // Cached Monte-Carlo products. The sweep depends on (λ, seed); the histogram
  // depends on (λ, n, seed). Recomputed only when those change, never per frame.
  let sweepVar: number[] = []; // empirical Var(λ̂) at each N_GRID[i]
  const HIST_BINS = 60;
  let histCounts = new Float64Array(HIST_BINS);
  let histLo = -8;
  let histHi = 8;
  let histPeak = 1e-6;

  /** One Poisson(λ) draw via Knuth's algorithm on the seeded uniform stream. */
  function poisson(rng: Rng, lam: number): number {
    const L = Math.exp(-lam);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rng.next();
    } while (p > L);
    return k - 1;
  }

  /** Mean of n Poisson(λ) samples — the MLE λ̂ for one dataset. */
  function poissonMean(rng: Rng, n: number, lam: number): number {
    let s = 0;
    for (let i = 0; i < n; i++) s += poisson(rng, lam);
    return s / n;
  }

  // --- the variance sweep (left panel): Var(λ̂) vs n ----------------------
  function buildSweep(): void {
    const rng = new Rng(seed);
    sweepVar = N_GRID.map((n) => {
      let m = 0;
      let m2 = 0;
      for (let r = 0; r < reps; r++) {
        const lh = poissonMean(rng, n, lambda);
        m += lh;
        m2 += lh * lh;
      }
      m /= reps;
      m2 /= reps;
      return Math.max(m2 - m * m, 1e-12); // population variance, like numpy .var()
    });
  }

  // --- the rescaled-error histogram (right panel) -------------------------
  function buildHist(): void {
    // Mirror the notebook: z = √n (λ̂ − λ) over histReps datasets of n samples.
    const rng = new Rng(seed ^ 0x9e3779b9);
    const z = new Float64Array(histReps);
    let zmin = Infinity;
    let zmax = -Infinity;
    const root = Math.sqrt(nSel);
    for (let r = 0; r < histReps; r++) {
      const v = root * (poissonMean(rng, nSel, lambda) - lambda);
      z[r] = v;
      if (v < zmin) zmin = v;
      if (v > zmax) zmax = v;
    }
    // A symmetric window around 0 that comfortably holds the bulk and the curve.
    const span = Math.max(Math.abs(zmin), Math.abs(zmax), 4 * Math.sqrt(lambda));
    histLo = -span;
    histHi = span;
    histCounts = new Float64Array(HIST_BINS);
    const bw = (histHi - histLo) / HIST_BINS;
    for (let r = 0; r < histReps; r++) {
      let b = Math.floor((z[r] - histLo) / bw);
      if (b < 0) b = 0;
      else if (b >= HIST_BINS) b = HIST_BINS - 1;
      histCounts[b] += 1;
    }
    // Normalize to a density (∫ = 1), exactly like density=True.
    histPeak = 1e-6;
    for (let b = 0; b < HIST_BINS; b++) {
      histCounts[b] /= histReps * bw;
      if (histCounts[b] > histPeak) histPeak = histCounts[b];
    }
  }

  // The limit law N(0, σ² = 1/I₁ = λ): density and its peak.
  const limitDensity = (x: number): number => Math.exp(-(x * x) / (2 * lambda)) / Math.sqrt(2 * Math.PI * lambda);

  function recomputeAll(): void {
    buildSweep();
    buildHist();
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  // Two stacked sub-panels, like projection.ts (main + trace).
  const sweepHost = el("div", { style: "position:relative;" });
  const histHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(sweepHost, histHost);
  root.append(panel);

  // top-left of the sweep panel: the floor and the live empirical variance
  const overlay = overlayLayer("tl");
  const eqFloor = liveEquation("\\dfrac{1}{n\\,I_1}=\\dfrac{\\lambda}{n}\\;=", theme, theme.gold);
  const eqVar = liveEquation("\\widehat{\\operatorname{Var}}(\\hat\\lambda)\\;=", theme, theme.blue);
  overlay.append(eqFloor.node, eqVar.node);
  sweepHost.append(overlay);

  // top-right: the model facts (information per sample, the CRB statement)
  const factOverlay = overlayLayer("tr");
  factOverlay.append(
    staticEquation("I_1=1/\\lambda,\\quad \\hat\\lambda=\\bar X", theme, theme.muted),
    staticEquation("\\operatorname{Var}(\\hat\\lambda)\\ge\\dfrac{1}{n\\,I_1}", theme, theme.gold),
  );
  sweepHost.append(factOverlay);

  // tag on the histogram sub-panel
  const histTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.green};font:12px ${theme.mono};z-index:2;`,
  });
  histTag.textContent = "√n (λ̂ − λ)  →  𝒩(0, 1/I₁)";
  histHost.append(histTag);

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";

  const nLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  nLabel.textContent = "sample size  n";
  const nSlider = el("input", {
    type: "range",
    min: String(N_MIN),
    max: String(N_MAX),
    step: "1",
    value: String(nSel),
    style: "flex:1;min-width:120px;max-width:200px;",
    "aria-label": "sample size n for the rescaled-error histogram and the highlighted variance point",
  });
  const nOut = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:4ch;text-align:right;` });
  nOut.textContent = String(nSel);

  const lamLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.cream};` });
  lamLabel.textContent = "rate  λ";
  const lamSlider = el("input", {
    type: "range",
    min: "0.5",
    max: "12",
    step: "0.5",
    value: String(lambda),
    style: "flex:1;min-width:110px;max-width:180px;",
    "aria-label": "true Poisson rate lambda",
  });
  const lamOut = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.cream};min-width:3.5ch;text-align:right;` });
  lamOut.textContent = lambda.toFixed(1);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ · reseed · ←→";
  controls.append(reseedBtn, nLabel, nSlider, nOut, lamLabel, lamSlider, lamOut, hint);
  root.append(controls);

  target.append(root);

  // --- canvases + the one eased scalar (the chosen n) ----------------------
  const rcSweep = responsiveCanvas(sweepHost, 2.1, () => render());
  const rcHist = responsiveCanvas(histHost, 2.4, () => render());

  // n eases on its log axis so the marker and the histogram resample glide.
  const logN = (n: number) => Math.log(n);
  // The tracker eases only the left-panel marker (the cream dot gliding along the
  // empirical curve). The histogram tracks the *committed* n (nSel), resampled
  // exactly once per selection — not on every intermediate eased integer — so the
  // glide stays smooth even though each resample draws nSel·histReps Poissons.
  let nDisp = nSel; // eased display n, marker position on the left panel
  const nTracker = valueTracker(logN(nSel), (v) => {
    nDisp = Math.exp(v);
    render();
  });

  /** Commit a new n: snap the histogram to it (one resample) and ease the marker. */
  function commitN(n: number, animate: boolean): void {
    const target = clamp(Math.round(n), N_MIN, N_MAX);
    const changed = target !== nSel;
    nSel = target;
    if (changed) buildHist();
    if (animate) nTracker.set(logN(target), true);
    else nTracker.jump(logN(target));
    render();
  }

  // first build
  recomputeAll();

  // --- interpolate the cached sweep at a continuous n (for the marker) -----
  function sweepVarAt(n: number): number {
    const ln = logN(clamp(n, N_MIN, N_MAX));
    // find bracketing grid indices in log space
    let i = 0;
    while (i < N_GRID.length - 1 && logN(N_GRID[i + 1]) < ln) i++;
    const j = Math.min(i + 1, N_GRID.length - 1);
    const a = logN(N_GRID[i]);
    const b = logN(N_GRID[j]);
    const t = b > a ? (ln - a) / (b - a) : 0;
    // interpolate in log–log (the relationship is a straight line there)
    const la = Math.log(sweepVar[i]);
    const lb = Math.log(sweepVar[j]);
    return Math.exp(la * (1 - t) + lb * t);
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    drawSweep();
    drawHist();

    const floor = lambda / nSel;
    eqFloor.set(floor.toFixed(4));
    eqVar.set(sweepVarAt(nSel).toFixed(4));

    if (document.activeElement !== nSlider) {
      nSlider.value = String(nSel);
      nOut.textContent = String(nSel);
    }
    if (document.activeElement !== lamSlider) {
      lamSlider.value = String(lambda);
      lamOut.textContent = lambda.toFixed(1);
    }
  }

  function drawSweep(): void {
    const { ctx, width: W, height: H } = rcSweep;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 52;
    const mr = 16;
    const mt = 14;
    const mb = 32;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // log–log extents. x over the n grid; y over the floor & empirical variance.
    const xLo = logN(N_MIN);
    const xHi = logN(N_MAX);
    let yMaxV = lambda / N_MIN;
    let yMinV = lambda / N_MAX;
    for (const v of sweepVar) {
      if (v > yMaxV) yMaxV = v;
      if (v > 0 && v < yMinV) yMinV = v;
    }
    const yLo = Math.log10(yMinV) - 0.18;
    const yHi = Math.log10(yMaxV) + 0.12;
    const X = (n: number) => ml + ((logN(n) - xLo) / (xHi - xLo)) * plotW;
    const Y = (v: number) => mt + plotH - ((Math.log10(v) - yLo) / (yHi - yLo)) * plotH;

    // faint decade grid (BLUE_D), like a receded log number plane
    for (let d = Math.ceil(xLo / Math.LN10); 10 ** d <= N_MAX + 1; d++) {
      const gx = X(10 ** d);
      if (gx >= ml - 0.5 && gx <= W - mr + 0.5) strokeLine(ctx, [gx, mt], [gx, mt + plotH], theme.grid, 1, 0.26);
    }
    for (let d = Math.ceil(yLo); d <= Math.floor(yHi); d++) {
      const gy = Y(10 ** d);
      strokeLine(ctx, [ml, gy], [W - mr, gy], theme.grid, 1, 0.26);
    }
    // axes
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);

    // the Cramér–Rao floor λ/n — a straight line of slope −1 in log–log (the answer)
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[X(N_MIN), Y(lambda / N_MIN)], [X(N_MAX), Y(lambda / N_MAX)]], {
        color: theme.gold,
        width: 1.8,
        alpha: 0.9,
        dash: [7, 6],
      });
    });

    // empirical Var(λ̂): bright blue points joined by a faint line
    const pts: Pt[] = N_GRID.map((n, i) => [X(n), Y(sweepVar[i])]);
    polyline(ctx, pts, { color: theme.blue, width: 1.5, alpha: 0.55 });
    glow(ctx, theme.blue, 4, () => {
      for (const p of pts) disc(ctx, p, 3, theme.blue, 0.95);
    });

    // the chosen-n marker (the movable variable): cream vertical rule + a dot on
    // the empirical curve, with a companion tick on the floor it is racing.
    const nx = X(nDisp);
    strokeLine(ctx, [nx, mt], [nx, mt + plotH], theme.cream, 1.3, 0.6);
    glow(ctx, theme.cream, 7, () => disc(ctx, [nx, Y(sweepVarAt(nDisp))], 4, theme.cream, 1));
    disc(ctx, [nx, Y(lambda / nDisp)], 2.6, theme.gold, 0.95);

    // y ticks (powers of ten, roman) + x ticks (decades)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let d = Math.ceil(yLo); d <= Math.floor(yHi); d++) {
      ctx.fillText(`10`, ml - 14, Y(10 ** d));
      ctx.save();
      ctx.font = `8px ${ROMAN_FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(String(d), ml - 13, Y(10 ** d) - 5);
      ctx.restore();
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let d = Math.ceil(xLo / Math.LN10); 10 ** d <= N_MAX + 1; d++) {
      const gx = X(10 ** d);
      if (gx >= ml - 0.5 && gx <= W - mr + 0.5) ctx.fillText(`10`, gx, mt + plotH + 5);
    }
    // axis titles
    mathLabel(ctx, "n", [W - mr - 2, mt + plotH + 24], { color: theme.tick, size: 15, align: "right" });
    ctx.save();
    ctx.translate(13, mt + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    mathLabel(ctx, "variance", [0, 0], { color: theme.tick, size: 12, italic: false, align: "center" });
    ctx.restore();
  }

  function drawHist(): void {
    const { ctx, width: W, height: H } = rcHist;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 44;
    const mr = 16;
    const mt = 26;
    const mb = 30;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const yMax = Math.max(histPeak, limitDensity(0)) * 1.12;
    const X = (z: number) => ml + ((clamp(z, histLo, histHi) - histLo) / (histHi - histLo)) * plotW;
    const Y = (d: number) => mt + plotH - (clamp(d, 0, yMax) / yMax) * plotH;

    // faint vertical grid at integer z (the receded plane)
    const zTickStep = histHi - histLo > 12 ? 2 : 1;
    for (let z = Math.ceil(histLo); z <= Math.floor(histHi); z += zTickStep) {
      strokeLine(ctx, [X(z), mt], [X(z), mt + plotH], theme.grid, 1, z === 0 ? 0.34 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    // histogram bars: blue, translucent (the empirical sampling distribution)
    const bw = (histHi - histLo) / HIST_BINS;
    const px = (plotW / HIST_BINS);
    for (let b = 0; b < HIST_BINS; b++) {
      const x0 = X(histLo + b * bw);
      const top = Y(histCounts[b]);
      ctx.fillStyle = theme.blue;
      ctx.globalAlpha = 0.42;
      ctx.fillRect(x0, top, Math.max(1, px - 0.6), mt + plotH - top);
      ctx.globalAlpha = 1;
    }
    // green step outline tracing the empirical density
    const outline: Pt[] = [];
    for (let b = 0; b < HIST_BINS; b++) {
      const x0 = X(histLo + b * bw);
      const x1 = X(histLo + (b + 1) * bw);
      const yb = Y(histCounts[b]);
      outline.push([x0, yb], [x1, yb]);
    }
    polyline(ctx, outline, { color: theme.green, width: 1.4, alpha: 0.85 });

    // the limit law N(0, λ): a smooth gold curve, faintly glowing (the answer)
    const curve: Pt[] = [];
    const S = 200;
    for (let i = 0; i <= S; i++) {
      const z = histLo + (histHi - histLo) * (i / S);
      curve.push([X(z), Y(limitDensity(z))]);
    }
    glow(ctx, theme.gold, 6, () => polyline(ctx, curve, { color: theme.gold, width: 2.4, alpha: 1 }));

    // mark the center and ±σ (= ±√λ) of the limit law
    strokeLine(ctx, [X(0), Y(0)], [X(0), Y(limitDensity(0))], theme.gold, 1, 0.4);
    const sig = Math.sqrt(lambda);
    for (const s of [-sig, sig]) {
      strokeLine(ctx, [X(s), mt + plotH], [X(s), Y(limitDensity(s))], theme.gold, 1, 0.32);
    }

    // x ticks (roman) and titles
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let z = Math.ceil(histLo); z <= Math.floor(histHi); z += zTickStep) ctx.fillText(String(z), X(z), mt + plotH + 5);

    mathLabel(ctx, "n", [W - mr - 44, mt + 4], { color: theme.cream, size: 14, align: "right" });
    ctx.fillStyle = theme.cream;
    ctx.font = `13px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(` = ${nSel}`, W - mr - 44, mt + 4 - 12);

    // legend
    drawLegend(ctx, W - mr, mt);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["empirical", theme.green, []],
      ["𝒩(0, λ)", theme.gold, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = top + 26;
    for (const [name, color, dash] of rows) {
      const lx = right - 92;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
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
  function setN(n: number, animate = false): void {
    commitN(n, animate);
  }

  function setLambda(lam: number): void {
    lambda = clamp(Math.round(lam * 2) / 2, 0.5, 12);
    recomputeAll();
    render();
  }

  function reseed(): void {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0 || 1;
    recomputeAll();
    render();
  }

  reseedBtn.addEventListener("click", reseed);
  nSlider.addEventListener("input", () => {
    nOut.textContent = nSlider.value;
    // jump the marker directly (continuous control), and resample the histogram
    // for the newly committed n.
    commitN(Number(nSlider.value), false);
  });
  lamSlider.addEventListener("input", () => {
    lamOut.textContent = Number(lamSlider.value).toFixed(1);
    setLambda(Number(lamSlider.value));
  });

  // Horizontal drag on the sweep canvas scrubs n along its log axis. The marker
  // follows the pointer continuously; the histogram resamples only when the
  // rounded n actually changes.
  const stopDragSweep = draggable(
    rcSweep.canvas,
    (px) => {
      const ml = 52;
      const mr = 16;
      const plotW = rcSweep.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      const ln = logN(N_MIN) + f * (logN(N_MAX) - logN(N_MIN));
      const target = clamp(Math.round(Math.exp(ln)), N_MIN, N_MAX);
      if (target !== nSel) {
        nSel = target;
        buildHist();
      }
      nTracker.jump(ln); // marker glides directly to the pointer (continuous)
    },
    { onStart: () => (rcSweep.canvas.style.cursor = "grabbing"), onEnd: () => (rcSweep.canvas.style.cursor = "grab") },
  );
  rcSweep.canvas.style.cursor = "grab";

  rcSweep.canvas.tabIndex = 0;
  rcSweep.canvas.setAttribute("role", "img");
  rcSweep.canvas.setAttribute(
    "aria-label",
    "Log-log plot of the empirical variance of the Poisson MLE against the sample size n; the blue points ride on the gold Cramér–Rao floor one over n times I one. A cream marker tracks the chosen n.",
  );
  rcHist.canvas.tabIndex = 0;
  rcHist.canvas.setAttribute("role", "img");
  rcHist.canvas.setAttribute(
    "aria-label",
    "Histogram of the rescaled MLE error root-n times lambda-hat minus lambda, collapsing onto the gold normal limit law with mean zero and variance lambda as n grows.",
  );

  const onKey = (e: KeyboardEvent) => {
    // step along the n grid; Shift jumps by decade-ish strides
    const idx = nearestGridIndex(nSel);
    const stride = e.shiftKey ? 3 : 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setN(N_GRID[clamp(idx - stride, 0, N_GRID.length - 1)], true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setN(N_GRID[clamp(idx + stride, 0, N_GRID.length - 1)], true);
    else return;
    e.preventDefault();
  };
  rcSweep.canvas.addEventListener("keydown", onKey);
  rcHist.canvas.addEventListener("keydown", onKey);

  function nearestGridIndex(n: number): number {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < N_GRID.length; i++) {
      const d = Math.abs(logN(N_GRID[i]) - logN(n));
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  // Reduced motion: land directly on the selected n (no glide).
  if (prefersReducedMotion()) nDisp = nSel;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setN,
    setLambda,
    reseed,
    destroy() {
      destroyed = true;
      nTracker.stop();
      stopDragSweep();
      rcSweep.canvas.removeEventListener("keydown", onKey);
      rcHist.canvas.removeEventListener("keydown", onKey);
      rcSweep.destroy();
      rcHist.destroy();
      root.remove();
    },
  };
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
