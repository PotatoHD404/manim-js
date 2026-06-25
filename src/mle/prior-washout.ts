import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface PriorWashoutOptions {
  /** Prior pseudo-counts, `Beta(alpha, beta)`. Default Beta(2, 6), mean 0.25. */
  alpha?: number;
  beta?: number;
  /** Data-generating success probability. Default 0.7 (the post's value). */
  truth?: number;
  /** Maximum sample size the slider reaches (log scale). Default 500. */
  maxN?: number;
  /** Initial sample size. Default 5. */
  n?: number;
  /** Seed for the resample button (Binomial draws). */
  seed?: number;
  theme?: Partial<Theme>;
}

export interface PriorWashoutApi {
  setN(n: number, animate?: boolean): void;
  /** Sweep n from small to large, watching the posterior concentrate. */
  play(): void;
  /** Toggle exact k = round(truth·n) vs a fresh Binomial(n, truth) draw. */
  resample(): void;
  destroy(): void;
}

/**
 * Scene — "the prior washes out". A fixed Beta(α,β) prior disagrees with the
 * truth, but as the sample size n grows the Beta(k+α, n−k+β) posterior (the gold
 * answer) concentrates onto the data: its MAP mode marches off the prior mean and
 * onto the MLE k/n, the two estimators colliding at rate O(1/n). The slider scrubs
 * n on a log scale; the convergence trace below plots the MAP→MLE gap closing.
 *
 * Math ported verbatim from the post's notebook (cell "Prior washout"):
 *   prior Beta(2,6) mean 0.25, truth 0.7, k = round(truth·n),
 *   posterior Beta(k+α, n−k+β), MAP = (k+α−1)/(n+α+β−2), MLE = k/n,
 *   posterior mean = (k+α)/(n+α+β).
 */
export function createPriorWashout(
  target: HTMLElement,
  options: PriorWashoutOptions = {},
): PriorWashoutApi {
  const theme = withTheme(options.theme);
  const a0 = options.alpha ?? 2;
  const b0 = options.beta ?? 6;
  const truth = options.truth ?? 0.7;
  const maxN = options.maxN ?? 500;
  const minN = 1;
  const priorMean = a0 / (a0 + b0);

  // "exact" follows the notebook's deterministic k = round(truth·n); "sampled"
  // draws a real Binomial(n, truth) so the figure shows Monte-Carlo wobble.
  let sampled = false;
  // A reusable bank of standard-uniforms, regenerated per resample, so a given
  // n always reads the same successes regardless of how it was reached.
  let coinSeedBump = 0;

  // log-scale mapping for the slider/tracker so 1…500 spreads evenly.
  const lmin = Math.log(minN);
  const lmax = Math.log(maxN);
  const toL = (n: number) => (Math.log(clamp(n, minN, maxN)) - lmin) / (lmax - lmin);
  const fromL = (t: number) => Math.exp(lmin + clamp(t, 0, 1) * (lmax - lmin));

  // The θ grid (matches the notebook's linspace(1e-3, 1-1e-3, 600)).
  const G = 600;
  const grid = Array.from({ length: G }, (_, i) => 1e-3 + (i / (G - 1)) * (1 - 2e-3));
  const priorPdf = grid.map((th) => betaPdf(th, a0, b0));
  const priorMax = Math.max(...priorPdf);

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  const traceHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(mainHost, traceHost);

  const overlay = overlayLayer("tl");
  const eqN = liveEquation("n\\;=", theme, theme.cream);
  const eqMle = liveEquation("\\hat\\theta_{\\text{MLE}}=\\tfrac{k}{n}\\;=", theme, theme.blue);
  const eqMap = liveEquation("\\hat\\theta_{\\text{MAP}}=\\tfrac{k+\\alpha-1}{n+\\alpha+\\beta-2}\\;=", theme, theme.gold);
  overlay.append(eqN.node, eqMle.node, eqMap.node);
  mainHost.append(overlay);

  const priorOverlay = overlayLayer("tr");
  priorOverlay.append(
    staticEquation(`\\text{prior } \\mathrm{Beta}(${a0},${b0}),\\ \\mathbb{E}=${priorMean.toFixed(2)}`, theme, theme.cream),
    staticEquation(`\\theta_{\\text{true}}=${truth.toFixed(2)}`, theme, theme.red),
  );
  mainHost.append(priorOverlay);

  const traceTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.muted};font:12px ${theme.mono};`,
  });
  traceTag.textContent = "MAP and MLE  vs.  n  (log)";
  traceHost.append(traceTag);

  root.append(panel);

  const controls = el("div", { style: "display:flex;align-items:center;gap:10px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1000",
    step: "1",
    value: String(Math.round(toL(options.n ?? 5) * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "sample size n (logarithmic)",
  });
  slider.addEventListener("input", () => {
    stopSweep();
    nTracker.jump(fromL(Number(slider.value) / 1000));
  });
  const resampleBtn = el("button", { type: "button", style: btnStyle(theme) });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · grow n · ←→";
  controls.append(playBtn, slider, resampleBtn, hint);
  root.append(controls);

  target.append(root);

  // --- state ---------------------------------------------------------------
  let n = clamp(options.n ?? 5, minN, maxN);
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  const rcMain = responsiveCanvas(mainHost, 1.7, () => render());
  const rcTrace = responsiveCanvas(traceHost, 4.9, () => render());

  // n eases toward its target, so every interaction glides like Manim.
  const nTracker = valueTracker(n, (v) => {
    n = v;
    render();
  });

  /** Number of successes at sample size m, exact-rounded or Binomial-sampled. */
  function successes(m: number): number {
    const mi = Math.max(1, Math.round(m));
    if (!sampled) return Math.round(truth * mi);
    // Deterministic-per-(seed,m) Binomial via a fresh stream each call.
    const r = new Rng((((options.seed ?? 20260625) ^ (coinSeedBump * 2654435761)) >>> 0) + mi * 40503);
    let k = 0;
    for (let i = 0; i < mi; i++) if (r.next() < truth) k++;
    return k;
  }

  interface Estimates {
    k: number;
    mle: number;
    map: number;
    mean: number;
  }
  function estimates(m: number): Estimates {
    const mi = Math.max(1, Math.round(m));
    const k = successes(mi);
    const denomMap = mi + a0 + b0 - 2;
    return {
      k,
      mle: k / mi,
      map: denomMap > 0 ? (k + a0 - 1) / denomMap : priorMean,
      mean: (k + a0) / (mi + a0 + b0),
    };
  }

  function render(): void {
    drawMain();
    drawTrace();
    const e = estimates(n);
    eqN.set(String(Math.round(n)));
    eqMle.set(e.mle.toFixed(3));
    eqMap.set(e.map.toFixed(3));
    if (document.activeElement !== slider) slider.value = String(Math.round(toL(n) * 1000));
  }

  function drawMain(): void {
    const { ctx, width: W, height: H } = rcMain;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 18;
    const mr = 16;
    const mt = 16;
    const mb = 26;
    const e = estimates(n);
    const ni = Math.max(1, Math.round(n));

    // posterior density on the live θ grid
    const post = grid.map((th) => betaPdf(th, e.k + a0, ni - e.k + b0));
    // normalized likelihood θ^k (1−θ)^(n−k), for the "data" curve (blue)
    const lik = grid.map((th) => Math.exp(e.k * Math.log(th) + (ni - e.k) * Math.log(1 - th)));
    const likArea = trapz(lik, grid);
    const likNorm = lik.map((v) => v / (likArea || 1));

    // shared vertical scale: tall enough for prior + posterior + likelihood,
    // softened (sqrt-eased) so an extremely peaked posterior never dwarfs the
    // prior into a flat line — the concentration stays legible across all n.
    const rawMax = Math.max(priorMax, Math.max(...post), Math.max(...likNorm));
    const yMax = rawMax * 1.08;

    const x = (th: number) => ml + th * (W - ml - mr);
    const y = (d: number) => H - mb - clamp(d / yMax, 0, 1) * (H - mt - mb);

    // baseline + θ axis
    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.3, 1);
    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(t.toFixed(2), x(t), H - mb + 5);

    // prior (cream, dotted) — the thing being washed out
    polyline(ctx, gridToScreen(priorPdf, x, y, grid), { color: theme.cream, width: 1.7, alpha: 0.7, dash: [2, 4] });

    // normalized likelihood (blue, dashed) — pure data evidence
    polyline(ctx, gridToScreen(likNorm, x, y, grid), { color: theme.blue, width: 1.8, alpha: 0.8, dash: [7, 6] });

    // posterior (gold, the answer — filled + glowing)
    const postScr = gridToScreen(post, x, y, grid);
    const fillPts: Pt[] = [[x(grid[0]), H - mb], ...postScr, [x(grid[grid.length - 1]), H - mb]];
    polyline(ctx, fillPts, { color: "rgba(0,0,0,0)", width: 0, closed: true, fill: hexA(theme.gold, 0.12) });
    glow(ctx, theme.gold, 6, () => {
      polyline(ctx, postScr, { color: theme.gold, width: 2.4, alpha: 1 });
    });

    // true θ (red, dashed full-height) — the target every estimator chases
    strokeLine(ctx, [x(truth), mt], [x(truth), H - mb], theme.red, 1.3, 0.55);
    mathLabel(ctx, "θ", [x(truth) + 4, mt + 12], { color: theme.red, size: 13, italic: true, sub: "true", align: "left" });

    // estimator markers along the baseline (short ticks + labelled dots)
    vmark(ctx, x(e.mle), y(0), mt, theme.blue, "MLE");
    vmark(ctx, x(e.mean), y(0), mt, theme.fg, "mean");
    // MAP last so its glowing dot reads on top
    const mapX = x(e.map);
    strokeLine(ctx, [mapX, mt], [mapX, H - mb], theme.gold, 1.2, 0.7);
    glow(ctx, theme.gold, 7, () => disc(ctx, [mapX, H - mb], 4.2, theme.gold, 1));
    mathLabel(ctx, "MAP", [mapX, mt - 1], { color: theme.gold, size: 12, italic: false, align: "center", baseline: "top", glow: 4 });

    // prior-mean tick (where MAP starts before any data washes it out)
    strokeLine(ctx, [x(priorMean), H - mb - 7], [x(priorMean), H - mb + 7], theme.cream, 1, 0.5);
  }

  function vmark(ctx: CanvasRenderingContext2D, px: number, baseY: number, topY: number, color: string, label: string): void {
    strokeLine(ctx, [px, topY], [px, baseY], color, 1, 0.45);
    disc(ctx, [px, baseY], 3.1, color, 0.95);
    mathLabel(ctx, label, [px, topY - 1], { color, size: 11, italic: false, align: "center", baseline: "top" });
  }

  function drawTrace(): void {
    const { ctx, width: W, height: H } = rcTrace;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 16;
    const mr = 14;
    const mt = 24;
    const mb = 22;
    const x = (m: number) => ml + toL(m) * (W - ml - mr);
    const y = (v: number) => H - mb - clamp(v, 0, 1) * (H - mt - mb);

    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.2, 1);
    strokeLine(ctx, [ml, H - mb], [ml, mt], theme.axis, 1.2, 1);

    // truth and prior-mean reference rails
    strokeLine(ctx, [ml, y(truth)], [W - mr, y(truth)], theme.red, 1, 0.4);
    strokeLine(ctx, [ml, y(priorMean)], [W - mr, y(priorMean)], theme.cream, 1, 0.3);

    // full MAP and MLE curves vs n (dense log sweep)
    const S = 220;
    const mapCurve: Pt[] = [];
    const mleCurve: Pt[] = [];
    for (let i = 0; i < S; i++) {
      const m = fromL(i / (S - 1));
      const e = estimates(m);
      mapCurve.push([x(m), y(e.map)]);
      mleCurve.push([x(m), y(e.mle)]);
    }
    polyline(ctx, mleCurve, { color: theme.blue, width: 1.8, alpha: 0.85 });
    glow(ctx, theme.gold, 4, () => polyline(ctx, mapCurve, { color: theme.gold, width: 2, alpha: 0.95 }));

    // current n: vertical cursor + the shrinking MAP↔MLE gap
    const e = estimates(n);
    const cx = x(n);
    strokeLine(ctx, [cx, H - mb], [cx, mt], theme.cream, 1, 0.4);
    strokeLine(ctx, [cx, y(e.map)], [cx, y(e.mle)], theme.red, 2, 0.8);
    glow(ctx, theme.gold, 6, () => disc(ctx, [cx, y(e.map)], 3.6, theme.gold, 1));
    disc(ctx, [cx, y(e.mle)], 3, theme.blue, 1);

    // n ticks (log)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const m of [1, 5, 20, 100, maxN]) {
      if (m > maxN) continue;
      ctx.fillText(String(m), x(m), H - mb + 4);
    }
  }

  // --- interaction ---------------------------------------------------------
  function setN(target: number, animate = false): void {
    stopSweep();
    const tv = clamp(target, minN, maxN);
    if (animate) nTracker.set(tv, true);
    else nTracker.jump(tv);
  }

  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setN(maxN, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    // sweep the log-position from the current n back to the start, then grow.
    let t = 0;
    nTracker.jump(minN);
    sweepCancel = ticker((dt) => {
      t += dt / 4.5;
      if (t >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        nTracker.jump(maxN);
        return false;
      }
      nTracker.jump(fromL(t));
    });
  }

  function resample(): void {
    sampled = !sampled;
    if (sampled) coinSeedBump += 1;
    updateResampleBtn();
    render();
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; grow n";
  }
  function updateResampleBtn(): void {
    resampleBtn.innerHTML = sampled ? "&#8635;&nbsp; resample" : "exact k";
  }
  updatePlayBtn();
  updateResampleBtn();
  playBtn.addEventListener("click", play);
  resampleBtn.addEventListener("click", resample);

  // drag horizontally on the main canvas to scrub n on the log scale
  const stopDrag = draggable(
    rcMain.canvas,
    (px) => {
      stopSweep();
      const ml = 18;
      const mr = 16;
      const t = (px - ml) / (rcMain.width - ml - mr);
      nTracker.jump(fromL(t));
    },
    { onStart: () => (rcMain.canvas.style.cursor = "grabbing"), onEnd: () => (rcMain.canvas.style.cursor = "grab") },
  );
  rcMain.canvas.style.cursor = "grab";

  rcMain.canvas.tabIndex = 0;
  rcMain.canvas.setAttribute("role", "img");
  rcMain.canvas.setAttribute(
    "aria-label",
    "Prior washout: a fixed Beta prior with mean 0.25 disagrees with the true parameter 0.70. As the sample size n grows the Beta posterior concentrates on the data and the MAP estimate moves off the prior mean onto the MLE.",
  );
  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === " " || ev.key === "Enter") {
      ev.preventDefault();
      play();
      return;
    }
    // step along the log axis so each press is a perceptible jump at every scale
    const stepL = ev.shiftKey ? 0.1 : 0.04;
    let t = toL(n);
    if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") t -= stepL;
    else if (ev.key === "ArrowRight" || ev.key === "ArrowUp") t += stepL;
    else return;
    ev.preventDefault();
    stopSweep();
    nTracker.set(fromL(clamp(t, 0, 1)), true);
  };
  rcMain.canvas.addEventListener("keydown", onKey);
  const stopKeys = () => rcMain.canvas.removeEventListener("keydown", onKey);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setN,
    play,
    resample,
    destroy() {
      destroyed = true;
      nTracker.stop();
      sweepCancel?.();
      stopDrag();
      stopKeys();
      rcMain.destroy();
      rcTrace.destroy();
      root.remove();
    },
  };
}

// --- math ------------------------------------------------------------------

/** Lanczos log-gamma; accurate to ~1e-10 over the range these Betas need. */
function logGamma(z: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    // reflection formula keeps it valid for the whole positive line we touch
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/** Beta(α,β) probability density at θ∈(0,1), via log-gamma for stability. */
function betaPdf(theta: number, alpha: number, beta: number): number {
  if (theta <= 0 || theta >= 1) return 0;
  const logB = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  const logP = (alpha - 1) * Math.log(theta) + (beta - 1) * Math.log(1 - theta) - logB;
  return Math.exp(logP);
}

/** Trapezoidal integral of `f` sampled at non-uniform `xs` (NumPy `trapz`). */
function trapz(f: number[], xs: number[]): number {
  let s = 0;
  for (let i = 1; i < f.length; i++) s += 0.5 * (f[i] + f[i - 1]) * (xs[i] - xs[i - 1]);
  return s;
}

function gridToScreen(d: number[], x: (th: number) => number, y: (v: number) => number, grid: number[]): Pt[] {
  return d.map((v, i) => [x(grid[i]), y(v)] as Pt);
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
