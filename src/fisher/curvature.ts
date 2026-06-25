import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface FisherCurvatureOptions {
  /** Initial sample size n (the slider's value). */
  n?: number;
  /** Smallest n the slider reaches. */
  minN?: number;
  /** Largest n the slider reaches. */
  maxN?: number;
  /** True rate λ⋆ of the exponential model (mean 1/λ⋆). The post uses 1. */
  lambdaTrue?: number;
  /** RNG seed for the Monte-Carlo draw (the post fixes seed 3). */
  seed?: number;
  theme?: Partial<Theme>;
}

export interface FisherCurvatureApi {
  /** Set the sample size n (eased glide when `animate`). */
  setN(n: number, animate?: boolean): void;
  /** Sweep n from min to max and settle at the top (Manim-style traversal). */
  play(): void;
  /** Redraw a fresh exponential sample from the next seed. */
  reseed(): void;
  destroy(): void;
}

/**
 * Scene — "curvature at the MLE is information". For an exponential model with
 * rate λ, the total log-likelihood ℓ_n(λ) = n(log λ − λ x̄) peaks at the MLE
 * λ̂ = 1/x̄. Near that peak it is a downward parabola whose second derivative is
 * the observed information J_n = n/λ̂². The gold second-order Taylor parabola
 * −½J(λ−λ̂)² hugs the blue log-likelihood at the apex, and the osculating circle
 * of radius r = 1/J nestles inside it. Slide n upward: J grows like n, the peak
 * sharpens, and the circle shrinks — a live readout shows r = 1/J collapsing.
 * Small information → flat peak, big circle, high-variance estimate; large
 * information → sharp spike, tiny circle, a pinned-down estimate.
 *
 * Ported verbatim from the post's notebook cell (exponential, λ⋆ = 1, seed 3):
 * λ̂ = 1/x̄, J = n/λ̂², r = 1/J, parabola −½J(λ−λ̂)², circle centred at (λ̂, −r),
 * drawn on an equal-aspect axis so the osculating circle reads as a circle.
 */
export function createFisherCurvature(
  target: HTMLElement,
  options: FisherCurvatureOptions = {},
): FisherCurvatureApi {
  const theme = withTheme(options.theme);
  const lamTrue = options.lambdaTrue ?? 1.0;
  const minN = Math.max(2, Math.round(options.minN ?? 3));
  const maxN = Math.max(minN + 1, Math.round(options.maxN ?? 120));
  let seed = (options.seed ?? 3) >>> 0 || 3;

  // --- model: one fresh exponential draw of maxN samples, sub-sampled by n ---
  // The notebook draws `r.exponential(1/λ⋆, n)` fresh for each n. To keep the
  // figure continuous as the slider moves, we draw a single long stream once per
  // seed and read the running mean of its first n entries — the same estimator,
  // λ̂ = 1/x̄, just evaluated incrementally. A reseed redraws the whole stream.
  let stream: Float64Array = new Float64Array(maxN);
  let prefix: Float64Array = new Float64Array(maxN + 1); // prefix sums for x̄
  function drawStream(): void {
    const rng = new Rng(seed);
    stream = new Float64Array(maxN);
    prefix = new Float64Array(maxN + 1);
    for (let i = 0; i < maxN; i++) {
      // inverse-CDF exponential with rate λ⋆ (mean 1/λ⋆): x = −ln(1−u)/λ⋆.
      let u = rng.next();
      if (u <= 0) u = 1e-12;
      const x = -Math.log(1 - u) / lamTrue;
      stream[i] = x;
      prefix[i + 1] = prefix[i] + x;
    }
  }
  drawStream();

  /** Sample mean x̄ of the first n draws (n need not be an integer; we ease it). */
  function xbarAt(nReal: number): number {
    const n = clamp(nReal, 1, maxN);
    const lo = Math.floor(n);
    const hi = Math.min(lo + 1, maxN);
    const frac = n - lo;
    // running sum interpolated between ⌊n⌋ and ⌈n⌉ samples
    const sum = prefix[lo] + frac * (prefix[hi] - prefix[lo]);
    return sum / n;
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the live observed information and osculating radius
  const overlay = overlayLayer("tl");
  const eqN = liveEquation("n\\;=", theme, theme.cream);
  const eqJ = liveEquation("J_n=-\\,\\ell_n''(\\hat\\lambda)=", theme, theme.green);
  const eqR = liveEquation("r=1/J_n=", theme, theme.gold);
  overlay.append(eqN.node, eqJ.node, eqR.node);
  mainHost.append(overlay);

  // top-right: the static model identities the picture realizes
  const idOverlay = overlayLayer("tr");
  idOverlay.append(
    staticEquation("\\ell_n(\\lambda)=n\\,(\\log\\lambda-\\lambda\\bar x)", theme, theme.blue),
    staticEquation("\\hat\\lambda=1/\\bar x,\\quad J_n=n/\\hat\\lambda^{\\,2}", theme, theme.gold),
  );
  mainHost.append(idOverlay);

  const rc = responsiveCanvas(mainHost, 1.42, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "sample size  n";
  const slider = el("input", {
    type: "range",
    min: String(minN),
    max: String(maxN),
    step: "1",
    value: String(clamp(Math.round(options.n ?? 6), minN, maxN)),
    style: `flex:1;min-width:150px;max-width:260px;accent-color:${theme.cream};`,
    "aria-label": "sample size n of the exponential model",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ · grow n · reseed · ←→";
  controls.append(playBtn, reseedBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- state ---------------------------------------------------------------
  let nShown = clamp(options.n ?? 6, minN, maxN); // eased display value of n
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  // n is the one eased scalar; the parabola, circle, and view all read from it.
  const nTracker = valueTracker(nShown, (v) => {
    nShown = clamp(v, minN, maxN);
    render();
  });

  // --- the closed forms, exactly as in the notebook ------------------------
  interface Model {
    n: number;
    xbar: number;
    lamHat: number;
    J: number;
    R: number;
  }
  function modelAt(nReal: number): Model {
    const n = clamp(nReal, minN, maxN);
    const xbar = xbarAt(n);
    const lamHat = 1 / xbar; // MLE
    const J = n / (lamHat * lamHat); // observed information −ℓ''(λ̂)
    const R = 1 / J; // osculating radius
    return { n, xbar, lamHat, J, R };
  }

  /** ℓ_n(λ) − ℓ_n(λ̂): the log-likelihood shifted so its peak sits at 0. */
  function loglikShifted(lam: number, m: Model): number {
    const ll = m.n * (Math.log(lam) - lam * m.xbar);
    const llHat = m.n * (Math.log(m.lamHat) - m.lamHat * m.xbar);
    return ll - llHat;
  }

  // --- render --------------------------------------------------------------
  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function render(): void {
    const m = modelAt(nShown);
    draw(m);
    const nInt = Math.round(nShown);
    eqN.set(String(nInt));
    eqJ.set(m.J.toFixed(1));
    eqR.set(m.R.toFixed(3));
    if (document.activeElement !== slider) slider.value = String(nInt);
  }

  function draw(m: Model): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    // Equal-aspect window centred on the MLE, matching the notebook's
    // set_aspect("equal") with xlim = λ̂ ± 2.2r and ylim = [−2.4r, 0.45r]. We
    // honour the y-range and widen x just enough to keep the aspect square, so
    // the osculating circle is a true circle on screen.
    const ml = 50;
    const mr = 18;
    const mt = 18;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const yLo = -2.4 * m.R;
    const yHi = 0.45 * m.R;
    const worldH = yHi - yLo;
    // one common pixels-per-world-unit so circles stay round
    const s = Math.min(plotW / (4.4 * m.R), plotH / worldH);
    const cxPix = ml + plotW / 2;
    // place the world point (λ̂, 0)…(λ̂, yLo) so the chosen y-range is centred
    const yMidWorld = (yHi + yLo) / 2;
    const cyPix = mt + plotH / 2;
    const X = (lam: number) => cxPix + (lam - m.lamHat) * s;
    const Y = (yy: number) => cyPix - (yy - yMidWorld) * s;

    const xMinWorld = m.lamHat - plotW / 2 / s;
    const xMaxWorld = m.lamHat + plotW / 2 / s;

    // faint number plane: vertical grid at nice λ steps, horizontal at nice y.
    const gx = niceStep(xMaxWorld - xMinWorld);
    ctx.lineWidth = 1;
    for (let g = Math.ceil(xMinWorld / gx) * gx; g <= xMaxWorld; g += gx) {
      strokeLine(ctx, [X(g), Y(yHi)], [X(g), Y(yLo)], theme.grid, 1, 0.26);
    }
    const gy = niceStep(worldH);
    for (let g = Math.ceil(yLo / gy) * gy; g <= yHi; g += gy) {
      strokeLine(ctx, [X(xMinWorld), Y(g)], [X(xMaxWorld), Y(g)], theme.grid, 1, 0.26);
    }

    // axes: the λ-axis (y = 0, where the peak sits) and a vertical guide at λ̂.
    strokeLine(ctx, [X(xMinWorld), Y(0)], [X(xMaxWorld), Y(0)], theme.axis, 1.4, 1);

    // --- osculating circle: radius r = 1/J, centred at (λ̂, −r) (the answer) --
    // Drawn first so the curves overlay its rim.
    const circR = m.R * s;
    const ccx = X(m.lamHat);
    const ccy = Y(-m.R);
    glow(ctx, theme.gold, 6, () => {
      ctx.strokeStyle = theme.gold;
      ctx.globalAlpha = 0.92;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(ccx, ccy, circR, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    // a thin radius spoke from the apex to the circle centre, labelled r
    strokeLine(ctx, [ccx, Y(0)], [ccx, ccy], theme.gold, 1, 0.5);
    disc(ctx, [ccx, ccy], 2.4, theme.muted, 0.9);
    mathLabel(ctx, "r", [ccx + 6, (Y(0) + ccy) / 2], { color: theme.gold, size: 14, glow: 4 });

    // --- the curves, sampled on the notebook's grid g over the window --------
    const G = 240;
    const llPts: Pt[] = [];
    const quadPts: Pt[] = [];
    for (let i = 0; i < G; i++) {
      const lam = xMinWorld + (i / (G - 1)) * (xMaxWorld - xMinWorld);
      if (lam <= 1e-3) continue; // notebook filters g > 1e-3 (log domain)
      const ll = loglikShifted(lam, m);
      const quad = -0.5 * m.J * (lam - m.lamHat) * (lam - m.lamHat);
      llPts.push([X(lam), Y(ll)]);
      quadPts.push([X(lam), Y(quad)]);
    }
    // 2nd-order Taylor parabola (gold, dashed) — what the circle's apex matches
    polyline(ctx, quadPts, { color: theme.gold, width: 1.6, alpha: 0.8, dash: [7, 5] });
    // the true log-likelihood (bright blue, faintly glowing) — the figure
    ctx.save();
    ctx.shadowColor = theme.blue;
    ctx.shadowBlur = 6;
    polyline(ctx, llPts, { color: theme.blue, width: 2.4, alpha: 1 });
    ctx.restore();

    // --- the MLE point (the apex, where gradient vanishes) -------------------
    glow(ctx, theme.cream, 8, () => disc(ctx, [X(m.lamHat), Y(0)], 4.4, theme.cream, 1));
    mathLabel(ctx, "λ", [X(m.lamHat) + 8, Y(0) - 9], {
      color: theme.cream,
      size: 17,
      sub: "MLE",
      glow: 5,
    });

    // λ-axis ticks (roman numerals) at the nice grid steps
    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = Math.ceil(xMinWorld / gx) * gx; g <= xMaxWorld; g += gx) {
      if (Math.abs(g) < 1e-9) continue;
      ctx.fillText(fmtTick(g), X(g), Y(0) + 6);
    }
    mathLabel(ctx, "λ", [X(xMaxWorld) - 6, Y(0) + 24], { color: theme.tick, size: 15, align: "right" });

    // y-axis label: the log-likelihood, with its peak pinned at 0
    ctx.save();
    ctx.translate(16, mt + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = theme.muted;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("log-likelihood  (peak at 0)", 0, 0);
    ctx.restore();

    // a one-line verdict tying curvature to variance, like the post's captions
    const sharp = m.J > (minN + maxN) / (lamTrueSq() * 2);
    ctx.fillStyle = theme.muted;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      sharp ? "Fisher large → sharp peak → low variance" : "Fisher small → flat peak → high variance",
      ml + 4,
      H - mb - 6,
    );
  }

  function lamTrueSq(): number {
    return lamTrue * lamTrue;
  }

  // --- interaction ---------------------------------------------------------
  function setN(n: number, animate = false): void {
    stopSweep();
    const v = clamp(Math.round(n), minN, maxN);
    if (animate) nTracker.set(v, true);
    else nTracker.jump(v);
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
    // traverse n from the bottom to the top on a smooth log schedule, so the
    // early sharpening (small n) is as visible as the late (large n).
    let u = (Math.log(nShown / minN)) / Math.log(maxN / minN);
    u = clamp(u, 0, 1);
    sweepCancel = ticker((dt) => {
      u += dt / 4;
      if (u >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        nTracker.set(maxN, true);
        return false;
      }
      const n = minN * Math.pow(maxN / minN, u);
      nTracker.jump(n);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; grow n";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    drawStream();
    render();
  }
  reseedBtn.addEventListener("click", reseed);

  slider.addEventListener("input", () => {
    stopSweep();
    nTracker.jump(Number(slider.value));
  });

  // Horizontal drag on the canvas scrubs n across [minN, maxN].
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      stopSweep();
      const ml = 50;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      nTracker.jump(minN + f * (maxN - minN));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "ew-resize") },
  );
  rc.canvas.style.cursor = "ew-resize";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Curvature at the maximum likelihood estimate. The log-likelihood of an exponential model is drawn with its second-order Taylor parabola and the osculating circle of radius one over the observed information J. As the sample size n grows, J grows, the peak sharpens, and the osculating circle shrinks, so the estimate is pinned down more precisely.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const step = e.shiftKey ? 10 : 1;
    let n = nShown;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") n -= step;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") n += step;
    else return;
    e.preventDefault();
    setN(n, true);
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on a sharp peak immediately (no glide to watch).
  if (prefersReducedMotion()) nShown = clamp(options.n ?? 6, minN, maxN);

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
    reseed,
    destroy() {
      destroyed = true;
      nTracker.stop();
      sweepCancel?.();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- helpers ----------------------------------------------------------------

/** A round-number grid step roughly dividing `span` into ~6 intervals. */
function niceStep(span: number): number {
  const raw = span / 6 || 1;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}

/** Tick label: integers plain, otherwise trimmed to the grid's precision. */
function fmtTick(v: number): string {
  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
  const a = Math.abs(v);
  const dp = a >= 1 ? 1 : a >= 0.1 ? 2 : 3;
  return v.toFixed(dp);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
