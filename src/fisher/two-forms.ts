import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface FisherTwoFormsOptions {
  /** Poisson rate λ; the closed-form information is 1/λ. Default 3 (the post). */
  lambda?: number;
  /** RNG seed for the Monte-Carlo draw. Default 7 (the notebook's draw). */
  seed?: number;
  /** Initial cursor position as a sample size N (snapped onto the log grid). */
  n?: number;
  theme?: Partial<Theme>;
}

export interface FisherTwoFormsApi {
  /** Move the N cursor (eased when animate). */
  setN(n: number, animate?: boolean): void;
  /** Change the Poisson rate λ (rebuilds the draw). */
  setLambda(lam: number): void;
  /** Draw a fresh Monte-Carlo sample from the next seed. */
  reseed(): void;
  destroy(): void;
}

/**
 * Scene — "two faces of one matrix". For the Poisson(λ) model the Fisher
 * information has two Monte-Carlo estimators of the same quantity: the
 * score-variance form E[s²] = E[(x/λ − 1)²] (blue) and the negative-Hessian
 * form −E[∂²log p] = E[x/λ²] (gold). Both running means walk along a log-N axis
 * and converge to the closed form 1/λ (the dashed gold answer line) as the
 * sample size grows. A cursor reads each running mean off at a chosen N; a slider
 * sets λ; a re-seed button redraws the Monte-Carlo sample so the convergence is
 * visibly random yet reproducible.
 *
 * Ported verbatim from the post's notebook cell: one fixed draw of the largest N
 * Poisson variates, then prefix-means of (x/λ − 1)² and x/λ² over the same
 * log-spaced grid of sample sizes (logspace(1.3, 6, 40), unique-rounded).
 */
export function createFisherTwoForms(
  target: HTMLElement,
  options: FisherTwoFormsOptions = {},
): FisherTwoFormsApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let lam = clamp(Math.round(options.lambda ?? 3), 1, 12);
  let seed = (options.seed ?? 7) >>> 0 || 1;
  let destroyed = false;

  // The notebook's sample-size grid: 40 points log-spaced from 10^1.3≈20 to 1e6,
  // rounded and de-duplicated. Drawn once at the largest N; every smaller N is a
  // prefix of the same sequence, so the running means are exact prefix means.
  const Ns = logspaceUniqueInt(1.3, 6, 40);
  const G = Ns.length;
  const Nmax = Ns[G - 1];
  const logNs = Ns.map((n) => Math.log10(n));
  const logLo = logNs[0];
  const logHi = logNs[G - 1];

  // Running-mean estimators over the grid (recomputed on reseed / λ change).
  const estScore = new Float64Array(G); // E[(x/λ − 1)²]  — variance of the score
  const estHess = new Float64Array(G); // E[x/λ²]        — negative Hessian
  let counts = new Int32Array(0); // the single fixed draw of Nmax Poisson variates

  function drawSample(): void {
    const rng = new Rng(seed);
    counts = new Int32Array(Nmax);
    for (let i = 0; i < Nmax; i++) counts[i] = poisson(rng, lam);
  }

  // Prefix means of the two per-sample quantities along the grid. Walking the
  // full draw once and snapshotting at each grid N keeps this O(Nmax) per build.
  function recompute(): void {
    const invLam = 1 / lam;
    const invLam2 = invLam * invLam;
    let sumScore = 0;
    let sumHess = 0;
    let gi = 0;
    for (let i = 0; i < Nmax; i++) {
      const x = counts[i];
      const s = x * invLam - 1; // score = x/λ − 1  (mean zero)
      sumScore += s * s; // (x/λ − 1)²
      sumHess += x * invLam2; // x/λ²  (= −∂²log p)
      const n = i + 1;
      while (gi < G && Ns[gi] === n) {
        estScore[gi] = sumScore / n;
        estHess[gi] = sumHess / n;
        gi++;
      }
    }
  }

  drawSample();
  recompute();

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the two estimators read off at the cursor, live
  const overlay = overlayLayer("tl");
  const eqScore = liveEquation("\\mathbb{E}[s^2]=", theme, theme.blue);
  const eqHess = liveEquation("-\\mathbb{E}[\\partial^2\\log p]=", theme, theme.gold);
  const eqN = liveEquation("N=", theme, theme.cream);
  overlay.append(eqScore.node, eqHess.node, eqN.node);
  panel.append(overlay);

  // top-right: the closed form being approached
  const idOverlay = overlayLayer("tr");
  const eqClosed = staticEquation("", theme, theme.gold);
  idOverlay.append(
    staticEquation("I(\\lambda)=\\mathbb{E}[s^2]=-\\mathbb{E}[\\partial^2\\log p]", theme, theme.muted),
    eqClosed,
  );
  panel.append(idOverlay);

  const rc = responsiveCanvas(panel, 1.66, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";

  const nLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  nLabel.textContent = "sample size  N";
  const nSlider = el("input", {
    type: "range",
    min: "0",
    max: String(G - 1),
    step: "1",
    value: String(nearestGridIndex(options.n ?? Ns[Math.floor(G / 2)])),
    style: "flex:1;min-width:130px;max-width:220px;",
    "aria-label": "sample size N (log scale)",
  });

  const lamRow = sliderRow(theme, "rate  λ", 1, 12, lam, "Poisson rate lambda");

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ N  ·  reseed  ·  ←→";
  controls.append(reseedBtn, nLabel, nSlider, lamRow.row, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the cursor over log10(N) ----------------------
  let logN = clamp(logNs[nearestGridIndex(options.n ?? Ns[Math.floor(G / 2)])], logLo, logHi);
  const logNTracker = valueTracker(logN, (v) => {
    logN = clamp(v, logLo, logHi);
    render();
  });

  // Linear interpolation of a grid array at a continuous log10(N).
  function estAt(arr: Float64Array, lg: number): number {
    const x = clamp(lg, logLo, logHi);
    // grid is monotone increasing in logNs; find the bracketing pair
    let i = 0;
    while (i < G - 2 && logNs[i + 1] < x) i++;
    const t = (x - logNs[i]) / (logNs[i + 1] - logNs[i] || 1);
    return arr[i] * (1 - t) + arr[i + 1] * t;
  }

  // Shared y-window: centred on the closed form 1/λ, wide enough to show the
  // early Monte-Carlo wobble of both estimators but clamped so it stays legible.
  function yWindow(): { lo: number; hi: number } {
    const closed = 1 / lam;
    let span = 0;
    // look at the noisy early third of the grid to size the window
    const upto = Math.max(4, Math.round(G * 0.4));
    for (let i = 0; i < upto; i++) {
      span = Math.max(span, Math.abs(estScore[i] - closed), Math.abs(estHess[i] - closed));
    }
    span = clamp(span * 1.15, closed * 0.18, closed * 0.9);
    return { lo: closed - span, hi: closed + span };
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    draw();
    const closed = 1 / lam;
    eqScore.set(estAt(estScore, logN).toFixed(4));
    eqHess.set(estAt(estHess, logN).toFixed(4));
    eqN.set(formatN(Math.round(Math.pow(10, logN))));
    eqClosed.innerHTML = staticEquation(
      `=\\tfrac{1}{\\lambda}=\\tfrac{1}{${lam}}=${closed.toFixed(4)}`,
      theme,
      theme.gold,
    ).innerHTML;
    if (document.activeElement !== nSlider) nSlider.value = String(nearestGridIndexLog(logN));
    if (document.activeElement !== lamRow.input) lamRow.set(lam);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 48;
    const mr = 16;
    const mt = 18;
    const mb = 32;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const { lo, hi } = yWindow();
    const x = (lg: number) => ml + ((clamp(lg, logLo, logHi) - logLo) / (logHi - logLo)) * plotW;
    const y = (v: number) => mt + plotH - ((clamp(v, lo, hi) - lo) / (hi - lo)) * plotH;
    const closed = 1 / lam;

    // faint number-plane: vertical decade lines (the receding cyan grid)
    for (let d = Math.ceil(logLo); d <= Math.floor(logHi); d++) {
      strokeLine(ctx, [x(d), mt], [x(d), mt + plotH], theme.grid, 1, 0.22);
    }
    // horizontal grid at a few round info levels
    for (const gv of niceLevels(lo, hi)) {
      strokeLine(ctx, [ml, y(gv)], [W - mr, y(gv)], theme.grid, 1, 0.14);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // the closed-form answer line 1/λ (gold, dashed, faintly glowing)
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[ml, y(closed)], [W - mr, y(closed)]], {
        color: theme.gold,
        width: 1.6,
        alpha: 0.85,
        dash: [7, 6],
      });
    });

    // both running-mean curves over the log-N grid
    const scorePts: Pt[] = [];
    const hessPts: Pt[] = [];
    for (let i = 0; i < G; i++) {
      scorePts.push([x(logNs[i]), y(estScore[i])]);
      hessPts.push([x(logNs[i]), y(estHess[i])]);
    }
    // negative-Hessian form: gold, dotted (matches the notebook's ":" style)
    polyline(ctx, hessPts, { color: theme.gold, width: 1.8, alpha: 0.95, dash: [2, 4] });
    // score-variance form: bright blue, solid, glowing — the data estimator
    glow(ctx, theme.blue, 4, () => polyline(ctx, scorePts, { color: theme.blue, width: 2.2, alpha: 1 }));

    // the cursor (the movable variable): vertical rule reading off both curves
    const cxp = x(logN);
    const sNow = estAt(estScore, logN);
    const hNow = estAt(estHess, logN);
    strokeLine(ctx, [cxp, mt], [cxp, mt + plotH], theme.cream, 1.2, 0.55);
    glow(ctx, theme.blue, 7, () => disc(ctx, [cxp, y(sNow)], 3.8, theme.blue, 1));
    glow(ctx, theme.gold, 7, () => disc(ctx, [cxp, y(hNow)], 3.4, theme.gold, 1));
    mathLabel(ctx, "N", [cxp + (logN < logHi - 0.4 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 15,
      align: logN < logHi - 0.4 ? "left" : "right",
      glow: 5,
    });

    // x ticks: powers of ten (10^1 … 10^6) in roman numerals
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let d = Math.ceil(logLo); d <= Math.floor(logHi); d++) {
      mathLabel(ctx, "10", [x(d) - 6, mt + plotH + 14], { color: theme.tick, size: 12, sub: String(d), italic: false });
    }
    // y ticks
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const gv of niceLevels(lo, hi)) ctx.fillText(gv.toFixed(2), ml - 6, y(gv));

    // axis titles (KaTeX serif)
    mathLabel(ctx, "I(\\lambda)", [ml - 34, mt + 10], { color: theme.tick, size: 13, italic: true });

    // legend
    drawLegend(ctx, W - mr, mt);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["1/λ  closed form", theme.gold, [7, 6]],
      ["E[s²]  score var", theme.blue, []],
      ["−E[∂²]  neg-Hess", theme.gold, [2, 4]],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = top + 52;
    for (const [name, color, dash] of rows) {
      const lx = right - 132;
      ctx.strokeStyle = color;
      ctx.lineWidth = dash.length === 0 ? 2.2 : 1.8;
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
  function setN(n: number, animate = false): void {
    const lg = clamp(Math.log10(Math.max(1, n)), logLo, logHi);
    if (animate) logNTracker.set(lg, true);
    else logNTracker.jump(lg);
  }

  function setLambda(l: number): void {
    lam = clamp(Math.round(l), 1, 12);
    drawSample();
    recompute();
    render();
  }

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    drawSample();
    recompute();
    render();
  }

  reseedBtn.addEventListener("click", reseed);
  // The N slider indexes the log grid directly; dragging it jumps the cursor.
  nSlider.addEventListener("input", () => {
    const i = clamp(Math.round(Number(nSlider.value)), 0, G - 1);
    logNTracker.jump(logNs[i]);
  });
  lamRow.input.addEventListener("input", () => setLambda(Number(lamRow.input.value)));

  // Horizontal drag on the canvas scrubs N across the log axis.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 48;
      const mr = 16;
      const plotW = rc.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      logNTracker.jump(logLo + f * (logHi - logLo));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Two estimators of the Fisher information for a Poisson model: the score-variance form and the negative-Hessian form, both as running Monte-Carlo means over a log sample-size axis, converging to the closed-form value one over lambda. A cursor reads both running means off at a chosen sample size; a slider sets lambda and a button redraws the random sample.",
  );

  const onKey = (e: KeyboardEvent) => {
    const i = nearestGridIndexLog(logN);
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      logNTracker.set(logNs[clamp(i - 1, 0, G - 1)], true);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      logNTracker.set(logNs[clamp(i + 1, 0, G - 1)], true);
    } else if (e.key === "Home") {
      logNTracker.set(logLo, true);
    } else if (e.key === "End") {
      logNTracker.set(logHi, true);
    } else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land at the converged (largest-N) end with no glide.
  if (prefersReducedMotion()) logN = logHi;

  // --- grid index helpers --------------------------------------------------
  function nearestGridIndex(n: number): number {
    return nearestGridIndexLog(Math.log10(Math.max(1, n)));
  }
  function nearestGridIndexLog(lg: number): number {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < G; i++) {
      const d = Math.abs(logNs[i] - lg);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

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
      logNTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
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
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: "1",
    value: String(value),
    style: "flex:1;min-width:80px;max-width:130px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:2ch;text-align:right;` });
  out.textContent = String(value);
  input.addEventListener("input", () => (out.textContent = input.value));
  row.append(lab, input, out);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
      out.textContent = String(v);
    },
  };
}

// --- math helpers ----------------------------------------------------------

/**
 * The notebook's sample-size grid: `np.unique(np.round(np.logspace(a, b, n)))`.
 * Integer, strictly increasing, de-duplicated after rounding.
 */
function logspaceUniqueInt(a: number, b: number, n: number): number[] {
  const out: number[] = [];
  let prev = -1;
  for (let i = 0; i < n; i++) {
    const t = a + ((b - a) * i) / (n - 1);
    const v = Math.round(Math.pow(10, t));
    if (v !== prev) {
      out.push(v);
      prev = v;
    }
  }
  return out;
}

/**
 * A seeded Poisson(λ) variate via Knuth's multiplicative algorithm — exact for
 * the small λ ∈ [1, 12] this scene uses, and reproducible from the engine's Rng.
 */
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

/** Round-number horizontal levels inside [lo, hi], ~4–6 of them. */
function niceLevels(lo: number, hi: number): number[] {
  const span = hi - lo;
  const raw = span / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  const first = Math.ceil(lo / step) * step;
  for (let v = first; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** Compact sample-size formatting: 12 345 → "12.3k", 1 000 000 → "1.0M". */
function formatN(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
