import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface RandomEnsemblesOptions {
  /** Number of binary rows k; the joint lives on {0,1}^k. Default 4. */
  k?: number;
  /** Dirichlet concentration α for the random joint prior. Default 1.0 (uniform). */
  alpha?: number;
  /** Monte-Carlo sample size (joints drawn per ensemble). Default 4000. */
  samples?: number;
  /** Seed for the random ensemble. */
  seed?: number;
  theme?: Partial<Theme>;
}

export interface RandomEnsemblesApi {
  /** Set the Dirichlet concentration α (eased unless animate=false). */
  setAlpha(alpha: number, animate?: boolean): void;
  /** Set the number of binary rows k (resamples the ensemble). */
  setK(k: number): void;
  /** Move the read-off cursor over the capacity axis (eased). */
  setCursor(c: number, animate?: boolean): void;
  /** Draw a fresh Monte-Carlo ensemble from the next seed. */
  reseed(): void;
  destroy(): void;
}

const KMIN = 3;
const KMAX = 5;
const AMIN = 0.05; // sparsest Dirichlet
const AMAX = 2.0; // smoothest Dirichlet shown
const NBINS = 60;
const CMAX_VIEW = 3.8; // top of the capacity axis (matches the post's panel-b range)

/**
 * Scene — "how compressible is a generic joint?". For a random discrete joint law
 * over {0,1}^k drawn from a symmetric Dirichlet prior, the i.i.d. completion
 * capacity is C = S / H_joint, where S = Σℓ H(Xℓ) sums the marginal entropies and
 * H_joint is the joint entropy. A whole Monte-Carlo ensemble of such joints is
 * histogrammed (green bars, the traced quantity); the gold dashed rule at C = 1 is
 * the independence baseline — a generic joint sits barely above it. The cream
 * Dirichlet-concentration slider α is the knob: at α = 1 (uniform) the law
 * concentrates just above C = 1, because a random joint has only weak dependence;
 * sharpen the concentration (small α) and mass piles onto fewer patterns,
 * manufacturing redundancy and a heavy right tail of high-capacity matrices.
 *
 * Ported verbatim from the post's notebook capacity functionals (H, marginal,
 * capacity = S / H_joint) with the Dirichlet ensemble of §7; a reseed control
 * redraws the Monte-Carlo sample. The drag/scrub cursor reads the histogram
 * density and the right-tail mass P(C > c) at a chosen capacity c.
 */
export function createMatrixRandomEnsembles(
  target: HTMLElement,
  options: RandomEnsemblesOptions = {},
): RandomEnsemblesApi {
  const theme = withTheme(options.theme);
  let k = clamp(Math.round(options.k ?? 4), KMIN, KMAX);
  let alpha = clamp(options.alpha ?? 1.0, AMIN, AMAX);
  const M = clamp(Math.round(options.samples ?? 4000), 200, 20000);
  let seed = (options.seed ?? 12345) >>> 0 || 1;

  // --- the Monte-Carlo ensemble (the §7 random-ensembles cell) --------------
  // For each of M trials, draw a joint pmf over {0,1}^k from a symmetric
  // Dirichlet(α·1) and record its completion capacity C = S / H_joint. The
  // histogram of those capacities is the figure.
  let caps: number[] = []; // capacities of the current ensemble
  let hist = new Float64Array(NBINS); // density per bin

  function rebuild(): void {
    const rng = new Rng(seed);
    const N = 1 << k; // 2^k joint cells
    caps = new Array<number>(M);
    for (let t = 0; t < M; t++) {
      const joint = dirichlet(rng, alpha, N);
      caps[t] = capacity(joint, k);
    }
    rebuildHistogram();
  }

  function rebuildHistogram(): void {
    hist = new Float64Array(NBINS);
    const binW = CMAX_VIEW / NBINS;
    let inRange = 0;
    for (const c of caps) {
      if (!Number.isFinite(c)) continue;
      const b = Math.floor(clamp(c, 0, CMAX_VIEW - 1e-9) / binW);
      hist[b] += 1;
      inRange++;
    }
    // Normalize to a density (area = fraction of mass in range), like the
    // notebook's `density=True` histograms.
    const denom = (inRange || 1) * binW;
    for (let b = 0; b < NBINS; b++) hist[b] /= denom;
  }

  rebuild();

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the live ensemble statistics
  const overlay = overlayLayer("tl");
  const eqAlpha = liveEquation("\\alpha\\;=", theme, theme.cream);
  const eqMean = liveEquation("\\mathbb{E}[C]\\;=", theme, theme.green);
  const eqTail = liveEquation("\\Pr[C>c]\\;=", theme, theme.green);
  overlay.append(eqAlpha.node, eqMean.node, eqTail.node);
  mainHost.append(overlay);

  // top-right: the capacity definition, restated for the current k
  const defOverlay = overlayLayer("tr");
  const eqDef = staticEquation("", theme, theme.blue);
  defOverlay.append(eqDef);
  mainHost.append(defOverlay);

  const rc = responsiveCanvas(mainHost, 1.62, () => render());

  // --- controls -------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";

  const alphaLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  alphaLabel.textContent = "concentration  α";
  const alphaSlider = el("input", {
    type: "range",
    min: String(AMIN),
    max: String(AMAX),
    step: "0.01",
    value: String(alpha),
    style: "flex:1;min-width:130px;max-width:220px;",
    "aria-label": "Dirichlet concentration alpha; smaller alpha makes sparser, more redundant joints",
  });

  const kLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  kLabel.textContent = "rows  k";
  const kSlider = el("input", {
    type: "range",
    min: String(KMIN),
    max: String(KMAX),
    step: "1",
    value: String(k),
    style: "min-width:80px;max-width:120px;",
    "aria-label": "number of binary rows k",
  });
  const kOut = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:1.5ch;` });
  kOut.textContent = String(k);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag C · reseed · ←→";
  controls.append(reseedBtn, alphaLabel, alphaSlider, kLabel, kSlider, kOut, hint);
  root.append(controls);

  target.append(root);

  // --- the eased scalars: α drives the ensemble, the cursor reads it off -----
  let cursor = clamp(options.alpha !== undefined ? 1.0 : 1.2, 0, CMAX_VIEW);
  let destroyed = false;

  // α eases; whenever it settles into a new value the ensemble is rebuilt. To
  // stay real-time during a drag we resample on every change of the displayed
  // value (the histogram is cheap to recompute for these sizes).
  const alphaTracker = valueTracker(alpha, (v) => {
    alpha = clamp(v, AMIN, AMAX);
    rebuild();
    render();
  });

  const cursorTracker = valueTracker(cursor, (v) => {
    cursor = clamp(v, 0, CMAX_VIEW);
    render();
  });

  // --- render ---------------------------------------------------------------
  function render(): void {
    draw();

    eqAlpha.set(alpha.toFixed(2));
    eqMean.set(meanCapacity().toFixed(3));
    eqTail.set(tailFraction(cursor).toFixed(3));
    eqDef.innerHTML = staticEquation(
      `C=\\dfrac{S}{H_{\\text{joint}}},\\ S=\\sum_{\\ell=1}^{${k}}H(X_\\ell)`,
      theme,
      theme.blue,
    ).innerHTML;

    if (document.activeElement !== alphaSlider) alphaSlider.value = String(alpha);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 40;
    const mr = 18;
    const mt = 16;
    const mb = 32;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    const binW = CMAX_VIEW / NBINS;
    let ymax = 1e-6;
    for (let b = 0; b < NBINS; b++) if (hist[b] > ymax) ymax = hist[b];
    ymax *= 1.12;

    const X = (c: number) => ml + (clamp(c, 0, CMAX_VIEW) / CMAX_VIEW) * plotW;
    const Y = (d: number) => mt + plotH - (clamp(d, 0, ymax) / ymax) * plotH;

    // faint number plane (the cyan grid, receding): vertical rules at integer C,
    // horizontal rules across the density axis.
    for (let g = 0; g <= CMAX_VIEW + 1e-9; g += 1) {
      strokeLine(ctx, [X(g), mt], [X(g), mt + plotH], theme.grid, 1, g === 0 ? 0.28 : 0.16);
    }
    const dStep = niceStep(ymax);
    for (let g = dStep; g <= ymax + 1e-9; g += dStep) {
      strokeLine(ctx, [ml, Y(g)], [W - mr, Y(g)], theme.grid, 1, 0.14);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.5, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.5, 1);

    // the right tail (C > cursor) — shade those bars brighter, the rest dim. This
    // is the redundancy the figure is about: mass pushed above C = 1.
    const baseY = mt + plotH;
    for (let b = 0; b < NBINS; b++) {
      const c0 = b * binW;
      const c1 = (b + 1) * binW;
      const x0 = X(c0);
      const x1 = X(c1);
      const yTop = Y(hist[b]);
      const inTail = c0 >= cursor - 1e-9;
      ctx.globalAlpha = inTail ? 0.85 : 0.4;
      ctx.fillStyle = theme.green;
      ctx.fillRect(x0, yTop, Math.max(1, x1 - x0 - 0.6), baseY - yTop);
      ctx.globalAlpha = 1;
    }
    // a crisp outline over the histogram silhouette (chalk on the board)
    const outline: Pt[] = [[X(0), baseY]];
    for (let b = 0; b < NBINS; b++) {
      const x0 = X(b * binW);
      const x1 = X((b + 1) * binW);
      const y = Y(hist[b]);
      outline.push([x0, y], [x1, y]);
    }
    outline.push([X(CMAX_VIEW), baseY]);
    glow(ctx, theme.green, 4, () => {
      polyline(ctx, outline, { color: theme.green, width: 1.6, alpha: 0.9 });
    });

    // the independence baseline C = 1 — the gold "answer": a generic joint barely
    // clears it.
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[X(1), mt], [X(1), mt + plotH]], {
        color: theme.gold,
        width: 1.8,
        alpha: 0.9,
        dash: [6, 6],
      });
    });
    mathLabel(ctx, "C", [X(1) + 6, mt + 12], { color: theme.gold, size: 14, sub: "ind", glow: 4 });

    // the mean capacity marker (where the ensemble sits on average)
    const mc = meanCapacity();
    strokeLine(ctx, [X(mc), mt + 2], [X(mc), mt + plotH], theme.blue, 1.2, 0.55);
    mathLabel(ctx, "E[C]", [X(mc) + (mc < CMAX_VIEW * 0.85 ? 5 : -5), mt + 26], {
      color: theme.blue,
      size: 12,
      italic: false,
      align: mc < CMAX_VIEW * 0.85 ? "left" : "right",
    });

    // the movable read-off cursor over the capacity axis (the cream variable)
    const cx = X(cursor);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.3, 0.6);
    glow(ctx, theme.cream, 7, () => disc(ctx, [cx, Y(densityAt(cursor))], 4, theme.cream, 1));
    mathLabel(ctx, "c", [cx + (cursor < CMAX_VIEW * 0.9 ? 7 : -7), mt + 14], {
      color: theme.cream,
      size: 15,
      align: cursor < CMAX_VIEW * 0.9 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "C", [W - mr - 4, mt + plotH + 22], { color: theme.tick, size: 16, align: "right" });
    mathLabel(ctx, "density", [ml - 30, mt + 8], { color: theme.tick, size: 12, italic: false });

    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = 0; g <= CMAX_VIEW + 1e-9; g += 0.5) ctx.fillText(g.toFixed(1), X(g), mt + plotH + 6);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let g = dStep; g <= ymax + 1e-9; g += dStep) ctx.fillText(fmt(g), ml - 6, Y(g));

    drawLegend(ctx, W - mr - 8, mt + 56);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["density of C", theme.green, []],
      ["C = 1 (independent)", theme.gold, [6, 6]],
      ["mean E[C]", theme.blue, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = top;
    for (const [name, color, dash] of rows) {
      const lx = right - 130;
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
      yy += 16;
    }
    ctx.restore();
  }

  // --- ensemble statistics --------------------------------------------------
  function meanCapacity(): number {
    let s = 0;
    let n = 0;
    for (const c of caps) {
      if (Number.isFinite(c)) {
        s += c;
        n++;
      }
    }
    return n > 0 ? s / n : 0;
  }

  /** Fraction of the ensemble with capacity strictly above `c` (the right tail). */
  function tailFraction(c: number): number {
    let n = 0;
    let above = 0;
    for (const v of caps) {
      if (!Number.isFinite(v)) continue;
      n++;
      if (v > c) above++;
    }
    return n > 0 ? above / n : 0;
  }

  /** Histogram density at a continuous capacity value `c`. */
  function densityAt(c: number): number {
    const binW = CMAX_VIEW / NBINS;
    const b = clamp(Math.floor(c / binW), 0, NBINS - 1);
    return hist[b];
  }

  // --- interaction ----------------------------------------------------------
  function setAlpha(a: number, animate = false): void {
    const t = clamp(a, AMIN, AMAX);
    if (animate) alphaTracker.set(t, true);
    else alphaTracker.jump(t);
  }

  function setK(nk: number): void {
    k = clamp(Math.round(nk), KMIN, KMAX);
    kSlider.value = String(k);
    kOut.textContent = String(k);
    rebuild();
    render();
  }

  function setCursor(c: number, animate = false): void {
    const t = clamp(c, 0, CMAX_VIEW);
    if (animate) cursorTracker.set(t, true);
    else cursorTracker.jump(t);
  }

  function reseed(): void {
    seed = (seed + 0x9e3779b1) >>> 0 || 1;
    rebuild();
    render();
  }

  reseedBtn.addEventListener("click", reseed);
  alphaSlider.addEventListener("input", () => alphaTracker.jump(clamp(Number(alphaSlider.value), AMIN, AMAX)));
  kSlider.addEventListener("input", () => setK(Number(kSlider.value)));

  // Drag on the canvas scrubs the capacity cursor over the C axis.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 40;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      cursorTracker.jump(clamp((px - ml) / plotW, 0, 1) * CMAX_VIEW);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Random ensembles: a histogram of the completion capacity C equals the sum of marginal entropies over the joint entropy, for random discrete joints over k binary rows drawn from a symmetric Dirichlet prior. At concentration alpha equal to one the capacity concentrates just above one, the independence baseline; lowering alpha makes the joints sparser and more redundant, raising the mean capacity and fattening the right tail. Drag to read off the right-tail fraction at a chosen capacity.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setCursor(1, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setCursor(meanCapacity(), true);
      return;
    }
    const step = e.shiftKey ? 0.25 : 0.05;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setCursor(cursor - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setCursor(cursor + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: park the cursor on the ensemble mean, no glide.
  if (prefersReducedMotion()) cursor = clamp(meanCapacity(), 0, CMAX_VIEW);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setAlpha,
    setK,
    setCursor,
    reseed,
    destroy() {
      destroyed = true;
      alphaTracker.stop();
      cursorTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- math ports (verbatim from the notebook capacity functionals) -----------

const LOG2 = Math.log(2);

/** Shannon entropy of a flat pmf (ignoring zeros), in bits. (notebook `H`) */
function entropy(p: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const v = p[i];
    if (v > 0) s -= v * Math.log(v);
  }
  return s / LOG2;
}

/**
 * Completion capacity of a joint pmf over {0,1}^k as a flat array of length 2^k:
 * C = S / H_joint, with S = Σℓ H(Xℓ) the sum of the marginal entropies and
 * H_joint the joint entropy. Independence gives C = 1; redundancy raises it.
 * (notebook `capacity`, `marginal`, `H`)
 */
function capacity(joint: ArrayLike<number>, k: number): number {
  const N = joint.length;
  let S = 0;
  for (let l = 0; l < k; l++) {
    // marginal pmf [p0, p1] of row ℓ (bit ℓ of the flat index)
    let p0 = 0;
    let p1 = 0;
    for (let idx = 0; idx < N; idx++) {
      if ((idx >> l) & 1) p1 += joint[idx];
      else p0 += joint[idx];
    }
    S += entropy([p0, p1]);
  }
  const Hj = entropy(joint);
  return Hj < 1e-12 ? Number.POSITIVE_INFINITY : S / Hj;
}

/**
 * A draw from a symmetric Dirichlet(α·1) of length `n`, built from independent
 * Gamma(α) samples normalized to sum to one — the deterministic stand-in for the
 * notebook's `rng.dirichlet(α·ones(n))`.
 */
function dirichlet(rng: Rng, alpha: number, n: number): Float64Array {
  const g = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = gammaSample(rng, alpha);
    g[i] = v;
    sum += v;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < n; i++) g[i] *= inv;
  return g;
}

/**
 * A Gamma(shape, 1) sample via Marsaglia–Tsang, using the engine RNG's uniform
 * and normal generators. For shape < 1 it boosts with the standard
 * Gamma(a) = Gamma(a+1)·U^{1/a} identity. This reproduces `numpy`'s Dirichlet
 * draws in distribution (the figure is a Monte-Carlo histogram, so only the law
 * matters, not the exact stream).
 */
function gammaSample(rng: Rng, shape: number): number {
  if (shape < 1) {
    let u = rng.next();
    if (u <= 0) u = 1e-12;
    return gammaSample(rng, shape + 1) * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = rng.gauss();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    const x2 = x * x;
    if (u < 1 - 0.0331 * x2 * x2) return d * v;
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
  }
}

// --- small helpers ----------------------------------------------------------

/** Format a density tick compactly (integers without a trailing .00). */
function fmt(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(x < 1 ? 2 : 1);
}

/** A round-number grid step roughly dividing `max` into ~4 intervals. */
function niceStep(max: number): number {
  const raw = max / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
