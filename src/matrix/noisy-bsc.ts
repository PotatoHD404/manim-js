import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Mat, type Vec } from "../core/math/linalg";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface NoisyBscOptions {
  /** Rows of the binary code (the number of correlated entries). Default 6. */
  k?: number;
  /** Rank of the generator matrix G (k × r). Default 2. */
  r?: number;
  /** Seed for the random full-rank generator G. */
  seed?: number;
  /** Initial BSC crossover probability δ ∈ [0, ½]. */
  delta?: number;
  theme?: Partial<Theme>;
}

export interface NoisyBscApi {
  /** Move the crossover probability δ ∈ [0, ½] (eased unless animate=false). */
  setDelta(delta: number, animate?: boolean): void;
  /** Snap to the noiseless channel δ = 0, where C = k/r. */
  toNoiseless(): void;
  /** Snap to the useless channel δ = ½, where C = 0. */
  toUseless(): void;
  /** Draw a fresh full-rank generator G from the next seed. */
  reseed(): void;
  destroy(): void;
}

const DMAX = 0.5; // δ ranges over [0, ½]; beyond ½ the channel is just relabeled

/**
 * Scene — "noise scales capacity by 1 − h(δ)". A rank-r binary code is observed
 * through a binary symmetric channel with crossover probability δ. The completion
 * capacity has the closed form C(δ) = k(1 − h(δ))/r, where h is the binary
 * entropy. The gold curve is that closed form; the blue dots are the capacity
 * recomputed the long way, as a sum of per-row mutual informations
 * I(Xℓ;Yℓ) = H(Yℓ) − H(Yℓ|Xℓ) over the seeded joint law of the code, exactly as
 * the notebook does — they coincide. The red curve traces the channel's own
 * binary entropy h(δ) on the same axes: it climbs from 0 to 1 bit as δ → ½,
 * where the noiseless capacity C₀ = k/r is scaled all the way to zero.
 *
 * Drag / slide / arrow-key the crossover δ; a reseed control redraws the random
 * generator G (the Monte-Carlo part of the notebook cell).
 *
 * Ported verbatim from the post's notebook BSC cell (hbin, bsc, capacity_noisy,
 * linear_code; k = 6, r = 2).
 */
export function createMatrixNoisyBsc(target: HTMLElement, options: NoisyBscOptions = {}): NoisyBscApi {
  const theme = withTheme(options.theme);
  const k = clamp(Math.round(options.k ?? 6), 2, 12);
  const r = clamp(Math.round(options.r ?? 2), 1, k);
  let seed = (options.seed ?? 0) >>> 0;

  // --- the seeded binary code (the Monte-Carlo part of the notebook cell) ---
  // A random full-rank k×r generator G with no zero row; its column space over
  // GF(2) is a rank-r code whose 2^r codewords are equiprobable.
  let G: Mat = [];
  let joint: number[] = []; // flattened pmf over {0,1}^k, length 2^k
  let C0 = k / r; // noiseless capacity C(0) = k/r

  function buildCode(): void {
    const rng = new Rng(seed || 1);
    G = drawGenerator(rng, k, r);
    joint = linearCodeJoint(G, k, r);
    C0 = k / r;
  }
  buildCode();

  // The full mutual-information capacity, recomputed exactly as `capacity_noisy`
  // in the notebook: Σℓ [H(Yℓ) − H(Yℓ|Xℓ)] / H(X₁,…,X_k), with the BSC channel
  // W(δ) on every row. For a code this equals the closed form k(1−h(δ))/r.
  function capacityComputed(delta: number): number {
    const Hj = entropyJoint(joint, k);
    if (Hj < 1e-12) return Number.POSITIVE_INFINITY;
    let iSum = 0;
    const hDgivenX = hbin(delta); // H(Yℓ|Xℓ) = h(δ) for either input symbol
    for (let l = 0; l < k; l++) {
      const px = marginal(joint, k, l); // [p0, p1] on row ℓ
      // py = pxᵀ W; W = [[1-δ, δ], [δ, 1-δ]]
      const py0 = px[0] * (1 - delta) + px[1] * delta;
      const py1 = px[0] * delta + px[1] * (1 - delta);
      const hY = entropy2(py0, py1);
      iSum += hY - hDgivenX;
    }
    return iSum / Hj;
  }

  // The closed form from the post: C(δ) = k(1 − h(δ))/r.
  const capacityFormula = (delta: number): number => (k * (1 - hbin(delta))) / r;

  // --- sampled curves (δ ∈ [0, ½], matching np.linspace(0, .5, 120)) --------
  const N = 120;
  const deltas = new Float64Array(N);
  const cComputed = new Float64Array(N);
  const cFormula = new Float64Array(N);
  const hCurve = new Float64Array(N);
  function resampleCurves(): void {
    for (let i = 0; i < N; i++) {
      const d = (i / (N - 1)) * DMAX;
      deltas[i] = d;
      cFormula[i] = capacityFormula(d);
      cComputed[i] = capacityComputed(d);
      hCurve[i] = hbin(d);
    }
  }
  resampleCurves();
  const cMax = C0 * 1.06; // top of the capacity axis (C(0) = k/r)

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the live capacity and the channel factor
  const overlay = overlayLayer("tl");
  const eqC = liveEquation("C(\\delta)=\\dfrac{k\\,(1-h(\\delta))}{r}\\;=", theme, theme.gold);
  const eqFac = liveEquation("1-h(\\delta)\\;=", theme, theme.green);
  const eqDelta = liveEquation("\\delta\\;=", theme, theme.cream);
  overlay.append(eqC.node, eqFac.node, eqDelta.node);
  mainHost.append(overlay);

  // top-right: the code constants, restated for the current k, r
  const codeOverlay = overlayLayer("tr");
  const eqCode = staticEquation("", theme, theme.blue);
  codeOverlay.append(eqCode);
  mainHost.append(codeOverlay);

  const rc = responsiveCanvas(mainHost, 1.5, () => render());

  // --- controls -------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const noiselessBtn = el("button", { type: "button", style: btnStyle(theme) });
  noiselessBtn.innerHTML = "&#9650;&nbsp; δ = 0";
  const uselessBtn = el("button", { type: "button", style: btnStyle(theme) });
  uselessBtn.innerHTML = "&#9660;&nbsp; δ = ½";
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed G";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "crossover  δ";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "0.5",
    step: "0.002",
    value: String(clamp(options.delta ?? 0.1, 0, DMAX)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "binary-symmetric-channel crossover probability delta, from zero to one half",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · reseed · ←→";
  controls.append(noiselessBtn, uselessBtn, reseedBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the crossover probability δ --------------------
  let delta = clamp(options.delta ?? 0.1, 0, DMAX);
  let destroyed = false;
  const deltaTracker = valueTracker(delta, (v) => {
    delta = clamp(v, 0, DMAX);
    render();
  });

  // --- render ---------------------------------------------------------------
  function render(): void {
    draw();

    const C = capacityFormula(delta);
    const fac = 1 - hbin(delta);
    eqC.set(C.toFixed(3));
    eqFac.set(fac.toFixed(3));
    eqDelta.set(delta.toFixed(3));
    eqCode.innerHTML = staticEquation(
      `\\text{rank-}${r}\\ \\text{code},\\ k=${k}:\\quad C_0=\\tfrac{k}{r}=${fmt(C0)}`,
      theme,
      theme.blue,
    ).innerHTML;

    if (document.activeElement !== slider) slider.value = String(delta);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 18;
    const mt = 18;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // Two stacked quantities share δ on x: capacity C on the left axis (gold),
    // the binary entropy h on the right axis [0,1] (red). x maps δ ∈ [0, ½].
    const X = (d: number) => ml + (clamp(d, 0, DMAX) / DMAX) * plotW;
    const Yc = (c: number) => mt + plotH - (clamp(c, 0, cMax) / cMax) * plotH;
    const Yh = (h: number) => mt + plotH - clamp(h, 0, 1) * plotH;

    // faint number plane (the cyan grid, receding) on δ and on the h-axis
    for (let g = 0; g <= DMAX + 1e-9; g += 0.1) {
      strokeLine(ctx, [X(g), mt], [X(g), mt + plotH], theme.grid, 1, g === 0 ? 0.28 : 0.16);
    }
    for (let g = 0; g <= 1.0001; g += 0.25) {
      strokeLine(ctx, [ml, Yh(g)], [W - mr, Yh(g)], theme.grid, 1, g === 0 ? 0.28 : 0.14);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.5, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.5, 1);
    strokeLine(ctx, [W - mr, mt], [W - mr, mt + plotH], theme.axis, 1.2, 0.6);

    const screen = (yMap: (v: number) => number, arr: Float64Array): Pt[] => {
      const out: Pt[] = new Array(N);
      for (let i = 0; i < N; i++) out[i] = [X(deltas[i]), yMap(arr[i])];
      return out;
    };

    // the channel's binary entropy h(δ): the red "noise" climbing 0 → 1 bit
    const hPts = screen(Yh, hCurve);
    polyline(ctx, hPts, { color: theme.red, width: 1.8, alpha: 0.85, dash: [6, 5] });

    // the closed-form capacity C(δ) = k(1−h(δ))/r: the gold answer, filled under
    const cFormPts = screen(Yc, cFormula);
    const fillPts: Pt[] = [[X(0), mt + plotH], ...cFormPts, [X(DMAX), mt + plotH]];
    polyline(ctx, fillPts, { color: "transparent", width: 0, closed: true, fill: hexA(theme.gold, 0.08) });
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, cFormPts, { color: theme.gold, width: 2.6, alpha: 1 });
    });

    // the long-way capacity (per-row mutual informations) — blue dots that land
    // right on the gold curve, the notebook's "computed == formula" check
    glow(ctx, theme.blue, 4, () => {
      for (let i = 0; i < N; i += 4) {
        disc(ctx, [X(deltas[i]), Yc(cComputed[i])], 2.4, theme.blue, 0.9);
      }
    });

    // C₀ = k/r reference line at δ = 0 and the collapse to 0 at δ = ½
    strokeLine(ctx, [ml, Yc(C0)], [W - mr, Yc(C0)], theme.gold, 1, 0.32);
    mathLabel(ctx, "C", [ml + 6, Yc(C0) - 6], { color: theme.gold, size: 13, sub: "0", glow: 4 });

    // the movable point: δ on the gold capacity curve and on the red entropy
    const cx = X(delta);
    const cNow = capacityFormula(delta);
    const hNow = hbin(delta);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.55);
    // tick on the capacity curve (the answer the user is reading)
    glow(ctx, theme.gold, 7, () => disc(ctx, [cx, Yc(cNow)], 4, theme.gold, 1));
    // tick on the entropy curve
    disc(ctx, [cx, Yh(hNow)], 3, theme.red, 0.95);
    mathLabel(ctx, "δ", [cx + (delta < DMAX * 0.85 ? 7 : -7), mt + 14], {
      color: theme.cream,
      size: 15,
      align: delta < DMAX * 0.85 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "δ", [W - mr - 4, mt + plotH + 24], { color: theme.tick, size: 16, align: "right" });
    mathLabel(ctx, "C", [ml - 30, mt + 12], { color: theme.gold, size: 15 });
    mathLabel(ctx, "h", [W - mr + 6, mt + 12], { color: theme.red, size: 15, align: "left" });

    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = 0; g <= DMAX + 1e-9; g += 0.1) ctx.fillText(g.toFixed(1), X(g), mt + plotH + 6);
    // left capacity ticks
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const cStep = niceStep(cMax);
    for (let g = 0; g <= cMax + 1e-9; g += cStep) ctx.fillText(fmt(g), ml - 6, Yc(g));
    // right entropy ticks (the [0,1]-bit scale)
    ctx.textAlign = "left";
    for (const g of [0, 0.5, 1]) ctx.fillText(g.toFixed(1), W - mr + 6, Yh(g));

    drawLegend(ctx, ml + 8, mt + 6);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, left: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["C(δ) = k(1−h)/r", theme.gold, []],
      ["computed (ΣI)", theme.blue, []],
      ["h(δ)", theme.red, [6, 5]],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = top + 70;
    for (const [name, color, dash] of rows) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(left, yy);
      ctx.lineTo(left + lineW, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      if (name.startsWith("computed")) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(left + lineW / 2, yy, 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = theme.fg;
      ctx.fillText(name, left + lineW + 6, yy + 0.5);
      yy += 16;
    }
    ctx.restore();
  }

  // --- interaction ----------------------------------------------------------
  function setDelta(d: number, animate = false): void {
    const t = clamp(d, 0, DMAX);
    if (animate) deltaTracker.set(t, true);
    else deltaTracker.jump(t);
  }

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    buildCode();
    resampleCurves();
    render();
  }

  noiselessBtn.addEventListener("click", () => setDelta(0, true));
  uselessBtn.addEventListener("click", () => setDelta(DMAX, true));
  reseedBtn.addEventListener("click", reseed);
  slider.addEventListener("input", () => deltaTracker.jump(clamp(Number(slider.value), 0, DMAX)));

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 46;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      deltaTracker.jump(clamp((px - ml) / plotW, 0, 1) * DMAX);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Binary symmetric channel: the completion capacity of a rank-r binary code, C equals k times one minus the binary entropy of the crossover probability, all over r. As the crossover delta rises from zero to one half the binary-entropy curve climbs from zero to one bit and the capacity falls from k over r to zero. Drag to move the crossover.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setDelta(0, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setDelta(DMAX, true);
      return;
    }
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setDelta(delta - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setDelta(delta + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on a representative mid-channel state, no glide.
  if (prefersReducedMotion()) delta = clamp(options.delta ?? 0.1, 0, DMAX);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setDelta,
    toNoiseless: () => setDelta(0, true),
    toUseless: () => setDelta(DMAX, true),
    reseed,
    destroy() {
      destroyed = true;
      deltaTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- math ports (verbatim from the notebook BSC cell) -----------------------

/** Binary entropy h(δ) in bits; h(0) = h(1) = 0. (notebook `hbin`) */
function hbin(d: number): number {
  if (d <= 0 || d >= 1) return 0;
  const LOG2 = Math.log(2);
  return (-d * Math.log(d) - (1 - d) * Math.log(1 - d)) / LOG2;
}

/** Shannon entropy of a 2-symbol distribution (p0, p1), in bits. */
function entropy2(p0: number, p1: number): number {
  const LOG2 = Math.log(2);
  let s = 0;
  if (p0 > 0) s -= p0 * Math.log(p0);
  if (p1 > 0) s -= p1 * Math.log(p1);
  return s / LOG2;
}

/** Shannon entropy of a flat pmf over {0,1}^k (ignoring zeros), in bits. */
function entropyJoint(p: number[], _k: number): number {
  const LOG2 = Math.log(2);
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const v = p[i];
    if (v > 0) s -= v * Math.log(v);
  }
  return s / LOG2;
}

/** Marginal pmf [p0, p1] of row ℓ from a flattened {0,1}^k joint. */
function marginal(p: number[], k: number, l: number): [number, number] {
  let p0 = 0;
  let p1 = 0;
  const bit = k - 1 - l; // row ℓ is bit (k-1-ℓ) in the big-endian index
  for (let idx = 0; idx < p.length; idx++) {
    if ((idx >> bit) & 1) p1 += p[idx];
    else p0 += p[idx];
  }
  return [p0, p1];
}

/**
 * The joint pmf of a rank-r binary linear code as a flat array over {0,1}^k.
 * Each information word u ∈ {0,1}^r maps to codeword x = (G u) mod 2; the 2^r
 * codewords are equiprobable. (notebook `linear_code`)
 */
function linearCodeJoint(G: Mat, k: number, r: number): number[] {
  const size = 1 << k;
  const j = new Array<number>(size).fill(0);
  const words = 1 << r;
  for (let u = 0; u < words; u++) {
    let idx = 0;
    for (let row = 0; row < k; row++) {
      let bit = 0;
      for (let c = 0; c < r; c++) {
        if ((u >> c) & 1) bit ^= G[row][c] & 1;
      }
      idx = (idx << 1) | bit; // big-endian: row 0 is the most significant bit
    }
    j[idx] += 1;
  }
  const total = j.reduce((a, b) => a + b, 0) || 1;
  for (let i = 0; i < size; i++) j[i] /= total;
  return j;
}

/**
 * A random full-rank k×r binary generator with no all-zero row — the seeded
 * loop from the notebook (`rng.integers(0,2,(k,r))` rejected until rank r and no
 * zero row). Rank over GF(2) is checked by Gaussian elimination.
 */
function drawGenerator(rng: Rng, k: number, r: number): Mat {
  for (let attempt = 0; attempt < 2000; attempt++) {
    const M: Mat = [];
    for (let i = 0; i < k; i++) {
      const row: Vec = [];
      for (let j = 0; j < r; j++) row.push(rng.next() < 0.5 ? 0 : 1);
      M.push(row);
    }
    const noZeroRow = M.every((row) => row.some((b) => b !== 0));
    if (noZeroRow && gf2Rank(M, k, r) === r) return M;
  }
  // Deterministic fallback: identity in the top r rows, ones elsewhere.
  const M: Mat = [];
  for (let i = 0; i < k; i++) {
    const row: Vec = new Array(r).fill(0);
    row[i % r] = 1;
    M.push(row);
  }
  return M;
}

/** Rank of a k×r binary matrix over GF(2) via Gaussian elimination on columns. */
function gf2Rank(M: Mat, k: number, r: number): number {
  // Work on a copy of the rows as bitmask numbers (r ≤ 12 fits easily).
  const rows: number[] = [];
  for (let i = 0; i < k; i++) {
    let m = 0;
    for (let j = 0; j < r; j++) if (M[i][j] & 1) m |= 1 << j;
    rows.push(m);
  }
  let rank = 0;
  for (let col = 0; col < r; col++) {
    let pivot = -1;
    for (let i = rank; i < k; i++) {
      if ((rows[i] >> col) & 1) {
        pivot = i;
        break;
      }
    }
    if (pivot === -1) continue;
    [rows[rank], rows[pivot]] = [rows[pivot], rows[rank]];
    for (let i = 0; i < k; i++) {
      if (i !== rank && (rows[i] >> col) & 1) rows[i] ^= rows[rank];
    }
    rank++;
  }
  return rank;
}

// --- small helpers ----------------------------------------------------------

/** Format a capacity number compactly (integers without a trailing .00). */
function fmt(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(2);
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

/** Append an alpha byte to a #rrggbb color. */
function hexA(hex: string, alpha: number): string {
  const a = Math.round(clamp(alpha, 0, 1) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}
