import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface SymbolErrorOptions {
  /**
   * The (k, r) pairs to plot, all at the same code rate r/k. Defaults to the
   * post's four curves at rate 1/4: (4,1), (8,2), (16,4), (32,8).
   */
  ks?: Array<[number, number]>;
  /** Monte-Carlo erasure trials per (k, p) grid point. */
  trials?: number;
  /** Number of random generators G averaged per curve. */
  generators?: number;
  /** Initial operating point — the observation rate p ∈ (0,1). */
  p?: number;
  /** Number of points on the p grid. */
  grid?: number;
  seed?: number;
  theme?: Partial<Theme>;
}

export interface SymbolErrorApi {
  /** Move the operating point p ∈ (0,1) (eased unless animate=false). */
  setP(p: number, animate?: boolean): void;
  /** Snap the operating point to the converse threshold p★ = r/k. */
  snapToThreshold(): void;
  /** Redraw the Monte-Carlo curves from the next seed. */
  reseed(): void;
  destroy(): void;
}

const ROW_COLORS = ["#58C4DD", "#F0AC5F", "#83C167", "#FC6255"] as const;

/**
 * Scene — "the operational decoder". The rank-r binary-code model made literal:
 * a k×n matrix whose columns are random GF(2) codewords X = G·u is erased entry
 * by entry with probability 1−p, then every column is decoded by Gaussian
 * elimination over GF(2). A coordinate is determined iff its generator row lies
 * in the span of the observed rows; the rest are guessed and wrong half the
 * time, so the expected symbol-error rate is ½ · (undetermined / k), computed
 * exactly per erasure pattern with integer-XOR row reduction.
 *
 * Sweeping the observation rate p traces four decreasing error curves at fixed
 * code rate r/k = ¼ (k = 4, 8, 16, 32). They pivot at the converse threshold
 * p★ = 1/C = r/k = ¼: above it the larger-k curve decays faster, below it the
 * larger-k curve errs more — the sharpening transition that is the operational
 * signature of the capacity. A cream cursor reads every curve off at the chosen
 * p; a reseed control redraws the Monte-Carlo noise; an inset shows one matrix
 * being erased and decoded at the live p.
 *
 * Ported from the post's section-8 decoder (X = Gu over GF(2), erase with
 * 1−p, recover a coordinate iff its generator row is spanned by the observed
 * rows); the curves reproduce artifacts/symbol-error.png.
 */
export function createSymbolError(target: HTMLElement, options: SymbolErrorOptions = {}): SymbolErrorApi {
  const theme = withTheme(options.theme);
  const ks = (options.ks ?? [
    [4, 1],
    [8, 2],
    [16, 4],
    [32, 8],
  ]).map(([k, r]) => [Math.max(1, Math.round(k)), Math.max(1, Math.round(r))] as [number, number]);
  const trials = options.trials ?? 220;
  const gens = options.generators ?? 8;
  const G = options.grid ?? 49;
  let seed = (options.seed ?? 20260625) >>> 0 || 1;

  // The converse threshold p★ = 1/C = r/k (shared rate across all curves).
  const pStar = ks[0][1] / ks[0][0];

  // p grid on the open interval, matching the figure's span.
  const P_LO = 0.02;
  const P_HI = 0.98;
  const pGrid = new Float64Array(G);
  for (let i = 0; i < G; i++) pGrid[i] = P_LO + (P_HI - P_LO) * (i / (G - 1));

  // One error curve per (k, r); recomputed on reseed.
  let curves: Float64Array[] = ks.map(() => new Float64Array(G));

  // --- the exact GF(2) decoder (integer-XOR row reduction) -----------------
  // r ≤ 31, so each generator row is one bitmask integer over GF(2)^r.

  /** Reduce `row` against a reduced basis (pivot bit → mask); return its residue. */
  function reduce(row: number, basis: number[]): number {
    let v = row;
    for (const b of basis) {
      const piv = highBit(b);
      if (v & (1 << piv)) v ^= b;
    }
    return v;
  }

  /** Index of the highest set bit of a positive integer. */
  function highBit(x: number): number {
    let i = -1;
    while (x > 0) {
      i++;
      x >>= 1;
    }
    return i;
  }

  /** Insert `row` into the reduced basis; returns true if it raised the rank. */
  function insert(basis: number[], row: number): boolean {
    const v = reduce(row, basis);
    if (v === 0) return false;
    basis.push(v);
    basis.sort((a, b) => highBit(b) - highBit(a)); // highest pivot first
    return true;
  }

  /** Draw a generator G (k rows in GF(2)^r): every row nonzero, full rank r. */
  function drawGenerator(rng: Rng, k: number, r: number): number[] {
    for (let attempt = 0; attempt < 2000; attempt++) {
      const rows: number[] = new Array(k);
      const basis: number[] = [];
      let ok = true;
      for (let i = 0; i < k; i++) {
        let row = 0;
        for (let b = 0; b < r; b++) if (rng.next() < 0.5) row |= 1 << b;
        if (row === 0) {
          ok = false;
          break;
        }
        rows[i] = row;
        insert(basis, row);
      }
      if (ok && basis.length === r) return rows;
    }
    // Degenerate fallback (r ≥ k can't be full rank): identity-ish rows.
    return Array.from({ length: k }, (_, i) => 1 << (i % r));
  }

  /** Expected symbol-error rate ½·(undetermined/k) at rate p, Monte-Carlo. */
  function symbolError(rng: Rng, k: number, r: number, p: number): number {
    let acc = 0;
    for (let g = 0; g < gens; g++) {
      const rows = drawGenerator(rng, k, r);
      let und = 0;
      for (let t = 0; t < trials; t++) {
        // Observed rows span this basis; a coordinate is determined iff its
        // generator row reduces to 0 against it.
        const basis: number[] = [];
        for (let i = 0; i < k; i++) if (rng.next() < p) insert(basis, rows[i]);
        for (let j = 0; j < k; j++) if (reduce(rows[j], basis) !== 0) und++;
      }
      acc += (0.5 * und) / (k * trials);
    }
    return acc / gens;
  }

  function buildCurves(): void {
    const rng = new Rng(seed);
    curves = ks.map(([k, r]) => {
      const out = new Float64Array(G);
      for (let i = 0; i < G; i++) out[i] = symbolError(rng, k, r, pGrid[i]);
      return out;
    });
  }
  buildCurves();

  // Continuous sampling of a curve at an arbitrary p (linear interpolation).
  function curveAt(ci: number, p: number): number {
    const c = curves[ci];
    const f = clamp((p - P_LO) / (P_HI - P_LO), 0, 1) * (G - 1);
    const lo = Math.floor(f);
    const hi = Math.min(lo + 1, G - 1);
    return c[lo] * (1 - (f - lo)) + c[hi] * (f - lo);
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  const overlay = overlayLayer("tl");
  const eqP = liveEquation("p\\;=", theme, theme.cream);
  const eqErr = liveEquation("\\tfrac12\\,\\mathbb{E}\\!\\left[\\tfrac{\\#\\text{undet.}}{k}\\right]\\;=", theme, theme.red);
  overlay.append(eqP.node, eqErr.node);
  panel.append(overlay);

  const thrOverlay = overlayLayer("tr");
  thrOverlay.append(
    staticEquation(`p^{\\star}=\\tfrac1C=\\tfrac{r}{k}=${fmtRate(pStar)}`, theme, theme.gold),
    staticEquation(`X=G\\,u\\ \\ \\bmod 2`, theme, theme.muted),
  );
  panel.append(thrOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const snapBtn = el("button", { type: "button", style: btnStyle(theme) });
  snapBtn.innerHTML = "&#9650;&nbsp; to p&#9733;";
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "observation rate  p";
  const slider = el("input", {
    type: "range",
    min: String(Math.round(P_LO * 1000)),
    max: String(Math.round(P_HI * 1000)),
    step: "1",
    value: String(Math.round(clamp(options.p ?? pStar, P_LO, P_HI) * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "observation rate p between 0 and 1",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · snap to p★ · reseed · ←→";
  controls.append(snapBtn, reseedBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the operating point p ------------------------
  let p = clamp(options.p ?? pStar, P_LO, P_HI);
  let destroyed = false;
  const pTracker = valueTracker(p, (v) => {
    p = clamp(v, P_LO, P_HI);
    render();
  });

  function render(): void {
    draw();
    eqP.set(p.toFixed(3));
    // The headline error is the largest-k curve (the sharpest transition).
    eqErr.set(curveAt(ks.length - 1, p).toFixed(3));
    if (document.activeElement !== slider) slider.value = String(Math.round(p * 1000));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 16;
    const mt = 60;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const yMax = 0.5;

    const X = (pp: number) => ml + clamp((pp - 0) / 1, 0, 1) * plotW;
    const Y = (e: number) => mt + plotH - (clamp(e, 0, yMax) / yMax) * plotH;

    // faint receding number plane: vertical p grid + horizontal error grid
    for (let g = 0; g <= 1.0001; g += 0.2) {
      strokeLine(ctx, [X(g), mt], [X(g), mt + plotH], theme.grid, 1, g === 0 ? 0.3 : 0.16);
    }
    for (let e = 0; e <= yMax + 1e-6; e += 0.1) {
      strokeLine(ctx, [ml, Y(e)], [W - mr, Y(e)], theme.grid, 1, e === 0 ? 0.3 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.5, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.5, 1);

    // the converse threshold p★ (the answer): a gold dashed vertical rule
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[X(pStar), mt], [X(pStar), mt + plotH]], {
        color: theme.gold,
        width: 1.6,
        alpha: 0.9,
        dash: [7, 6],
      });
    });
    mathLabel(ctx, "p", [X(pStar) + 5, mt + 13], { color: theme.gold, size: 14, sub: "★", glow: 5 });

    // the four error curves, each a different (k, r) at the same rate
    ks.forEach((_pair, ci) => {
      const col = ROW_COLORS[ci % ROW_COLORS.length];
      const pts: Pt[] = [];
      for (let i = 0; i < G; i++) pts.push([X(pGrid[i]), Y(curves[ci][i])]);
      const isTop = ci === ks.length - 1;
      glow(ctx, col, isTop ? 5 : 0, () => {
        polyline(ctx, pts, { color: col, width: isTop ? 2.6 : 1.9, alpha: isTop ? 1 : 0.85 });
      });
    });

    // the cream cursor at the operating point p: a vertical rule + a read-off
    // dot on every curve, with the headline (largest-k) error called out.
    const cx = X(p);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.3, 0.55);
    ks.forEach((_pair, ci) => {
      const col = ROW_COLORS[ci % ROW_COLORS.length];
      const e = curveAt(ci, p);
      const isTop = ci === ks.length - 1;
      if (isTop) glow(ctx, theme.cream, 7, () => disc(ctx, [cx, Y(e)], 3.8, theme.cream, 1));
      else disc(ctx, [cx, Y(e)], 2.6, col, 0.95);
    });
    mathLabel(ctx, "p", [cx + (p < 0.85 ? 6 : -6), mt + plotH - 8], {
      color: theme.cream,
      size: 14,
      align: p < 0.85 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) + ticks (roman numerals)
    mathLabel(ctx, "p", [W - mr - 2, mt + plotH + 24], { color: theme.tick, size: 15, align: "right" });
    ctx.save();
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = 0; g <= 1.0001; g += 0.2) ctx.fillText(g.toFixed(1), X(g), mt + plotH + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let e = 0; e <= yMax + 1e-6; e += 0.1) ctx.fillText(e.toFixed(1), ml - 6, Y(e));
    ctx.restore();
    // y-axis caption
    ctx.save();
    ctx.translate(14, mt + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = theme.muted;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("expected symbol-error rate", 0, 0);
    ctx.restore();

    drawLegend(ctx, W - mr, mt);
    drawDecoderInset(ctx, W, H);
  }

  /** The curve legend, top-right under the threshold overlay. */
  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    let yy = top + 6;
    ks.forEach(([k, r], ci) => {
      const col = ROW_COLORS[ci % ROW_COLORS.length];
      const lx = right - 92;
      strokeLine(ctx, [lx, yy], [lx + 20, yy], col, ci === ks.length - 1 ? 2.6 : 1.9, 1);
      ctx.fillStyle = theme.fg;
      ctx.fillText(`k=${k}, r=${r}`, lx + 26, yy + 0.5);
      yy += 15;
    });
    ctx.restore();
  }

  /**
   * Operational-decoder inset (bottom-left): one k×n GF(2) matrix at the live p.
   * Observed cells glow blue; erased cells are dim. A column is tinted green when
   * fully determined (its information set is observed) and red when some of its
   * symbols are undetermined — the very event the error rate counts.
   */
  function drawDecoderInset(ctx: CanvasRenderingContext2D, W: number, H: number): void {
    const [k, r] = ks[1] ?? ks[0]; // a legible mid-size matrix (k=8)
    const nCols = 14;
    const cell = Math.max(5, Math.min(9, (W * 0.26) / nCols));
    const gw = nCols * cell;
    const gh = k * cell;
    const ox = 14;
    const oy = H - 14 - gh;

    // a deterministic per-frame view: erase with a frame-stable RNG seeded by p
    const fr = new Rng(((seed ^ Math.round(p * 100000)) >>> 0) || 1);
    const rows = drawGenerator(fr, k, r);

    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(ox - 6, oy - 18, gw + 12, gh + 24);
    ctx.globalAlpha = 1;
    ctx.fillStyle = theme.muted;
    ctx.font = `10px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`decode  ${k}×${nCols}  ·  erase 1−p`, ox, oy - 6);

    for (let c = 0; c < nCols; c++) {
      // observed rows for this column
      const basis: number[] = [];
      const obs: boolean[] = new Array(k);
      for (let i = 0; i < k; i++) {
        const o = fr.next() < p;
        obs[i] = o;
        if (o) insert(basis, rows[i]);
      }
      let colDetermined = true;
      for (let i = 0; i < k; i++) {
        const det = reduce(rows[i], basis) === 0;
        if (!det) colDetermined = false;
        const x = ox + c * cell;
        const y = oy + i * cell;
        if (obs[i]) {
          // observed: a bright blue cell (the datum)
          ctx.fillStyle = theme.blue;
          ctx.globalAlpha = 0.9;
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        } else if (det) {
          // recovered by elimination: faint green
          ctx.fillStyle = theme.green;
          ctx.globalAlpha = 0.4;
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        } else {
          // undetermined: faint red — these are the errors
          ctx.fillStyle = theme.red;
          ctx.globalAlpha = 0.42;
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
        ctx.globalAlpha = 1;
      }
      // column outline coloured by whether the whole column decoded
      ctx.strokeStyle = colDetermined ? theme.green : theme.red;
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1;
      ctx.strokeRect(ox + c * cell + 0.5, oy + 0.5, cell - 1, gh - 1);
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  // --- interaction ---------------------------------------------------------
  function setP(np: number, animate = false): void {
    const t = clamp(np, P_LO, P_HI);
    if (animate) pTracker.set(t, true);
    else pTracker.jump(t);
  }

  function reseed(): void {
    seed = (seed + 0x9e3779b1) >>> 0 || 1;
    buildCurves();
    render();
  }

  snapBtn.addEventListener("click", () => setP(pStar, true));
  reseedBtn.addEventListener("click", reseed);
  slider.addEventListener("input", () => pTracker.jump(Number(slider.value) / 1000));

  // Horizontal drag on the canvas scrubs the operating point p.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 46;
      const mr = 16;
      const plotW = rc.width - ml - mr;
      pTracker.jump(clamp((px - ml) / plotW, P_LO, P_HI));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Operational decoder: expected GF(2) symbol-error rate versus observation rate p, for four matrix heights at code rate one quarter. The curves pivot at the converse threshold p-star = r/k = one quarter, where the transition sharpens as the matrix grows. Drag to move the operating point; an inset shows one matrix being erased and decoded.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setP(pStar, true);
      return;
    }
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setP(p - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setP(p + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land exactly on the threshold so there is no glide.
  if (prefersReducedMotion()) p = pStar;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setP,
    snapToThreshold: () => setP(pStar, true),
    reseed,
    destroy() {
      destroyed = true;
      pTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- helpers ----------------------------------------------------------------

/** A compact rational for a rate like 1/4, or a decimal fallback. */
function fmtRate(x: number): string {
  for (const d of [2, 3, 4, 5, 6, 8]) {
    const n = Math.round(x * d);
    if (Math.abs(x - n / d) < 1e-9 && n >= 1 && n < d) return `\\tfrac{${n}}{${d}}`;
  }
  return x.toFixed(3);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
