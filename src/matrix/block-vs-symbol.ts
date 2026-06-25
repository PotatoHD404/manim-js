import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface BlockVsSymbolOptions {
  /**
   * Per-column non-recovery probability q (the "symbol" error). When
   * `deriveFromDecoder` is left on, this is overwritten by the seeded GF(2)
   * decoding experiment for (k, p); otherwise the slider sets it directly.
   */
  q?: number;
  /** Block length n: number of independent columns. Default 12. */
  n?: number;
  /** Matrix height k for the operational decoder (rate r/k = 1/4). Default 8. */
  k?: number;
  /** Observation rate p = m/(kn) for the decoder. Default 0.35 > p★ = 1/4. */
  p?: number;
  /** Derive q from the seeded GF(2) decoder (k, p) instead of the q slider. */
  deriveFromDecoder?: boolean;
  /** RNG seed for the Monte-Carlo decoder. */
  seed?: number;
  theme?: Partial<Theme>;
}

export interface BlockVsSymbolApi {
  /** Set the block length n (eased). */
  setN(n: number, animate?: boolean): void;
  /** Set the per-column failure q directly (switches off the decoder). */
  setQ(q: number): void;
  /** Set the decoder observation rate p (re-derives q when the decoder is on). */
  setP(p: number): void;
  /** Set the decoder matrix height k (re-derives q when the decoder is on). */
  setK(k: number): void;
  /** Draw a fresh Monte-Carlo estimate of q from the next seed. */
  reseed(): void;
  destroy(): void;
}

/** Rate of the rank-r binary-code model in the post: r/k = 1/4. */
const RATE = 0.25;
/** Converse / exact-recovery threshold p★ = 1/C = r/k. */
const P_STAR = RATE;
/** Monte-Carlo columns drawn per q estimate (the decoder is per-column). */
const MC_TRIALS = 4000;
/** Largest block length the amplification curve is drawn out to. */
const N_MAX = 512;

/**
 * Scene — "block error amplifies the symbol error". Take the post's rank-r
 * binary-code model literally (section 8): a k×n matrix whose columns are random
 * GF(2) codewords X = G u, each entry erased with probability 1−p, decoded
 * column-by-column by Gaussian elimination over GF(2). A single column is
 * recoverable iff its observed rows span the r-dimensional row space, so the
 * per-column non-recovery probability is q(k,p) = P[ rank(G over observed rows)
 * < r ] — estimated here by a seeded Monte-Carlo XOR row-reduction, exactly the
 * experiment described in the post. The whole-matrix (block) error is then the
 * amplification 1 − (1−q)^n, which climbs to 1 as the matrix widens even though
 * the per-column failure q is fixed. Above p★ = 1/C = r/k the per-column q
 * vanishes exponentially in k (so widen-and-grow-k recovers exactly); below it,
 * q saturates and the block error is hopeless.
 *
 * The gold curve is the block error 1−(1−q)^n over n (log-x); the red dashed
 * rule is the constant symbol error q; the cream marker is the chosen block
 * length n and the green trace is the block error realized up to it. Sliders set
 * the block length n and the per-column q; secondary sliders (k, p) and a reseed
 * button drive q from the real decoder.
 */
export function createBlockVsSymbol(
  target: HTMLElement,
  options: BlockVsSymbolOptions = {},
): BlockVsSymbolApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let n = clamp(Math.round(options.n ?? 12), 1, N_MAX);
  let k = clamp(Math.round(options.k ?? 8), 4, 64);
  let p = clamp(options.p ?? 0.35, 0.02, 0.98);
  let deriveFromDecoder = options.deriveFromDecoder ?? true;
  let seed = (options.seed ?? 20260625) >>> 0 || 1;
  // q is either the slider value or the decoder estimate; seeded so it is stable.
  let qDirect = clamp(options.q ?? 0.34, 1e-4, 0.999);
  let q = deriveFromDecoder ? estimateQ(k, p, seed) : qDirect;
  let destroyed = false;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the live block error, the symbol error, and the amplification ratio
  const overlay = overlayLayer("tl");
  const eqBlock = liveEquation("P_{\\text{block}}=1-(1-q)^{n}\\;=", theme, theme.gold);
  const eqSym = liveEquation("q\\;=", theme, theme.red);
  const eqAmp = liveEquation("P_{\\text{block}}/q\\;=", theme, theme.green);
  overlay.append(eqBlock.node, eqSym.node, eqAmp.node);
  panel.append(overlay);

  // top-right: the operational model, restated for the current numbers
  const modelOverlay = overlayLayer("tr");
  const eqModel = staticEquation("", theme, theme.blue);
  const eqStar = staticEquation(`p^{\\star}=1/C=r/k=${P_STAR.toFixed(2)}`, theme, theme.muted);
  modelOverlay.append(eqModel, eqStar);
  panel.append(modelOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:8px 18px;align-items:center;",
  });
  const nRow = sliderRow(theme, "block  n", 1, N_MAX, n, "block length n (number of columns)");
  const qRow = sliderRow(theme, "symbol  q", 1, 999, Math.round(qDirect * 1000), "per-column (symbol) failure probability q, in thousandths");
  const pRow = sliderRow(theme, "rate  p", 2, 98, Math.round(p * 100), "observation rate p as a percentage");
  const kRow = sliderRow(theme, "height  k", 4, 64, k, "matrix height k for the decoder");
  controls.append(nRow.row, qRow.row, pRow.row, kRow.row);
  root.append(controls);

  const buttons = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const deriveBtn = el("button", { type: "button", style: btnStyle(theme) });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed decoder";
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag the curve to set n  ·  ←→ scrub  ·  q from sliders or the GF(2) decoder";
  buttons.append(deriveBtn, reseedBtn, hint);
  root.append(buttons);

  target.append(root);

  // --- the one eased scalar: the block length n on a log axis --------------
  // Track log10(n) so dragging across the decades feels uniform, like the plot.
  const lnMin = 0;
  const lnMax = Math.log10(N_MAX);
  let logN = clamp(Math.log10(n), lnMin, lnMax);
  const nTracker = valueTracker(logN, (v) => {
    logN = clamp(v, lnMin, lnMax);
    n = clamp(Math.round(Math.pow(10, logN)), 1, N_MAX);
    render();
  });

  // --- math ----------------------------------------------------------------
  const blockError = (qq: number, nn: number): number => 1 - Math.pow(1 - qq, nn);

  function recomputeQ(): void {
    q = deriveFromDecoder ? estimateQ(k, p, seed) : qDirect;
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    recomputeQ();
    draw();

    const r = Math.max(1, Math.round(RATE * k));
    const pb = blockError(q, n);
    eqBlock.set(pb.toFixed(4));
    eqSym.set(q.toFixed(4));
    eqAmp.set(q > 1e-9 ? `${(pb / q).toFixed(1)}×` : "—");
    eqModel.innerHTML = staticEquation(
      `X=Gu\\in\\mathrm{GF}(2)^{${k}},\\ r=${r},\\ p=${p.toFixed(2)}`,
      theme,
      p > P_STAR ? theme.blue : theme.red,
    ).innerHTML;

    // keep controls in sync when changed programmatically
    if (document.activeElement !== nRow.input) nRow.set(n);
    if (document.activeElement !== qRow.input) qRow.set(Math.round(qDirect * 1000));
    if (document.activeElement !== pRow.input) pRow.set(Math.round(p * 100));
    if (document.activeElement !== kRow.input) kRow.set(k);
    deriveBtn.innerHTML = deriveFromDecoder ? "q: decoder &#10003;" : "q: slider";
    qRow.input.disabled = deriveFromDecoder;
    qRow.row.style.opacity = deriveFromDecoder ? "0.45" : "1";
    pRow.input.disabled = !deriveFromDecoder;
    kRow.input.disabled = !deriveFromDecoder;
    pRow.row.style.opacity = deriveFromDecoder ? "1" : "0.45";
    kRow.row.style.opacity = deriveFromDecoder ? "1" : "0.45";
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

    // n on a log axis [1, N_MAX]; block error on [0, 1].
    const X = (nn: number) => ml + (clamp(Math.log10(clamp(nn, 1, N_MAX)), lnMin, lnMax) / lnMax) * plotW;
    const Y = (f: number) => mt + plotH - clamp(f, 0, 1) * plotH;

    // faint number-plane grid: vertical at each decade + minor lines, horizontal
    // at the error gridlines (the cyan plane, receding).
    for (let d = 0; d <= lnMax + 1e-9; d++) {
      const base = Math.pow(10, d);
      strokeLine(ctx, [X(base), mt], [X(base), mt + plotH], theme.grid, 1, 0.3);
      for (let mul = 2; mul <= 9; mul++) {
        const xv = base * mul;
        if (xv > N_MAX) break;
        strokeLine(ctx, [X(xv), mt], [X(xv), mt + plotH], theme.grid, 1, 0.12);
      }
    }
    for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      strokeLine(ctx, [ml, Y(g)], [W - mr, Y(g)], theme.grid, 1, g === 0 || g === 1 ? 0.3 : 0.16);
    }

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // the constant symbol error q — the thing that does NOT change with n.
    const yq = Y(q);
    polyline(ctx, [[ml, yq], [W - mr, yq]], { color: theme.red, width: 1.6, alpha: 0.85, dash: [6, 6] });
    mathLabel(ctx, "q", [ml + 8, yq - 6], { color: theme.red, size: 15, glow: 4 });

    // the block-error curve 1−(1−q)^n over n — the answer object, gold & glowing.
    const SAMPLES = 220;
    const curve: Pt[] = [];
    for (let i = 0; i < SAMPLES; i++) {
      const ln = lnMin + (lnMax - lnMin) * (i / (SAMPLES - 1));
      const nn = Math.pow(10, ln);
      curve.push([X(nn), Y(blockError(q, nn))]);
    }
    // soft fill under the curve to read the "filled to 1" amplification
    const fill: Pt[] = [[X(1), mt + plotH], ...curve, [W - mr, mt + plotH]];
    polyline(ctx, fill, { color: "transparent", width: 0, alpha: 1, closed: true, fill: hexA(theme.gold, 0.08) });
    glow(ctx, theme.gold, 6, () => {
      polyline(ctx, curve, { color: theme.gold, width: 2.6, alpha: 1 });
    });

    // the green "realized" trace from n=1 up to the chosen n (the block error you pay)
    const realized: Pt[] = curve.filter((pt) => pt[0] <= X(n) + 0.01);
    if (realized.length > 1) {
      glow(ctx, theme.green, 4, () => polyline(ctx, realized, { color: theme.green, width: 3, alpha: 1 }));
    }

    // the movable variable: the chosen block length n.
    const nx = X(n);
    const pb = blockError(q, n);
    const ny = Y(pb);
    strokeLine(ctx, [nx, mt], [nx, mt + plotH], theme.cream, 1.2, 0.6);
    // drop the achieved block error onto the y-axis
    strokeLine(ctx, [ml, ny], [nx, ny], theme.cream, 1, 0.45);
    glow(ctx, theme.cream, 8, () => disc(ctx, [nx, ny], 4.4, theme.cream, 1));
    mathLabel(ctx, "n", [nx + (n < N_MAX * 0.5 ? 7 : -7), mt + 15], {
      color: theme.cream,
      size: 16,
      align: n < N_MAX * 0.5 ? "left" : "right",
      glow: 5,
    });

    // x ticks at decades (roman numerals), with "10^d" written via mathLabel
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let d = 0; d <= lnMax + 1e-9; d++) {
      const base = Math.pow(10, d);
      ctx.fillText(formatPow(base), X(base), mt + plotH + 6);
    }
    mathLabel(ctx, "n", [W - mr, mt + plotH + 26], { color: theme.tick, size: 14, align: "right" });

    // y ticks
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) ctx.fillText(g.toFixed(1), ml - 6, Y(g));
    mathLabel(ctx, "P", [ml - 28, mt + 10], { color: theme.tick, size: 14, sub: "block" });

    // legend
    drawLegend(ctx, W - mr, mt + plotH);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, bottom: number): void {
    const rows: [string, string, number[]][] = [
      ["block error  1−(1−q)ⁿ", theme.gold, []],
      ["symbol error  q", theme.red, [6, 6]],
      ["realized up to n", theme.green, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = bottom - 52;
    for (const [name, color, dash] of rows) {
      const lx = right - 150;
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

  // --- interaction ---------------------------------------------------------
  function setN(nn: number, animate = false): void {
    const target2 = clamp(Math.log10(clamp(Math.round(nn), 1, N_MAX)), lnMin, lnMax);
    if (animate) nTracker.set(target2, true);
    else nTracker.jump(target2);
  }
  function setQ(qq: number): void {
    deriveFromDecoder = false;
    qDirect = clamp(qq, 1e-4, 0.999);
    render();
  }
  function setP(pp: number): void {
    p = clamp(pp, 0.02, 0.98);
    if (deriveFromDecoder) render();
  }
  function setK(kk: number): void {
    k = clamp(Math.round(kk), 4, 64);
    if (deriveFromDecoder) render();
  }
  function reseed(): void {
    seed = (seed + 0x9e3779b1) >>> 0 || 1;
    render();
  }

  nRow.input.addEventListener("input", () => setN(Number(nRow.input.value)));
  qRow.input.addEventListener("input", () => setQ(Number(qRow.input.value) / 1000));
  pRow.input.addEventListener("input", () => setP(Number(pRow.input.value) / 100));
  kRow.input.addEventListener("input", () => setK(Number(kRow.input.value)));
  reseedBtn.addEventListener("click", reseed);
  deriveBtn.addEventListener("click", () => {
    deriveFromDecoder = !deriveFromDecoder;
    if (!deriveFromDecoder) qDirect = clamp(q, 1e-4, 0.999);
    render();
  });

  // Drag horizontally on the canvas to set the block length n (log scale).
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 46;
      const mr = 18;
      const plotW = rc.width - ml - mr;
      const f = clamp((px - ml) / plotW, 0, 1);
      nTracker.jump(lnMin + f * (lnMax - lnMin));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Block versus symbol error: the gold curve is the whole-matrix block error one minus one-minus-q to the n over the number of columns n on a log axis, the red dashed line is the constant per-column symbol error q, and a cream marker shows the block error at the chosen block length. Even a small fixed symbol error amplifies to a block error approaching one as the matrix widens.",
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setN(1, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setN(N_MAX, true);
      return;
    }
    const stepLog = e.shiftKey ? 0.3 : 0.08;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") nTracker.set(clamp(logN - stepLog, lnMin, lnMax), true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") nTracker.set(clamp(logN + stepLog, lnMin, lnMax), true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on a wide matrix immediately so the amplification reads
  // without a glide.
  if (prefersReducedMotion()) {
    logN = clamp(Math.log10(n), lnMin, lnMax);
  }

  render();
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setN,
    setQ,
    setP,
    setK,
    reseed,
    destroy() {
      destroyed = true;
      nTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- the operational decoder (post section 8, ported) -----------------------

/**
 * Seeded Monte-Carlo estimate of the per-column non-recovery probability q for
 * the rank-r binary-code model: columns are codewords X = G u over GF(2) with a
 * fixed random generator G (k rows, r = round(k/4) columns); each row is observed
 * independently with probability p. A column is uniquely recoverable iff its
 * observed rows span the r-dimensional row space — i.e. the observed generator
 * rows have full rank r over GF(2). q is the fraction of trials that do not.
 *
 * Recoverability depends only on the observed row set (not on u), so the estimate
 * is exactly E_mask[ rank_{GF(2)}(G over observed rows) < r ], computed with an
 * integer-bitmask XOR row reduction — the post's "integer-XOR row reduction".
 */
export function estimateQ(k: number, p: number, seed: number): number {
  const r = Math.max(1, Math.round(RATE * k));
  const rng = new Rng(seed);
  // Fixed full-rank generator: r distinct nonzero rows as r-bit bitmasks, drawn
  // from the same seed so the model is reproducible. Rows of G are vectors in
  // GF(2)^r; row i of the matrix is one such vector.
  const gen: number[] = [];
  for (let i = 0; i < k; i++) {
    let row = 0;
    for (let b = 0; b < r; b++) if (rng.next() < 0.5) row |= 1 << b;
    if (row === 0) row = 1 << (i % r); // avoid an all-zero generator row
    gen.push(row);
  }

  let fail = 0;
  for (let t = 0; t < MC_TRIALS; t++) {
    // Observed-row bitmasks of the generator, reduced to a basis over GF(2).
    const basis: number[] = [];
    let rank = 0;
    for (let i = 0; i < k && rank < r; i++) {
      if (rng.next() >= p) continue; // row i erased
      let v = gen[i];
      for (const b of basis) {
        const lead = b & -b; // lowest set bit of the basis vector
        if (v & lead) v ^= b;
      }
      if (v !== 0) {
        basis.push(v);
        rank++;
      }
    }
    if (rank < r) fail++;
  }
  return clamp(fail / MC_TRIALS, 1e-4, 0.999);
}

// --- small UI builder -------------------------------------------------------
interface Row {
  row: HTMLElement;
  input: HTMLInputElement;
  set(v: number): void;
}

function sliderRow(theme: Theme, label: string, min: number, max: number, value: number, aria: string): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:7.5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: "1",
    value: String(value),
    style: "flex:1;min-width:90px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:3.5ch;text-align:right;` });
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

// --- formatting & color helpers ---------------------------------------------

/** "1", "10", "100"… for decade ticks. */
function formatPow(v: number): string {
  return String(Math.round(v));
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
