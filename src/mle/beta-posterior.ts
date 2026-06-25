import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface BetaPosteriorOptions {
  /** Prior pseudo-counts, Beta(α, β). Defaults to the post's Beta(2,2). */
  alpha?: number;
  beta?: number;
  /** Observed successes k (heads). Default 7. */
  heads?: number;
  /** Observed failures n−k (tails). Default 3, giving n = 10. */
  tails?: number;
  theme?: Partial<Theme>;
}

export interface BetaPosteriorApi {
  setHeads(k: number): void;
  setTails(t: number): void;
  setAlpha(a: number): void;
  setBeta(b: number): void;
  /** Move the read-off cursor to θ (eased). */
  setCursor(theta: number, animate?: boolean): void;
  destroy(): void;
}

/**
 * Scene — "the prior is a regularizer". Over θ ∈ [0,1] three densities are drawn:
 * the prior Beta(α,β), the normalized Bernoulli likelihood θ^k(1−θ)^(n−k), and
 * the conjugate posterior Beta(k+α, n−k+β). Three vertical markers — the MLE
 * k/n, the MAP mode (k+α−1)/(n+α+β−2), and the posterior mean (k+α)/(n+α+β) —
 * pull apart at finite n and collapse onto one another as the data grow or the
 * prior flattens. Sliders dial the prior pseudo-counts and the observed heads /
 * tails; a draggable cursor reads each density off at a chosen θ.
 *
 * Ported from the post's notebook cell (k=7, n=10, Beta(2,2)); the likelihood is
 * normalized by trapezoid integration over the same grid, exactly as there.
 */
export function createBetaPosterior(
  target: HTMLElement,
  options: BetaPosteriorOptions = {},
): BetaPosteriorApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let a0 = clamp(Math.round(options.alpha ?? 2), 1, 12);
  let b0 = clamp(Math.round(options.beta ?? 2), 1, 12);
  let heads = clamp(Math.round(options.heads ?? 7), 0, 60);
  let tails = clamp(Math.round(options.tails ?? 3), 0, 60);
  let destroyed = false;

  // θ grid (matches the notebook: 600 points on the open interval).
  const G = 600;
  const th = new Float64Array(G);
  for (let i = 0; i < G; i++) th[i] = 1e-3 + (1 - 2e-3) * (i / (G - 1));

  // Reusable density buffers, recomputed in place each frame.
  const prior = new Float64Array(G);
  const lik = new Float64Array(G);
  const post = new Float64Array(G);

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the three estimators, live
  const overlay = overlayLayer("tl");
  const eqMle = liveEquation("\\hat\\theta_{\\text{MLE}}=", theme, theme.gold);
  const eqMap = liveEquation("\\hat\\theta_{\\text{MAP}}=", theme, theme.blue);
  const eqMean = liveEquation("\\mathbb{E}[\\theta\\mid D]=", theme, theme.cream);
  overlay.append(eqMle.node, eqMap.node, eqMean.node);
  panel.append(overlay);

  // top-right: the conjugate update, restated for the current numbers
  const postOverlay = overlayLayer("tr");
  const eqPost = staticEquation("", theme, theme.blue);
  postOverlay.append(eqPost);
  panel.append(postOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));gap:8px 18px;align-items:center;",
  });

  const headsRow = sliderRow(theme, "heads  k", 0, 60, heads, "observed successes (heads)");
  const tailsRow = sliderRow(theme, "tails  n−k", 0, 60, tails, "observed failures (tails)");
  const alphaRow = sliderRow(theme, "prior  α", 1, 12, a0, "prior pseudo-count alpha");
  const betaRow = sliderRow(theme, "prior  β", 1, 12, b0, "prior pseudo-count beta");
  controls.append(headsRow.row, tailsRow.row, alphaRow.row, betaRow.row);
  root.append(controls);

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag the curve to read off θ  ·  ←→ scrub  ·  sliders set the prior & data";
  root.append(hint);

  target.append(root);

  // --- the one eased scalar: the read-off cursor over θ --------------------
  let cursor = options.heads !== undefined || options.tails !== undefined ? heads / Math.max(heads + tails, 1) : 0.7;
  const cursorTracker = valueTracker(cursor, (v) => {
    cursor = clamp(v, 0, 1);
    render();
  });

  // --- math: Beta pdf via a Lanczos log-gamma (no scipy) -------------------
  function betaPdfInto(out: Float64Array, a: number, b: number): void {
    // log B(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b); pdf = θ^{a-1}(1-θ)^{b-1}/B.
    const logNorm = lgamma(a) + lgamma(b) - lgamma(a + b);
    for (let i = 0; i < G; i++) {
      const t = th[i];
      const logp = (a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - logNorm;
      out[i] = Math.exp(logp);
    }
  }

  function likelihoodInto(out: Float64Array, k: number, nk: number): void {
    // Unnormalized θ^k (1-θ)^{n-k}, then divide by its trapezoid integral so the
    // curve sits on the same density scale as the prior and posterior — exactly
    // as the notebook does with np.trapezoid.
    let peak = -Infinity;
    for (let i = 0; i < G; i++) {
      const t = th[i];
      const logl = k * Math.log(t) + nk * Math.log(1 - t);
      out[i] = logl;
      if (logl > peak) peak = logl;
    }
    let area = 0;
    let prevV = Math.exp(out[0] - peak);
    out[0] = prevV;
    for (let i = 1; i < G; i++) {
      const v = Math.exp(out[i] - peak);
      out[i] = v;
      area += ((prevV + v) / 2) * (th[i] - th[i - 1]);
      prevV = v;
    }
    const inv = area > 0 ? 1 / area : 0;
    for (let i = 0; i < G; i++) out[i] *= inv;
  }

  function recompute(): void {
    betaPdfInto(prior, a0, b0);
    likelihoodInto(lik, heads, tails);
    betaPdfInto(post, heads + a0, tails + b0);
  }

  // estimators (exact closed forms from the post)
  const mleOf = () => (heads + tails > 0 ? heads / (heads + tails) : NaN);
  const mapOf = () => {
    const denom = heads + tails + a0 + b0 - 2;
    return denom > 0 ? (heads + a0 - 1) / denom : NaN;
  };
  const meanOf = () => (heads + a0) / (heads + tails + a0 + b0);

  // linear interpolation of a density at a continuous θ
  function densAt(arr: Float64Array, t: number): number {
    const x = clamp(t, th[0], th[G - 1]);
    const f = ((x - 1e-3) / (1 - 2e-3)) * (G - 1);
    const i = clamp(Math.floor(f), 0, G - 2);
    const frac = f - i;
    return arr[i] * (1 - frac) + arr[i + 1] * frac;
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    recompute();
    draw();

    const mle = mleOf();
    const map = mapOf();
    const mean = meanOf();
    eqMle.set(Number.isFinite(mle) ? mle.toFixed(3) : "—");
    eqMap.set(Number.isFinite(map) ? map.toFixed(3) : "—");
    eqMean.set(mean.toFixed(3));
    eqPost.innerHTML = staticEquation(
      `\\text{posterior}=\\operatorname{Beta}(k{+}\\alpha,\\,n{-}k{+}\\beta)=\\operatorname{Beta}(${heads + a0},\\,${tails + b0})`,
      theme,
      theme.blue,
    ).innerHTML;

    // keep sliders in sync when changed programmatically
    if (document.activeElement !== headsRow.input) headsRow.set(heads);
    if (document.activeElement !== tailsRow.input) tailsRow.set(tails);
    if (document.activeElement !== alphaRow.input) alphaRow.set(a0);
    if (document.activeElement !== betaRow.input) betaRow.set(b0);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 18;
    const mr = 16;
    const mt = 16;
    const mb = 30;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // shared vertical scale across all three densities
    let ymax = 1e-6;
    for (let i = 0; i < G; i++) {
      if (prior[i] > ymax) ymax = prior[i];
      if (lik[i] > ymax) ymax = lik[i];
      if (post[i] > ymax) ymax = post[i];
    }
    ymax *= 1.1;

    const x = (t: number) => ml + clamp(t, 0, 1) * plotW;
    const y = (d: number) => mt + plotH - (clamp(d, 0, ymax) / ymax) * plotH;

    // faint vertical grid at the quartiles (the cyan plane, receding)
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      strokeLine(ctx, [x(g), mt], [x(g), mt + plotH], theme.grid, 1, g === 0 || g === 1 ? 0.28 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    const toScreen = (arr: Float64Array): Pt[] => {
      const out: Pt[] = new Array(G);
      for (let i = 0; i < G; i++) out[i] = [x(th[i]), y(arr[i])];
      return out;
    };

    // prior (muted, dotted) — what we believe before the data
    polyline(ctx, toScreen(prior), { color: theme.muted, width: 1.6, alpha: 0.9, dash: [2, 4] });
    // likelihood (gold, dashed) — what the data alone say; its peak is the MLE
    polyline(ctx, toScreen(lik), { color: theme.gold, width: 1.9, alpha: 0.85, dash: [7, 5] });
    // posterior (bright blue, solid, faintly glowing) — the answer object
    const postPts = toScreen(post);
    const fillPts: Pt[] = [[x(0), mt + plotH], ...postPts, [x(1), mt + plotH]];
    polyline(ctx, fillPts, { color: "transparent", width: 0, alpha: 1, closed: true, fill: hexA(theme.blue, 0.1) });
    ctx.save();
    ctx.shadowColor = theme.blue;
    ctx.shadowBlur = 6;
    polyline(ctx, postPts, { color: theme.blue, width: 2.4, alpha: 1 });
    ctx.restore();

    // estimator markers (vertical rules + a tick on the posterior)
    const mle = mleOf();
    const map = mapOf();
    const mean = meanOf();
    estimatorMark(ctx, x, y, mt, plotH, mle, theme.gold, "MLE", densAt(post, mle));
    estimatorMark(ctx, x, y, mt, plotH, map, theme.blue, "MAP", densAt(post, map));
    estimatorMark(ctx, x, y, mt, plotH, mean, theme.cream, "mean", densAt(post, mean));

    // the read-off cursor (the movable variable): vertical line + read every curve
    const cx = x(cursor);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.55);
    glow(ctx, theme.cream, 7, () => disc(ctx, [cx, y(densAt(post, cursor))], 3.6, theme.cream, 1));
    disc(ctx, [cx, y(densAt(prior, cursor))], 2.6, theme.muted, 0.95);
    disc(ctx, [cx, y(densAt(lik, cursor))], 2.6, theme.gold, 0.95);
    mathLabel(ctx, "θ", [cx + (cursor < 0.9 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 15,
      align: cursor < 0.9 ? "left" : "right",
      glow: 5,
    });

    // x ticks
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(t.toFixed(2), x(t), mt + plotH + 5);

    // legend
    drawLegend(ctx, W - mr, mt);
  }

  function estimatorMark(
    ctx: CanvasRenderingContext2D,
    x: (t: number) => number,
    y: (d: number) => number,
    mt: number,
    plotH: number,
    t: number,
    color: string,
    tag: string,
    dens: number,
  ): void {
    if (!Number.isFinite(t)) return;
    const px = x(t);
    strokeLine(ctx, [px, mt + 4], [px, mt + plotH], color, 1.3, 0.7);
    glow(ctx, color, 5, () => disc(ctx, [px, y(dens)], 3, color, 1));
    ctx.fillStyle = color;
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    ctx.textAlign = t < 0.85 ? "left" : "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(tag, px + (t < 0.85 ? 4 : -4), mt + plotH - 4);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["prior", theme.muted, [2, 4]],
      ["likelihood", theme.gold, [7, 5]],
      ["posterior", theme.blue, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = top + 46;
    for (const [name, color, dash] of rows) {
      const lx = right - 96;
      ctx.strokeStyle = color;
      ctx.lineWidth = name === "posterior" ? 2.4 : 1.8;
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
  function setCursor(theta: number, animate = false): void {
    if (animate) cursorTracker.set(clamp(theta, 0, 1), true);
    else cursorTracker.jump(clamp(theta, 0, 1));
  }

  function setHeads(k: number): void {
    heads = clamp(Math.round(k), 0, 60);
    render();
  }
  function setTails(t: number): void {
    tails = clamp(Math.round(t), 0, 60);
    render();
  }
  function setAlpha(a: number): void {
    a0 = clamp(Math.round(a), 1, 12);
    render();
  }
  function setBeta(b: number): void {
    b0 = clamp(Math.round(b), 1, 12);
    render();
  }

  headsRow.input.addEventListener("input", () => setHeads(Number(headsRow.input.value)));
  tailsRow.input.addEventListener("input", () => setTails(Number(tailsRow.input.value)));
  alphaRow.input.addEventListener("input", () => setAlpha(Number(alphaRow.input.value)));
  betaRow.input.addEventListener("input", () => setBeta(Number(betaRow.input.value)));

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 18;
      const mr = 16;
      const plotW = rc.width - ml - mr;
      cursorTracker.jump(clamp((px - ml) / plotW, 0, 1));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Beta-Bernoulli inference: a prior Beta density, the normalized Bernoulli likelihood, and the posterior Beta density over theta, with the MLE, MAP mode, and posterior mean marked. Drag to read each curve off at a chosen theta.",
  );

  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setCursor(cursor - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setCursor(cursor + step, true);
    else if (e.key === "Home") setCursor(mleOf(), true);
    else if (e.key === "End") setCursor(meanOf(), true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the posterior mean immediately so there's no glide.
  if (prefersReducedMotion()) cursor = clamp(meanOf(), 0, 1);

  render();
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setHeads,
    setTails,
    setAlpha,
    setBeta,
    setCursor,
    destroy() {
      destroyed = true;
      cursorTracker.stop();
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
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:2.5ch;text-align:right;` });
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

/** Run `body` with an additive glow, like chalk on a dark board. */
function glow(ctx: CanvasRenderingContext2D, color: string, blur: number, body: () => void): void {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  body();
  ctx.restore();
}

/** Lanczos approximation of ln Γ(x), good to ~1e−10 for x > 0. */
function lgamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    // reflection formula
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Append an alpha byte to a #rrggbb color. */
function hexA(hex: string, alpha: number): string {
  const a = Math.round(clamp(alpha, 0, 1) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}
