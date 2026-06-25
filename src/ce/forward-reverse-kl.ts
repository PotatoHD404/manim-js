import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface ForwardReverseKlOptions {
  /** Number of grid points over x ∈ [−10, 10] (matches the notebook: 2000). */
  grid?: number;
  /** Seed for the bimodal target; seed 0 reproduces the notebook's exact bumps. */
  seed?: number;
  /** Initial mean of the candidate Gaussian q. */
  mu?: number;
  /** Initial standard deviation of q. */
  sigma?: number;
  theme?: Partial<Theme>;
}

export interface ForwardReverseKlApi {
  /** Move q's mean μ (eased). */
  setMu(mu: number, animate?: boolean): void;
  /** Move q's width σ (eased). */
  setSigma(sigma: number, animate?: boolean): void;
  /** Glide q onto the forward-KL optimum (mode-covering). */
  snapForward(): void;
  /** Glide q onto the reverse-KL optimum (mode-seeking). */
  snapReverse(): void;
  /** Draw a fresh seeded bimodal target and re-solve both optima. */
  reseed(): void;
  destroy(): void;
}

const X_LO = -10;
const X_HI = 10;
const MU_LO = -5;
const MU_HI = 5;
const SD_LO = 0.3;
const SD_HI = 5.0;
const EPS = 1e-12; // keeps logs finite — the notebook's `eps`

/**
 * Scene — "one Gaussian, two notions of 'closest'". A fixed bimodal target p
 * (two Gaussian bumps) is approximated by a single Gaussian q(μ,σ) the reader
 * drags around. Two divergences are read off live, exactly as the post's
 * notebook computes them on a grid:
 *
 *   forward  D_KL(p‖q) = Σ p·log((p+ε)/(q+ε))·dx   →  mode-COVERING (wide)
 *   reverse  D_KL(q‖p) = Σ q·log((q+ε)/(p+ε))·dx   →  mode-SEEKING  (one bump)
 *
 * The same 121×121 (μ,σ) grid scan as the notebook locates each optimum; the
 * gold curve is whichever divergence is being minimized, and the two snap
 * buttons glide q onto its forward (mean-seeking) and reverse (mode-seeking)
 * minimizers — the punchline that "closest" is two different statements.
 *
 * Ported verbatim from the post's bimodal-KL cell: target
 * p = ½·N(−3.0, 0.7) + ½·N(3.5, 1.0) (normalized on the grid), ε = 1e−12. A
 * reseed jitters the two bumps to a fresh seeded target and re-solves; seed 0
 * reproduces the notebook's exact figure.
 */
export function createForwardReverseKl(
  target: HTMLElement,
  options: ForwardReverseKlOptions = {},
): ForwardReverseKlApi {
  const theme = withTheme(options.theme);
  const G = options.grid ?? 2000;
  let seed = (options.seed ?? 0) >>> 0;

  // x grid and step, matching np.linspace(-10, 10, G).
  const x = new Float64Array(G);
  for (let i = 0; i < G; i++) x[i] = X_LO + (X_HI - X_LO) * (i / (G - 1));
  const dx = x[1] - x[0];

  // Reusable density buffers, recomputed in place each frame.
  const p = new Float64Array(G); // fixed target (set by buildTarget)
  const q = new Float64Array(G); // the movable candidate

  // --- the bimodal target (the fixed "data"; reseed jitters the bumps) ------
  // Component params: [(weight, mean, sd), …]. Seed 0 = the notebook's bumps.
  let bumps: Array<[number, number, number]> = [];
  // Both optima, located by the notebook's grid scan over (μ, σ).
  let fwdOpt: [number, number] = [0, 1];
  let revOpt: [number, number] = [0, 1];
  let pMax = 1;

  function buildTarget(): void {
    if (seed === 0) {
      bumps = [
        [0.5, -3.0, 0.7],
        [0.5, 3.5, 1.0],
      ];
    } else {
      // A reseeded two-bump target: well-separated means, distinct heights, so
      // the mode-covering / mode-seeking split stays vivid. Same construction
      // as the notebook (mixture of two Gaussians), only the bumps move.
      const rng = new Rng(seed || 1);
      const m1 = -3.5 + (rng.next() - 0.5) * 2.4; // ≈ [−4.7, −2.3]
      const m2 = 3.0 + (rng.next() - 0.5) * 2.4; //  ≈ [ 1.8,  4.2]
      const s1 = 0.55 + rng.next() * 0.7; //  ≈ [0.55, 1.25]
      const s2 = 0.7 + rng.next() * 0.8; //   ≈ [0.70, 1.50]
      const w1 = 0.35 + rng.next() * 0.3; //  ≈ [0.35, 0.65]
      bumps = [
        [w1, m1, s1],
        [1 - w1, m2, s2],
      ];
    }
    fillTarget();
    [fwdOpt, revOpt] = solveOptima();
  }

  /** Fill `p` from the current bumps and normalize so Σ p·dx = 1 (notebook). */
  function fillTarget(): void {
    for (let i = 0; i < G; i++) {
      let v = 0;
      for (const [w, mu, sd] of bumps) v += w * gauss(x[i], mu, sd);
      p[i] = v;
    }
    normalize(p);
    pMax = 0;
    for (let i = 0; i < G; i++) if (p[i] > pMax) pMax = p[i];
  }

  /** Fill `q` with a single normalized Gaussian N(mu, sd) on the x grid. */
  function fillQ(mu: number, sd: number): void {
    for (let i = 0; i < G; i++) q[i] = gauss(x[i], mu, sd);
    normalize(q);
  }

  function normalize(arr: Float64Array): void {
    let s = 0;
    for (let i = 0; i < G; i++) s += arr[i];
    const inv = s > 0 ? 1 / (s * dx) : 0;
    for (let i = 0; i < G; i++) arr[i] *= inv;
  }

  // forward KL(p‖q) and reverse KL(q‖p) by the notebook's grid sums.
  function forwardKl(): number {
    let s = 0;
    for (let i = 0; i < G; i++) s += p[i] * Math.log((p[i] + EPS) / (q[i] + EPS));
    return s * dx;
  }
  function reverseKl(): number {
    let s = 0;
    for (let i = 0; i < G; i++) s += q[i] * Math.log((q[i] + EPS) / (p[i] + EPS));
    return s * dx;
  }

  /** The notebook's 121×121 scan over μ∈[−5,5], σ∈[0.3,5]; returns both argmins. */
  function solveOptima(): [[number, number], [number, number]] {
    const N = 121;
    let bestF = Infinity;
    let bestR = Infinity;
    let argF: [number, number] = [0, 1];
    let argR: [number, number] = [0, 1];
    for (let a = 0; a < N; a++) {
      const mu = MU_LO + (MU_HI - MU_LO) * (a / (N - 1));
      for (let b = 0; b < N; b++) {
        const sd = SD_LO + (SD_HI - SD_LO) * (b / (N - 1));
        fillQ(mu, sd);
        let f = 0;
        let r = 0;
        for (let i = 0; i < G; i++) {
          const lp = p[i];
          const lq = q[i];
          f += lp * Math.log((lp + EPS) / (lq + EPS));
          r += lq * Math.log((lq + EPS) / (lp + EPS));
        }
        f *= dx;
        r *= dx;
        if (f < bestF) {
          bestF = f;
          argF = [mu, sd];
        }
        if (r < bestR) {
          bestR = r;
          argR = [mu, sd];
        }
      }
    }
    return [argF, argR];
  }

  buildTarget();

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the two divergences, live; the active one drives the gold curve
  const overlay = overlayLayer("tl");
  const eqFwd = liveEquation("D_{\\mathrm{KL}}(p\\,\\|\\,q)=", theme, theme.green);
  const eqRev = liveEquation("D_{\\mathrm{KL}}(q\\,\\|\\,p)=", theme, theme.gold);
  overlay.append(eqFwd.node, eqRev.node);
  panel.append(overlay);

  // top-right: the current candidate q(μ, σ)
  const qOverlay = overlayLayer("tr");
  const eqQ = staticEquation("", theme, theme.cream);
  qOverlay.append(eqQ);
  panel.append(qOverlay);

  const rc = responsiveCanvas(panel, 1.78, () => render());

  // --- controls -------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:8px 18px;align-items:center;",
  });
  const muRow = sliderRow(theme, "mean  μ", MU_LO, MU_HI, options.mu ?? 0, 0.1, "candidate mean mu");
  const sdRow = sliderRow(theme, "width  σ", SD_LO, SD_HI, options.sigma ?? 2.0, 0.05, "candidate width sigma");
  controls.append(muRow.row, sdRow.row);
  root.append(controls);

  const buttons = el("div", { style: "display:flex;align-items:center;gap:10px;flex-wrap:wrap;" });
  const fwdBtn = el("button", { type: "button", style: btnStyle(theme) });
  fwdBtn.innerHTML = "&#8594;&nbsp; fit forward KL";
  const revBtn = el("button", { type: "button", style: btnStyle(theme) });
  revBtn.innerHTML = "&#8594;&nbsp; fit reverse KL";
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed target";
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag q  ·  ←→ μ  ·  ↑↓ σ";
  buttons.append(fwdBtn, revBtn, reseedBtn, hint);
  root.append(buttons);

  target.append(root);

  // --- state: μ and σ each ease toward their target (two trackers) ----------
  let mu = clamp(options.mu ?? 0, MU_LO, MU_HI);
  let sigma = clamp(options.sigma ?? 2.0, SD_LO, SD_HI);
  // Which divergence the gold trace highlights — set by the last snap pressed.
  let active: "fwd" | "rev" = "fwd";
  let destroyed = false;

  const muTracker = valueTracker(mu, (v) => {
    mu = clamp(v, MU_LO, MU_HI);
    render();
  });
  const sdTracker = valueTracker(sigma, (v) => {
    sigma = clamp(v, SD_LO, SD_HI);
    render();
  });

  // --- render ---------------------------------------------------------------
  function render(): void {
    fillQ(mu, sigma);
    const fwd = forwardKl();
    const rev = reverseKl();
    draw();
    eqFwd.set(fwd.toFixed(3));
    eqRev.set(rev.toFixed(3));
    eqQ.innerHTML = staticEquation(
      `q=\\mathcal{N}(\\mu,\\sigma^2),\\ \\ \\mu=${mu.toFixed(2)},\\ \\sigma=${sigma.toFixed(2)}`,
      theme,
      theme.cream,
    ).innerHTML;
    if (document.activeElement !== muRow.input) muRow.set(mu);
    if (document.activeElement !== sdRow.input) sdRow.set(sigma);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 20;
    const mr = 16;
    const mt = 16;
    const mb = 30;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // shared vertical scale: tall enough for the target, the reverse-optimal
    // spike (gold ghost) and the live q, but capped at ~3× the target peak so a
    // very narrow q doesn't crush the whole plot when the reader drags σ small.
    const revPeak = gauss(revOpt[0], revOpt[0], revOpt[1]);
    let qMax = 0;
    for (let i = 0; i < G; i++) if (q[i] > qMax) qMax = q[i];
    const ceiling = pMax * 3.4;
    let ymax = Math.max(pMax, revPeak, Math.min(qMax, ceiling)) * 1.1;

    const X = (xv: number) => ml + ((clamp(xv, X_LO, X_HI) - X_LO) / (X_HI - X_LO)) * plotW;
    const Y = (d: number) => mt + plotH - (clamp(d, 0, ymax) / ymax) * plotH;

    // faint vertical grid every 2.5 units (the cyan plane, receding)
    for (let g = X_LO; g <= X_HI + 1e-6; g += 2.5) {
      const edge = g === X_LO || g === X_HI;
      strokeLine(ctx, [X(g), mt], [X(g), mt + plotH], theme.grid, 1, edge ? 0.28 : 0.16);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    const toScreen = (arr: Float64Array): Pt[] => {
      const out: Pt[] = new Array(G);
      for (let i = 0; i < G; i++) out[i] = [X(x[i]), Y(arr[i])];
      return out;
    };

    // target p (bright blue, filled, faintly glowing) — the fixed data object
    const pPts = toScreen(p);
    const fillPts: Pt[] = [[X(X_LO), Y(0)], ...pPts, [X(X_HI), Y(0)]];
    polyline(ctx, fillPts, { color: "transparent", width: 0, alpha: 1, closed: true, fill: hexA(theme.blue, 0.16) });
    glow(ctx, theme.blue, 5, () => polyline(ctx, pPts, { color: theme.blue, width: 2.2, alpha: 1 }));

    // the two optimal q's, faint, as ghosts of where each notion of "closest"
    // wants to land: forward (green, wide) and reverse (gold, narrow).
    drawGhost(ctx, X, Y, fwdOpt, theme.green, [4, 5]);
    drawGhost(ctx, X, Y, revOpt, theme.gold, [4, 5]);

    // the movable candidate q (cream, solid, glowing) — the variable
    const qPts = toScreen(q);
    glow(ctx, theme.cream, 6, () => polyline(ctx, qPts, { color: theme.cream, width: 2.4, alpha: 1 }));

    // the pointwise log-ratio penalty of the ACTIVE divergence, shaded under
    // the axis: red where the active integrand is positive (q under-covers /
    // over-reaches the target), the running cost of the current fit.
    drawPenalty(ctx, X, mt, plotH);

    // peak marker on q (the movable mode)
    let qPeakI = 0;
    for (let i = 1; i < G; i++) if (q[i] > q[qPeakI]) qPeakI = i;
    glow(ctx, theme.cream, 7, () => disc(ctx, [X(x[qPeakI]), Y(q[qPeakI])], 3.6, theme.cream, 1));
    mathLabel(ctx, "q", [X(mu) + (mu < 8 ? 7 : -7), Y(q[qPeakI]) - 6], {
      color: theme.cream,
      size: 16,
      align: mu < 8 ? "left" : "right",
      glow: 5,
    });

    // x ticks (roman numerals)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = X_LO; g <= X_HI + 1e-6; g += 2.5) ctx.fillText(g.toFixed(1), X(g), mt + plotH + 5);
    mathLabel(ctx, "x", [W - mr - 2, mt + plotH + 20], { color: theme.tick, size: 15, align: "right" });

    // legend (top-center-ish so it never collides with the overlays)
    drawLegend(ctx, W, mt);
  }

  /** A faint dashed ghost of an optimal single Gaussian N(mu, sd). */
  function drawGhost(
    ctx: CanvasRenderingContext2D,
    X: (xv: number) => number,
    Y: (d: number) => number,
    opt: [number, number],
    color: string,
    dash: number[],
  ): void {
    const [omu, osd] = opt;
    const pts: Pt[] = new Array(G);
    for (let i = 0; i < G; i++) pts[i] = [X(x[i]), Y(gauss(x[i], omu, osd))];
    polyline(ctx, pts, { color, width: 1.4, alpha: 0.5, dash });
  }

  /**
   * Shade the pointwise integrand of the active divergence beneath the axis:
   * forward uses p·log(p/q) (penalizes q missing target mass — drives covering),
   * reverse uses q·log(q/p) (penalizes q on target valleys — drives seeking).
   * Drawn in RED for the cost being paid, scaled to a thin strip under the plot.
   */
  function drawPenalty(
    ctx: CanvasRenderingContext2D,
    X: (xv: number) => number,
    mt: number,
    plotH: number,
  ): void {
    const integ = new Float64Array(G);
    let imax = 1e-9;
    for (let i = 0; i < G; i++) {
      const lp = p[i];
      const lq = q[i];
      const v =
        active === "fwd"
          ? lp * Math.log((lp + EPS) / (lq + EPS))
          : lq * Math.log((lq + EPS) / (lp + EPS));
      const pos = v > 0 ? v : 0; // only the positive cost reads as "penalty"
      integ[i] = pos;
      if (pos > imax) imax = pos;
    }
    const base = mt + plotH;
    const strip = Math.min(46, plotH * 0.3);
    const yb = (v: number) => base - (clamp(v, 0, imax) / imax) * strip;
    const pts: Pt[] = new Array(G + 2);
    pts[0] = [X(x[0]), base];
    for (let i = 0; i < G; i++) pts[i + 1] = [X(x[i]), yb(integ[i])];
    pts[G + 1] = [X(x[G - 1]), base];
    polyline(ctx, pts, { color: theme.red, width: 1, alpha: 0.7, closed: true, fill: hexA(theme.red, 0.22) });
  }

  function drawLegend(ctx: CanvasRenderingContext2D, W: number, top: number): void {
    const rows: Array<[string, string, number[]]> = [
      ["target p (bimodal)", theme.blue, []],
      ["forward fit: mode-covering", theme.green, [4, 5]],
      ["reverse fit: mode-seeking", theme.gold, [4, 5]],
      ["your q", theme.cream, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    const lx = W * 0.5 - 60;
    let yy = top + 12;
    for (const [name, color, dash] of rows) {
      ctx.strokeStyle = color;
      ctx.lineWidth = name === "your q" || name.startsWith("target") ? 2.2 : 1.6;
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

  // --- interaction ----------------------------------------------------------
  function setMu(v: number, animate = false): void {
    const t = clamp(v, MU_LO, MU_HI);
    if (animate) muTracker.set(t, true);
    else muTracker.jump(t);
  }
  function setSigma(v: number, animate = false): void {
    const t = clamp(v, SD_LO, SD_HI);
    if (animate) sdTracker.set(t, true);
    else sdTracker.jump(t);
  }

  function snapForward(): void {
    active = "fwd";
    setMu(fwdOpt[0], true);
    setSigma(fwdOpt[1], true);
  }
  function snapReverse(): void {
    active = "rev";
    setMu(revOpt[0], true);
    setSigma(revOpt[1], true);
  }

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    buildTarget();
    render();
  }

  muRow.input.addEventListener("input", () => setMu(Number(muRow.input.value), false));
  sdRow.input.addEventListener("input", () => setSigma(Number(sdRow.input.value), false));
  fwdBtn.addEventListener("click", snapForward);
  revBtn.addEventListener("click", snapReverse);
  reseedBtn.addEventListener("click", reseed);

  // Drag the canvas: x → μ, y → σ (down = wider). Direct, real-time.
  const stopDrag = draggable(
    rc.canvas,
    (px, py) => {
      const ml = 20;
      const mr = 16;
      const mt = 16;
      const mb = 30;
      const plotW = rc.width - ml - mr;
      const plotH = rc.height - mt - mb;
      const xv = X_LO + clamp((px - ml) / plotW, 0, 1) * (X_HI - X_LO);
      muTracker.jump(clamp(xv, MU_LO, MU_HI));
      const f = clamp((py - mt) / plotH, 0, 1);
      sdTracker.jump(SD_LO + f * (SD_HI - SD_LO));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Forward versus reverse KL: a fixed bimodal target density p and a single movable Gaussian q. Drag q to change its mean and width; the forward KL of p to q and the reverse KL of q to p update live. The forward optimum is wide and covers both modes; the reverse optimum is narrow and locks onto one mode.",
  );

  const onKey = (e: KeyboardEvent) => {
    const muStep = e.shiftKey ? 0.5 : 0.1;
    const sdStep = e.shiftKey ? 0.25 : 0.05;
    if (e.key === "ArrowLeft") setMu(mu - muStep, true);
    else if (e.key === "ArrowRight") setMu(mu + muStep, true);
    else if (e.key === "ArrowUp") setSigma(sigma + sdStep, true);
    else if (e.key === "ArrowDown") setSigma(sigma - sdStep, true);
    else if (e.key === "Home") snapForward();
    else if (e.key === "End") snapReverse();
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the forward optimum immediately, no glide. jump()
  // syncs both the displayed and target tracker values so a later snap eases
  // from here, not from the constructed initial.
  if (prefersReducedMotion()) {
    active = "fwd";
    muTracker.jump(clamp(fwdOpt[0], MU_LO, MU_HI));
    sdTracker.jump(clamp(fwdOpt[1], SD_LO, SD_HI));
  }

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setMu,
    setSigma,
    snapForward,
    snapReverse,
    reseed,
    destroy() {
      destroyed = true;
      muTracker.stop();
      sdTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- math helpers -----------------------------------------------------------

/** Unnormalized-then-normalized Gaussian density, as the notebook's `gauss`. */
function gauss(xv: number, mu: number, sd: number): number {
  const z = (xv - mu) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}

// --- small UI builder -------------------------------------------------------
interface Row {
  row: HTMLElement;
  input: HTMLInputElement;
  set(v: number): void;
}

function sliderRow(theme: Theme, label: string, min: number, max: number, value: number, step: number, aria: string): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:6.5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(step),
    value: String(value),
    style: "flex:1;min-width:90px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:3.5ch;text-align:right;` });
  out.textContent = Number(value).toFixed(2);
  input.addEventListener("input", () => (out.textContent = Number(input.value).toFixed(2)));
  row.append(lab, input, out);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
      out.textContent = v.toFixed(2);
    },
  };
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
