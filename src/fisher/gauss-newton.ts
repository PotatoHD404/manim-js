import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import type { Mat, Vec } from "../core/math/linalg";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface GaussNewtonOptions {
  /** Number of (x,y) regression samples (the notebook uses 200). */
  points?: number;
  /** RNG seed for the Monte-Carlo data draw (the notebook's rng is default_rng(2)). */
  seed?: number;
  /** True parameters (a,b) of f(x)=a·exp(−b·x²). The post uses (2.0, 0.6). */
  truth?: [number, number];
  /** Noise standard deviation σ. Default 0.2 (the notebook). */
  sigma?: number;
  /** Optimization start θ₀ = (a,b). Default (0.5, 0.2) — the post's poor start. */
  start?: [number, number];
  /** Vanilla-gradient learning rate. */
  lr?: number;
  /** Gauss-Newton / Fisher damping λ added to the diagonal. Default 1e-3 (the post). */
  damping?: number;
  /** Number of iterations to unroll for each solver. */
  steps?: number;
  /** Initial iteration the slider sits on. */
  iter?: number;
  theme?: Partial<Theme>;
}

export interface GaussNewtonApi {
  /** Scrub to an iteration index (eased when animate). */
  setIter(k: number, animate?: boolean): void;
  /** Play the descent from the start (eased glide through the iterations). */
  play(): void;
  /** Draw a fresh Monte-Carlo data set from the next seed and re-solve. */
  reseed(): void;
  destroy(): void;
}

/**
 * Scene — "the Fisher matrix is a metric". Nonlinear least squares for
 * f(x;a,b)=a·exp(−b·x²): the loss surface L(a,b)=½Σ(yᵢ−f)²/σ² is a long curved
 * valley. From one poor start θ₀=(0.5,0.2) two solvers race to the optimum. The
 * cream **vanilla gradient** path θ←θ−α∇L crawls and zig-zags — the gradient
 * ignores the loss's anisotropy. The gold **natural / Gauss-Newton** path
 * θ←θ−(F+λI)⁻¹∇L preconditions by the Fisher matrix F=σ⁻²Σ∇fᵢ∇fᵢᵀ (the
 * Gauss-Newton matrix, which for this NLL loss *is* the Fisher) and beelines to
 * the bottom in a handful of steps. An inset traces the post's own diagnostic:
 * the relative gap ‖H−F‖/‖H‖ collapsing as the residuals vanish and ∇²L→F.
 *
 * Everything is the notebook's exact computation: the same f, residual gradient
 * g=−σ⁻²Σ∇fᵢrᵢ, Gauss-Newton matrix GN, and damped step (GN+λI)⁻¹g, solved live
 * in the browser. A play button eases the iteration cursor through the unrolled
 * trajectories; the slider scrubs it; a re-seed control redraws the random data.
 */
export function createFisherGaussNewton(
  target: HTMLElement,
  options: GaussNewtonOptions = {},
): GaussNewtonApi {
  const theme = withTheme(options.theme);

  // --- fixed model constants (the notebook's nonlinear-regression cell) -----
  const m = options.points ?? 200;
  const truth: [number, number] = options.truth ?? [2.0, 0.6];
  const sigma = options.sigma ?? 0.2;
  const start: Vec = (options.start ?? [0.5, 0.2]).slice();
  const lr = options.lr ?? 6e-5; // vanilla step; tuned so the badly-scaled GD path stays on-panel
  const damping = options.damping ?? 1e-3;
  const STEPS = options.steps ?? 15;
  let seed = (options.seed ?? 2) >>> 0 || 1;
  let destroyed = false;

  // --- the data draw: xN ~ U(−3,3), yN = a·exp(−b·x²) + N(0,σ²) -------------
  let xs = new Float64Array(m);
  let ys = new Float64Array(m);

  function drawData(): void {
    const rng = new Rng(seed);
    xs = new Float64Array(m);
    ys = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const x = -3 + 6 * rng.next(); // uniform(−3, 3)
      xs[i] = x;
      ys[i] = truth[0] * Math.exp(-truth[1] * x * x) + sigma * rng.gauss();
    }
  }

  // f(θ,x) = a·exp(−b·x²); ∇f = [e, −a·x²·e] with e = exp(−b·x²).
  const fAt = (th: Vec, x: number): number => th[0] * Math.exp(-th[1] * x * x);

  // L(θ) = ½ Σ (yᵢ − f)² / σ² — the notebook's `loss`.
  function lossAt(th: Vec): number {
    let s = 0;
    for (let i = 0; i < m; i++) {
      const r = ys[i] - fAt(th, xs[i]);
      s += r * r;
    }
    return (0.5 * s) / (sigma * sigma);
  }

  // Residual gradient g = −σ⁻² Σ ∇fᵢ rᵢ and Gauss-Newton matrix GN = σ⁻² Σ ∇fᵢ∇fᵢᵀ
  // (= the Fisher), plus the relative gap ‖H−GN‖/‖H‖ via a numerical Hessian of L.
  function localTerms(th: Vec): { g: Vec; gn: Mat; gap: number } {
    const inv = 1 / (sigma * sigma);
    let g0 = 0;
    let g1 = 0;
    let h00 = 0;
    let h01 = 0;
    let h11 = 0;
    for (let i = 0; i < m; i++) {
      const x2 = xs[i] * xs[i];
      const e = Math.exp(-th[1] * x2);
      const jf0 = e; // ∂f/∂a
      const jf1 = -th[0] * x2 * e; // ∂f/∂b
      const r = ys[i] - fAt(th, xs[i]);
      g0 -= jf0 * r;
      g1 -= jf1 * r;
      h00 += jf0 * jf0;
      h01 += jf0 * jf1;
      h11 += jf1 * jf1;
    }
    const g: Vec = [g0 * inv, g1 * inv];
    const gn: Mat = [
      [h00 * inv, h01 * inv],
      [h01 * inv, h11 * inv],
    ];
    return { g, gn, gap: hessianGap(th, gn) };
  }

  // The post's diagnostic: ‖H_full − GN‖ / ‖H_full‖, H_full a finite-difference
  // Hessian of L (same central 4-point stencil as the notebook's num_hess).
  function hessianGap(th: Vec, gn: Mat): number {
    const h = 1e-4;
    const H: Mat = [
      [0, 0],
      [0, 0],
    ];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const tpp = th.slice();
        tpp[i] += h;
        tpp[j] += h;
        const tpm = th.slice();
        tpm[i] += h;
        tpm[j] -= h;
        const tmp = th.slice();
        tmp[i] -= h;
        tmp[j] += h;
        const tmm = th.slice();
        tmm[i] -= h;
        tmm[j] -= h;
        H[i][j] = (lossAt(tpp) - lossAt(tpm) - lossAt(tmp) + lossAt(tmm)) / (4 * h * h);
      }
    }
    const dn = frob([
      [H[0][0] - gn[0][0], H[0][1] - gn[0][1]],
      [H[1][0] - gn[1][0], H[1][1] - gn[1][1]],
    ]);
    const fn = frob(H);
    return fn > 0 ? dn / fn : 0;
  }

  // --- the two unrolled trajectories from the same start --------------------
  interface Traj {
    theta: Vec[]; // θ at each iteration (length STEPS+1)
    loss: number[]; // L(θ) at each iteration
    gap: number[]; // ‖H−F‖/‖H‖ at each iteration
  }
  let vanilla: Traj = emptyTraj();
  let natural: Traj = emptyTraj();

  function emptyTraj(): Traj {
    return { theta: [], loss: [], gap: [] };
  }

  // Vanilla:  θ ← θ − α g.   Natural / Gauss-Newton:  θ ← θ − (GN + λI)⁻¹ g.
  function solve(): void {
    vanilla = runSolver(false);
    natural = runSolver(true);
  }

  function runSolver(useFisher: boolean): Traj {
    const traj = emptyTraj();
    let th = start.slice();
    for (let k = 0; k <= STEPS; k++) {
      const { g, gn, gap } = localTerms(th);
      traj.theta.push(th.slice());
      traj.loss.push(lossAt(th));
      traj.gap.push(gap);
      if (k === STEPS) break;
      if (useFisher) {
        // damped Gauss-Newton (= natural-gradient) step: (GN + λI)⁻¹ g
        const step = solve2(
          [
            [gn[0][0] + damping, gn[0][1]],
            [gn[1][0], gn[1][1] + damping],
          ],
          g,
        );
        th = [th[0] - step[0], th[1] - step[1]];
      } else {
        th = [th[0] - lr * g[0], th[1] - lr * g[1]];
      }
    }
    return traj;
  }

  drawData();
  solve();

  // (a,b) window: enclose both the start and the truth with a margin.
  const aLo = 0.2;
  const aHi = 2.55;
  const bLo = 0.05;
  const bHi = 0.95;

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the live loss for each path, read at the current iteration
  const overlay = overlayLayer("tl");
  const eqNat = liveEquation("L_{\\text{nat}}=", theme, theme.gold);
  const eqVan = liveEquation("L_{\\text{GD}}=", theme, theme.cream);
  const eqIter = liveEquation("k=", theme, theme.green);
  overlay.append(eqNat.node, eqVan.node, eqIter.node);
  panel.append(overlay);

  // top-right: the natural-gradient step, restated
  const idOverlay = overlayLayer("tr");
  idOverlay.append(
    staticEquation("\\Delta\\theta=-(F+\\lambda I)^{-1}\\nabla L", theme, theme.gold),
    staticEquation("F=\\tfrac{1}{\\sigma^2}\\textstyle\\sum_i\\nabla f_i\\,\\nabla f_i^{\\top}", theme, theme.muted),
  );
  panel.append(idOverlay);

  const rc = responsiveCanvas(panel, 1.42, () => render());

  // --- controls -------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "iteration  k";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: String(STEPS),
    step: "1",
    value: String(clamp(Math.round(options.iter ?? 0), 0, STEPS)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "Gauss-Newton iteration index",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "play · reseed · ←→";
  controls.append(playBtn, reseedBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the (continuous) iteration cursor --------------
  let iter = clamp(options.iter ?? 0, 0, STEPS);
  let playing = false;
  let playCancel: (() => void) | null = null;

  const iterTracker = valueTracker(iter, (v) => {
    iter = clamp(v, 0, STEPS);
    render();
  });

  function stopPlay(): void {
    if (!playing) return;
    playCancel?.();
    playCancel = null;
    playing = false;
    updatePlayBtn();
  }

  // map (a,b) → screen, sharing equal margins so contours read undistorted
  function frame(): { ml: number; mr: number; mt: number; mb: number } {
    return { ml: 46, mr: 16, mt: 16, mb: 34 };
  }

  // --- render ---------------------------------------------------------------
  function render(): void {
    draw();
    eqNat.set(interpLoss(natural, iter).toFixed(2));
    eqVan.set(interpLoss(vanilla, iter).toFixed(2));
    eqIter.set(iter.toFixed(1));
    if (document.activeElement !== slider) slider.value = String(Math.round(iter));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const { ml, mr, mt, mb } = frame();
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const X = (a: number) => ml + ((a - aLo) / (aHi - aLo)) * plotW;
    const Y = (b: number) => mt + plotH - ((b - bLo) / (bHi - bLo)) * plotH;

    // faint receding grid in BLUE_D at round (a,b) levels
    for (let a = 0.5; a <= aHi; a += 0.5) strokeLine(ctx, [X(a), mt], [X(a), mt + plotH], theme.grid, 1, 0.2);
    for (let b = 0.2; b <= bHi; b += 0.2) strokeLine(ctx, [ml, Y(b)], [W - mr, Y(b)], theme.grid, 1, 0.2);

    // the loss surface: filled+stroked contour bands of L(a,b) (blue, the data)
    drawContours(ctx, X, Y);

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // the optimum (the answer): a gold star at the true (a,b)
    const opt: Pt = [X(truth[0]), Y(truth[1])];
    glow(ctx, theme.gold, 8, () => disc(ctx, opt, 4.2, theme.gold, 1));
    strokeLine(ctx, [opt[0] - 7, opt[1]], [opt[0] + 7, opt[1]], theme.gold, 1, 0.7);
    strokeLine(ctx, [opt[0], opt[1] - 7], [opt[0], opt[1] + 7], theme.gold, 1, 0.7);

    // the two trajectories, drawn up to the eased iteration cursor
    drawPath(ctx, X, Y, vanilla, theme.cream, false);
    drawPath(ctx, X, Y, natural, theme.gold, true);

    // shared start marker (the movable variable's origin)
    const s0: Pt = [X(start[0]), Y(start[1])];
    disc(ctx, s0, 3.4, theme.cream, 0.9);
    mathLabel(ctx, "\\theta", [s0[0] - 8, s0[1] + 16], { color: theme.cream, size: 14, sub: "0", align: "right" });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "a", [X(aHi) - 4, mt + plotH + 22], { color: theme.tick, size: 16, align: "right" });
    mathLabel(ctx, "b", [ml - 32, Y(bHi) + 12], { color: theme.tick, size: 16 });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let a = 0.5; a <= aHi; a += 0.5) ctx.fillText(a.toFixed(1), X(a), mt + plotH + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let b = 0.2; b <= bHi; b += 0.2) ctx.fillText(b.toFixed(1), ml - 6, Y(b));

    // legend + the post's own gap diagnostic, as an inset trace
    drawLegend(ctx, W - mr, mt + 44);
    drawGapInset(ctx, W, H);
  }

  // Marching-squares contour bands of L on a grid — the loss landscape itself.
  function drawContours(ctx: CanvasRenderingContext2D, X: (a: number) => number, Y: (b: number) => number): void {
    const NA = 120;
    const NB = 90;
    const grid: number[][] = [];
    let gMin = Infinity;
    let gMax = -Infinity;
    for (let j = 0; j <= NB; j++) {
      const row: number[] = [];
      const b = bLo + ((bHi - bLo) * j) / NB;
      for (let i = 0; i <= NA; i++) {
        const a = aLo + ((aHi - aLo) * i) / NA;
        const v = Math.log(lossAt([a, b]) + 1); // log-compressed so the deep valley reads
        row.push(v);
        if (v < gMin) gMin = v;
        if (v > gMax) gMax = v;
      }
      grid.push(row);
    }
    const aOf = (i: number) => aLo + ((aHi - aLo) * i) / NA;
    const bOf = (j: number) => bLo + ((bHi - bLo) * j) / NB;

    const LEVELS = 11;
    for (let l = 1; l < LEVELS; l++) {
      const level = gMin + ((gMax - gMin) * l) / LEVELS;
      const alpha = 0.14 + 0.34 * (1 - l / LEVELS); // inner (low-loss) rings brighter
      const segs: [Pt, Pt][] = [];
      for (let j = 0; j < NB; j++) {
        for (let i = 0; i < NA; i++) {
          marchCell(segs, level, i, j, grid, aOf, bOf, X, Y);
        }
      }
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = theme.blue;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      for (const [p, q] of segs) {
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(q[0], q[1]);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function drawPath(
    ctx: CanvasRenderingContext2D,
    X: (a: number) => number,
    Y: (b: number) => number,
    traj: Traj,
    color: string,
    answer: boolean,
  ): void {
    const kf = iter;
    const full = Math.floor(kf);
    const frac = kf - full;
    const screen: Pt[] = [];
    for (let k = 0; k <= full && k < traj.theta.length; k++) {
      screen.push([X(traj.theta[k][0]), Y(traj.theta[k][1])]);
    }
    // interpolate the leading edge between step `full` and `full+1`
    let head: Pt | null = null;
    if (full < STEPS && frac > 0) {
      const p0 = traj.theta[full];
      const p1 = traj.theta[full + 1];
      const a = p0[0] + (p1[0] - p0[0]) * frac;
      const b = p0[1] + (p1[1] - p0[1]) * frac;
      head = [X(a), Y(b)];
      screen.push(head);
    } else {
      head = screen[screen.length - 1] ?? null;
    }

    if (screen.length > 1) {
      glow(ctx, color, answer ? 6 : 3, () => {
        polyline(ctx, screen, { color, width: answer ? 2.6 : 2, alpha: answer ? 1 : 0.85, dash: answer ? undefined : [5, 5] });
      });
    }
    // the discrete step waypoints already reached
    for (let k = 0; k <= full && k < traj.theta.length; k++) {
      disc(ctx, [X(traj.theta[k][0]), Y(traj.theta[k][1])], answer ? 2.4 : 2.0, color, answer ? 0.9 : 0.7);
    }
    // the moving head — the iterate the user is scrubbing
    if (head) {
      glow(ctx, color, answer ? 9 : 6, () => disc(ctx, head as Pt, answer ? 4 : 3.4, color, 1));
    }
  }

  // The original PNG's quantity: ‖H − Fisher‖/‖H‖ for the natural path, log-y,
  // collapsing toward the noise floor as the residuals vanish (∇²L → F).
  function drawGapInset(ctx: CanvasRenderingContext2D, W: number, H: number): void {
    const iw = Math.min(168, W * 0.36);
    const ih = Math.min(96, iw * 0.6);
    const x0 = W - 16 - iw;
    const y0 = H - 16 - ih;
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = "#101010";
    ctx.fillRect(x0 - 6, y0 - 18, iw + 12, ih + 30);
    ctx.globalAlpha = 1;

    const gLo = Math.log10(Math.max(1e-3, Math.min(...natural.gap.filter((g) => g > 0), 1e-2) * 0.5));
    const gHi = Math.log10(Math.max(...natural.gap, 1e-2) * 1.3);
    const gx = (k: number) => x0 + (k / STEPS) * iw;
    const gy = (g: number) => y0 + ih - ((clamp(Math.log10(Math.max(g, 1e-9)), gLo, gHi) - gLo) / (gHi - gLo)) * ih;

    strokeLine(ctx, [x0, y0 + ih], [x0 + iw, y0 + ih], theme.axis, 1, 0.7);
    strokeLine(ctx, [x0, y0], [x0, y0 + ih], theme.axis, 1, 0.7);

    const gapPts: Pt[] = natural.gap.map((g, k) => [gx(k), gy(g)]);
    polyline(ctx, gapPts, { color: theme.green, width: 1.8, alpha: 0.4 });
    const upto = gapPts.filter((_, k) => k <= iter);
    if (upto.length > 1) glow(ctx, theme.green, 4, () => polyline(ctx, upto, { color: theme.green, width: 2.2, alpha: 1 }));
    disc(ctx, [gx(iter), gy(interpGap(natural, iter))], 2.8, theme.green, 1);

    ctx.fillStyle = theme.green;
    ctx.font = `10px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("‖H − F‖ / ‖H‖", x0, y0 - 6);
    ctx.restore();
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: [string, string, number[]][] = [
      ["natural  (F⁻¹∇L)", theme.gold, []],
      ["vanilla  (α∇L)", theme.cream, [5, 5]],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 22;
    let yy = top;
    for (const [name, color, dash] of rows) {
      const lx = right - 130;
      ctx.strokeStyle = color;
      ctx.lineWidth = dash.length === 0 ? 2.6 : 2;
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

  // --- interaction ----------------------------------------------------------
  function setIter(k: number, animate = false): void {
    const t = clamp(k, 0, STEPS);
    if (animate) iterTracker.set(t, true);
    else iterTracker.jump(t);
  }

  function play(): void {
    if (playing) {
      stopPlay();
      return;
    }
    if (prefersReducedMotion()) {
      setIter(STEPS, false);
      return;
    }
    // Restart from the top if we're already at (or near) the end.
    if (iter >= STEPS - 1e-3) iterTracker.jump(0);
    playing = true;
    updatePlayBtn();
    let k = iter;
    playCancel = ticker((dt) => {
      k += dt * 2.2; // ~2.2 iterations / second
      if (k >= STEPS) {
        playing = false;
        playCancel = null;
        updatePlayBtn();
        iterTracker.jump(STEPS);
        return false;
      }
      iterTracker.jump(k);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = playing ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; descend";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    drawData();
    solve();
    render();
  }
  reseedBtn.addEventListener("click", reseed);

  slider.addEventListener("input", () => {
    stopPlay();
    iterTracker.jump(Number(slider.value));
  });

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Natural gradient versus vanilla gradient descent on a nonlinear least-squares loss surface. From the same poor start, the vanilla gradient path crawls and zig-zags down a curved valley, while the Fisher-preconditioned Gauss-Newton path beelines to the optimum in a few steps. An inset shows the relative gap between the full Hessian and the Fisher matrix collapsing as the fit converges. Press play to animate the descent, or scrub the iteration slider.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const step = e.shiftKey ? 3 : 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setIter(Math.round(iter) - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setIter(Math.round(iter) + step, true);
    else if (e.key === "Home") setIter(0, true);
    else if (e.key === "End") setIter(STEPS, true);
    else return;
    e.preventDefault();
    stopPlay();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land at the converged end with no glide.
  if (prefersReducedMotion()) iter = STEPS;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setIter,
    play,
    reseed,
    destroy() {
      destroyed = true;
      iterTracker.stop();
      playCancel?.();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- trajectory readout helpers --------------------------------------------

/** Linearly interpolate a per-iteration scalar at a continuous iteration k. */
function interpAt(arr: number[], k: number): number {
  const n = arr.length;
  if (n === 0) return 0;
  const x = clamp(k, 0, n - 1);
  const lo = Math.floor(x);
  const hi = Math.min(lo + 1, n - 1);
  return arr[lo] * (1 - (x - lo)) + arr[hi] * (x - lo);
}

function interpLoss(traj: { loss: number[] }, k: number): number {
  return interpAt(traj.loss, k);
}
function interpGap(traj: { gap: number[] }, k: number): number {
  return interpAt(traj.gap, k);
}

// --- 2×2 linear algebra -----------------------------------------------------

/** Solve A x = b for a 2×2 system (Cramer's rule); the damped GN normal equations. */
function solve2(A: Mat, b: Vec): Vec {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (Math.abs(det) < 1e-18) return [0, 0];
  return [(b[0] * A[1][1] - b[1] * A[0][1]) / det, (A[0][0] * b[1] - A[1][0] * b[0]) / det];
}

/** Frobenius norm of a 2×2 matrix. */
function frob(M: Mat): number {
  return Math.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1] + M[1][0] * M[1][0] + M[1][1] * M[1][1]);
}

// --- marching squares (one cell, one level) --------------------------------

/** Emit the contour segment(s) of `level` crossing grid cell (i, j). */
function marchCell(
  out: [Pt, Pt][],
  level: number,
  i: number,
  j: number,
  grid: number[][],
  aOf: (i: number) => number,
  bOf: (j: number) => number,
  X: (a: number) => number,
  Y: (b: number) => number,
): void {
  const v00 = grid[j][i];
  const v10 = grid[j][i + 1];
  const v11 = grid[j + 1][i + 1];
  const v01 = grid[j + 1][i];
  let code = 0;
  if (v00 > level) code |= 1;
  if (v10 > level) code |= 2;
  if (v11 > level) code |= 4;
  if (v01 > level) code |= 8;
  if (code === 0 || code === 15) return;

  const a0 = aOf(i);
  const a1 = aOf(i + 1);
  const b0 = bOf(j);
  const b1 = bOf(j + 1);
  const lerp = (lo: number, hi: number, t: number) => lo + (hi - lo) * t;
  // edge interpolation points (bottom, right, top, left)
  const bottom = (): Pt => [X(lerp(a0, a1, frac(v00, v10, level))), Y(b0)];
  const right = (): Pt => [X(a1), Y(lerp(b0, b1, frac(v10, v11, level)))];
  const top = (): Pt => [X(lerp(a0, a1, frac(v01, v11, level))), Y(b1)];
  const left = (): Pt => [X(a0), Y(lerp(b0, b1, frac(v00, v01, level)))];

  switch (code) {
    case 1:
    case 14:
      out.push([left(), bottom()]);
      break;
    case 2:
    case 13:
      out.push([bottom(), right()]);
      break;
    case 3:
    case 12:
      out.push([left(), right()]);
      break;
    case 4:
    case 11:
      out.push([right(), top()]);
      break;
    case 5:
      out.push([left(), top()]);
      out.push([bottom(), right()]);
      break;
    case 6:
    case 9:
      out.push([bottom(), top()]);
      break;
    case 7:
    case 8:
      out.push([left(), top()]);
      break;
    case 10:
      out.push([left(), bottom()]);
      out.push([right(), top()]);
      break;
    default:
      break;
  }
}

/** Where along an edge value crosses `level` (0..1), guarded against ties. */
function frac(v0: number, v1: number, level: number): number {
  const d = v1 - v0;
  if (Math.abs(d) < 1e-12) return 0.5;
  return clamp((level - v0) / d, 0, 1);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
