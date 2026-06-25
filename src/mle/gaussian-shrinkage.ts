import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { tex } from "../core/katex";
import { type Theme, withTheme } from "../core/theme";

export interface GaussianShrinkageOptions {
  /** Prior mean μ₀ — the belief before any data. */
  priorMean?: number;
  /** Prior variance τ² — how tightly the prior is held. */
  priorVar?: number;
  /** Sample mean x̄ — the maximum-likelihood estimate of μ. */
  sampleMean?: number;
  /** Known observation variance σ². */
  sigma2?: number;
  /** Number of observations n. */
  n?: number;
  theme?: Partial<Theme>;
}

export interface GaussianShrinkageApi {
  setN(n: number, animate?: boolean): void;
  setSampleMean(xbar: number): void;
  setPriorMean(mu0: number): void;
  setPriorVar(tau2: number): void;
  play(): void;
  destroy(): void;
}

/**
 * Scene — "the MAP estimate shrinks the MLE toward the prior". For
 * `xᵢ ~ N(μ, σ²)` with known `σ²` and prior `μ ~ N(μ₀, τ²)`, the posterior over
 * the mean is Gaussian and its mode equals its mean,
 *
 *   μ̂_MAP = w·x̄ + (1−w)·μ₀,   w = (n/σ²) / (n/σ² + 1/τ²).
 *
 * The board lives in μ-space: the cream prior density sits at μ₀, the blue
 * likelihood-of-the-mean `N(x̄, σ²/n)` sits at the MLE x̄, and the gold posterior
 * `N(μ̂_MAP, 1/precision)` sits exactly at the precision-weighted blend of the
 * two. Drag x̄, slide the prior, or grow n: as n → ∞ (or the prior loosens) the
 * weight w → 1 and the answer slides onto the MLE; shrink n (or tighten the
 * prior) and it is pulled back toward μ₀. A small inset traces the post's own
 * curve, w against n, with the current n marked.
 */
export function createGaussianShrinkage(
  target: HTMLElement,
  options: GaussianShrinkageOptions = {},
): GaussianShrinkageApi {
  const theme = withTheme(options.theme);

  const sigma2 = clampPos(options.sigma2 ?? 1.0);
  let mu0 = options.priorMean ?? -1.4; // prior mean
  let tau2 = clampPos(options.priorVar ?? 1.0); // prior variance
  let xbar = options.sampleMean ?? 1.8; // sample mean = MLE
  let nDisp = clamp(options.n ?? 4, N_MIN, N_MAX); // displayed (eased) n

  // μ-axis window, fixed so curves never jump under the controls.
  const half = 4.6;

  // ---- the conjugate Normal-Normal math (the post's source of truth) -------
  const dataPrecision = (n: number) => n / sigma2; // n/σ²
  const priorPrecision = () => 1 / tau2; // 1/τ²
  const weight = (n: number) => {
    const dp = dataPrecision(n);
    return dp / (dp + priorPrecision()); // w
  };
  const mapEstimate = (n: number) => {
    const w = weight(n);
    return w * xbar + (1 - w) * mu0; // μ̂_MAP = w x̄ + (1−w) μ₀
  };
  const postVar = (n: number) => 1 / (dataPrecision(n) + priorPrecision()); // posterior variance
  const likVar = (n: number) => sigma2 / n; // variance of x̄ : σ²/n

  // Unit Gaussian density, used for all three curves.
  const gauss = (x: number, mean: number, variance: number) => {
    const v = Math.max(variance, 1e-6);
    return Math.exp(-((x - mean) * (x - mean)) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
  };

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the live MAP blend
  const overlay = overlayLayer("tl");
  const eqMap = liveEquation("\\hat\\mu_{\\text{MAP}}=w\\,\\bar x+(1-w)\\,\\mu_0\\;=", theme, theme.gold);
  const eqW = liveEquation("w=\\dfrac{n/\\sigma^2}{n/\\sigma^2+1/\\tau^2}\\;=", theme, theme.green);
  overlay.append(eqMap.node, eqW.node);
  mainHost.append(overlay);

  // top-right: the fixed model statement
  const modelOverlay = overlayLayer("tr");
  modelOverlay.append(
    staticEquation(`x_i\\sim N(\\mu,\\sigma^2),\\ \\sigma^2=${sigma2.toFixed(2)}`, theme, theme.muted),
    staticEquation("\\mu\\sim N(\\mu_0,\\tau^2)", theme, theme.muted),
  );
  mainHost.append(modelOverlay);

  // --- controls -------------------------------------------------------------
  const controls = el("div", { style: "display:flex;flex-direction:column;gap:10px;" });

  const playBtn = el("button", { type: "button", style: btnStyle(theme) });

  const nSlider = makeSlider(String(N_MIN), String(N_MAX), "1", String(Math.round(nDisp)), "number of observations n");
  const xSlider = makeSlider(fmt(-half), fmt(half), "0.05", String(xbar), "sample mean x-bar (the MLE)");
  const muSlider = makeSlider(fmt(-half), fmt(half), "0.05", String(mu0), "prior mean mu-zero");
  const tauSlider = makeSlider("0.1", "4", "0.05", String(tau2), "prior variance tau-squared");

  const rowN = controlRow(theme, "n", nSlider);
  const rowX = controlRow(theme, "\\bar x", xSlider);
  const rowMu = controlRow(theme, "\\mu_0", muSlider);
  const rowTau = controlRow(theme, "\\tau^2", tauSlider);

  const nReadout = rowN.readout;
  const xReadout = rowX.readout;
  const muReadout = rowMu.readout;
  const tauReadout = rowTau.readout;

  const topBar = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag x̄ · grow n · ←→";
  topBar.append(playBtn, hint);

  const grid = el("div", {
    style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 22px;",
  });
  grid.append(rowN.node, rowX.node, rowMu.node, rowTau.node);

  controls.append(topBar, grid);
  root.append(controls);
  target.append(root);

  // --- state ----------------------------------------------------------------
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  const rc = responsiveCanvas(mainHost, 1.62, () => render());

  // The eased scalar for the scene is n: keyboard, sweep, and snaps all glide it,
  // and every density that depends on n (likelihood width, posterior, the blend)
  // animates together — the MAP marker visibly slides between x̄ and μ₀.
  const nTracker = valueTracker(nDisp, (v) => {
    nDisp = v;
    render();
  });

  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  // --- render ---------------------------------------------------------------
  function render(): void {
    draw();
    const nNow = nDisp;
    const w = weight(nNow);
    eqMap.set(mapEstimate(nNow).toFixed(2));
    eqW.set(w.toFixed(3));

    const nInt = Math.round(nNow);
    if (document.activeElement !== nSlider) nSlider.value = String(nInt);
    if (document.activeElement !== xSlider) xSlider.value = fmt(xbar);
    if (document.activeElement !== muSlider) muSlider.value = fmt(mu0);
    if (document.activeElement !== tauSlider) tauSlider.value = fmt(tau2);
    setReadout(nReadout, `n=${nInt}`);
    setReadout(xReadout, `\\bar x=${xbar.toFixed(2)}`);
    setReadout(muReadout, `\\mu_0=${mu0.toFixed(2)}`);
    setReadout(tauReadout, `\\tau^2=${tau2.toFixed(2)}`);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 18;
    const mr = 16;
    const mt = 92; // leave room for the overlays
    const mb = 30;
    const x = (mu: number) => ml + ((mu + half) / (2 * half)) * (W - ml - mr);
    const baseY = H - mb;

    const nNow = nDisp;
    const mapMu = mapEstimate(nNow);

    // Curve densities, sampled on a fine grid and scaled to a common pixel
    // height so all three are visible regardless of how peaked they are.
    const samples = 240;
    const xs = Array.from({ length: samples }, (_, i) => -half + (2 * half * i) / (samples - 1));
    const priorD = xs.map((mu) => gauss(mu, mu0, tau2));
    const likD = xs.map((mu) => gauss(mu, xbar, likVar(nNow)));
    const postD = xs.map((mu) => gauss(mu, mapMu, postVar(nNow)));
    const peak = Math.max(...priorD, ...likD, ...postD, 1e-6);
    const plotH = baseY - mt;
    const yOf = (d: number) => baseY - (d / peak) * plotH * 0.92;

    // baseline (the μ axis)
    strokeLine(ctx, [ml, baseY], [W - mr, baseY], theme.axis, 1.4, 1);

    // μ ticks
    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = -4; t <= 4; t++) {
      if (t === 0) continue;
      strokeLine(ctx, [x(t), baseY], [x(t), baseY + 4], theme.axis, 1, 0.6);
      ctx.fillText(String(t), x(t), baseY + 6);
    }

    const curve = (dens: number[], color: string, widthPx: number, fillA: number, strokeA: number) => {
      const top: Pt[] = dens.map((d, i) => [x(xs[i]), yOf(d)]);
      const filled: Pt[] = [[x(-half), baseY], ...top, [x(half), baseY]];
      glow(ctx, color, 4, () => {
        polyline(ctx, filled, { color, width: widthPx, alpha: strokeA, closed: true, fill: hexA(color, fillA) });
      });
    };

    // prior (cream — the belief), likelihood (blue — the data), posterior (gold — the answer)
    curve(priorD, theme.cream, 1.6, 0.06, 0.55);
    curve(likD, theme.blue, 1.8, 0.1, 0.78);
    curve(postD, theme.gold, 2.4, 0.14, 1);

    // label y just above a curve's peak, clamped under the overlays
    const labelY = (peakY: number) => Math.max(peakY - 9, mt - 4);

    // --- the three estimates as vertical markers --------------------------
    // μ₀ (prior mean)
    const priorPeakY = yOf(gauss(mu0, mu0, tau2));
    marker(ctx, x(mu0), baseY, priorPeakY, theme.cream, 0.7);
    mathLabel(ctx, "μ", [x(mu0) - 7, labelY(priorPeakY)], { color: theme.cream, size: 16, sub: "0", glow: 4 });

    // x̄ (MLE) — draggable
    const likPeakY = yOf(gauss(xbar, xbar, likVar(nNow)));
    marker(ctx, x(xbar), baseY, likPeakY, theme.blue, 0.85);
    mathLabel(ctx, "x̄", [x(xbar) - 5, labelY(likPeakY)], { color: theme.blue, size: 16, glow: 4 });

    // the shrinkage bracket: a bright segment from x̄ to μ̂_MAP showing the pull
    const yBr = baseY - 8;
    strokeLine(ctx, [x(xbar), yBr], [x(mapMu), yBr], theme.red, 2.2, 0.85);
    // little ticks at the ends of the bracket
    strokeLine(ctx, [x(xbar), yBr - 4], [x(xbar), yBr + 4], theme.red, 1.6, 0.7);
    strokeLine(ctx, [x(mapMu), yBr - 4], [x(mapMu), yBr + 4], theme.red, 1.6, 0.7);

    // μ̂_MAP (the answer) — the gold marker, glowing, the punchline
    const mapTopY = yOf(gauss(mapMu, mapMu, postVar(nNow)));
    glow(ctx, theme.gold, 8, () => {
      strokeLine(ctx, [x(mapMu), baseY], [x(mapMu), mapTopY], theme.gold, 2.6, 1);
    });
    glow(ctx, theme.gold, 8, () => disc(ctx, [x(mapMu), mapTopY], 4.2, theme.gold, 1));
    mathLabel(ctx, "μ̂", [x(mapMu) + 7, labelY(mapTopY)], { color: theme.gold, size: 16, sub: "MAP", glow: 6, align: "left" });

    // --- inset: w vs n (the post's actual figure) -------------------------
    drawInset(ctx, W, H);
  }

  function marker(
    ctx: CanvasRenderingContext2D,
    px: number,
    baseY: number,
    topY: number,
    color: string,
    alpha: number,
  ): void {
    polyline(ctx, [[px, baseY], [px, topY]], { color, width: 1.4, alpha, dash: [4, 5] });
    disc(ctx, [px, topY], 3, color, alpha);
  }

  function drawInset(ctx: CanvasRenderingContext2D, W: number, _H: number): void {
    // a compact w(n) curve pinned to the lower-right, matching the post's plot
    const iw = Math.min(176, W * 0.34);
    const ih = iw * 0.62;
    const ix = W - 16 - iw;
    const iy = 16 + 64; // sit just below the top-right model overlay
    const pad = 8;

    // frame
    ctx.fillStyle = hexA(theme.fg, 0.04);
    ctx.fillRect(ix, iy, iw, ih);
    strokeLine(ctx, [ix, iy + ih], [ix + iw, iy + ih], theme.axis, 1, 0.7);
    strokeLine(ctx, [ix, iy], [ix, iy + ih], theme.axis, 1, 0.7);

    const wx = (n: number) => ix + pad + ((n - N_MIN) / (N_MAX - N_MIN)) * (iw - 2 * pad);
    const wy = (w: number) => iy + ih - pad - clamp(w, 0, 1) * (ih - 2 * pad);

    // w = 1 asymptote
    polyline(ctx, [[ix + pad, wy(1)], [ix + iw - pad, wy(1)]], { color: theme.muted, width: 1, alpha: 0.6, dash: [3, 4] });

    // the curve w(n) for the current prior/σ²
    const pts: Pt[] = [];
    for (let n = N_MIN; n <= N_MAX; n++) pts.push([wx(n), wy(weight(n))]);
    glow(ctx, theme.green, 4, () => polyline(ctx, pts, { color: theme.green, width: 2, alpha: 1 }));

    // current-n dot
    const nNow = nDisp;
    const cx = wx(nNow);
    const cy = wy(weight(nNow));
    strokeLine(ctx, [cx, iy + ih - pad], [cx, cy], theme.cream, 1, 0.5);
    glow(ctx, theme.cream, 5, () => disc(ctx, [cx, cy], 3.4, theme.cream, 1));

    // labels
    ctx.fillStyle = theme.tick;
    ctx.font = `10px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("w", ix + 3, iy + 2);
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("n", ix + iw - 3, iy + ih - 2);
  }

  // --- interaction ----------------------------------------------------------
  function setN(n: number, animate = false): void {
    stopSweep();
    const t = clamp(Math.round(n), N_MIN, N_MAX);
    if (animate) nTracker.set(t, true);
    else nTracker.jump(t);
  }

  function setSampleMean(v: number): void {
    xbar = clamp(v, -half, half);
    render();
  }

  function setPriorMean(v: number): void {
    mu0 = clamp(v, -half, half);
    render();
  }

  function setPriorVar(v: number): void {
    tau2 = clampPos(v);
    render();
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setN(N_MAX, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    // sweep n from N_MIN up to N_MAX, watching the MAP estimate slide from the
    // prior mean onto the MLE as the data weight grows.
    let prog = (nDisp - N_MIN) / (N_MAX - N_MIN);
    nTracker.jump(N_MIN + prog * (N_MAX - N_MIN));
    sweepCancel = ticker((dt) => {
      prog += dt / 4.5;
      if (prog >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        nTracker.jump(N_MAX);
        return false;
      }
      // ease the traversal in n-space
      nTracker.jump(N_MIN + prog * (N_MAX - N_MIN));
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; grow n";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  nSlider.addEventListener("input", () => {
    stopSweep();
    nTracker.jump(clamp(Number(nSlider.value), N_MIN, N_MAX));
  });
  xSlider.addEventListener("input", () => {
    xbar = clamp(Number(xSlider.value), -half, half);
    render();
  });
  muSlider.addEventListener("input", () => {
    mu0 = clamp(Number(muSlider.value), -half, half);
    render();
  });
  tauSlider.addEventListener("input", () => {
    tau2 = clampPos(Number(tauSlider.value));
    render();
  });

  // drag on the canvas → move the MLE x̄ (the data) directly under the pointer
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      stopSweep();
      const ml = 18;
      const mr = 16;
      const frac = clamp((px - ml) / (rc.width - ml - mr), 0, 1);
      xbar = -half + frac * 2 * half;
      render();
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  // a11y + keyboard
  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Gaussian shrinkage: the MAP estimate of the mean is a precision-weighted blend of the prior mean and the sample mean. " +
      "Drag the blue sample-mean marker, or grow the number of observations n with the arrow keys; the gold MAP marker " +
      "shrinks from the sample mean toward the prior mean as n shrinks or the prior tightens.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const step = e.shiftKey ? 5 : 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      setN(Math.round(nDisp) - step, true);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      setN(Math.round(nDisp) + step, true);
    }
  };
  rc.canvas.addEventListener("keydown", onKey);

  render();
  // KaTeX labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setN,
    setSampleMean,
    setPriorMean,
    setPriorVar,
    play,
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

// --- module-local helpers ---------------------------------------------------

const N_MIN = 1;
const N_MAX = 60;

interface ControlRow {
  node: HTMLElement;
  readout: HTMLElement;
}

function controlRow(theme: Theme, labelTex: string, slider: HTMLInputElement): ControlRow {
  const node = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const label = el("span", { style: `min-width:2.2ch;color:${theme.fg};font-size:14px;` });
  label.innerHTML = tex(labelTex);
  const readout = el("div", { style: `font:13px ${ROMAN_FONT};color:${theme.muted};min-width:7.5ch;` });
  node.append(label, slider, readout);
  return { node, readout };
}

function makeSlider(min: string, max: string, step: string, value: string, ariaLabel: string): HTMLInputElement {
  return el("input", {
    type: "range",
    min,
    max,
    step,
    value,
    style: "flex:1;min-width:90px;",
    "aria-label": ariaLabel,
  });
}

function setReadout(node: HTMLElement, src: string): void {
  node.innerHTML = tex(src);
}

function clampPos(v: number): number {
  return Math.max(v, 1e-3);
}

function fmt(v: number): string {
  return v.toFixed(2);
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
