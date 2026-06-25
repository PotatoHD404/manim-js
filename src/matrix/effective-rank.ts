import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { type Theme, withTheme } from "../core/theme";

export type SpectrumModel = "equicorr" | "ar1";

export interface EffectiveRankOptions {
  /** Number of coordinates k (the nominal rank). Default 12 (clamped 2..24). */
  k?: number;
  /** Correlation ρ ∈ [0, 0.99]. The movable variable. Default 0.6. */
  rho?: number;
  /**
   * Covariance family: "equicorr" = Σ=(1−ρ)I+ρ·11ᵀ (one common mode),
   * "ar1" = Σ_ij=ρ^|i−j| (a colored stationary process). Default "equicorr".
   */
  model?: SpectrumModel;
  theme?: Partial<Theme>;
}

export interface EffectiveRankApi {
  /** Set the correlation ρ (eased glide unless `animate` is false). */
  setRho(rho: number, animate?: boolean): void;
  /** Set the number of coordinates k (integer, clamped to [2, 24]). */
  setK(k: number): void;
  /** Switch covariance family. */
  setModel(model: SpectrumModel): void;
  /** Sweep ρ from 0 to 1 and back, tracing the effective-rank collapse. */
  sweep(): void;
  destroy(): void;
}

const K_MIN = 2;
const K_MAX = 24;
const RHO_MAX = 0.99;

/**
 * Scene — "the effective rank is the entropy of the spectrum". A k-coordinate
 * covariance — equicorrelated Σ=(1−ρ)I+ρ·11ᵀ or AR(1) Σ_ij=ρ^|i−j| — has a
 * spectrum {λ_i}; normalize it to a probability distribution p_i = λ_i/Σλ and
 * the spectral entropy H(p)=−Σ p_i ln p_i defines the effective rank
 *
 *     r_eff = exp(H(p)) = exp(−Σ p_i ln p_i),
 *
 * the smooth dimension count (the participation / Roy–Vetterli number). At ρ=0
 * the spectrum is flat — every p_i=1/k, H=ln k, and r_eff equals the nominal
 * rank k. As ρ→1 the variance concentrates into one common direction, the
 * spectrum spikes, H collapses, and r_eff slides toward 1. The gold answer is
 * r_eff; the cream nominal rank k sits above it; the green curve below traces
 * r_eff(ρ) across the whole correlation range and the hard count #{λ_i>ε} steps
 * underneath it. The geometric-to-arithmetic mean ratio of the eigenvalues —
 * the post's spectral-flatness γ (§11, Wiener entropy) — is shown alongside,
 * since exp(H) and γ are the same "how spread out is the spectrum" statement.
 *
 * Math ported from the post's demo library (buildEquicorr, buildAR1, the cyclic
 * Jacobi eigensolver). The spectrum is computed from the assembled matrix — never
 * a closed form — so the bars are real numerics; the closed-form equicorrelated
 * levels λ₁=1+(k−1)ρ and λ=1−ρ appear only as overlay labels.
 */
export function createMatrixEffectiveRank(
  target: HTMLElement,
  options: EffectiveRankOptions = {},
): EffectiveRankApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let k = clamp(Math.round(options.k ?? 12), K_MIN, K_MAX);
  let rho = clamp(options.rho ?? 0.6, 0, RHO_MAX); // displayed (eased) ρ
  let model: SpectrumModel = options.model === "ar1" ? "ar1" : "equicorr";
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  // The hard-count threshold ε relative to the mean eigenvalue (trace/k = 1):
  // a direction "counts" only if it carries more than this fraction of the
  // average variance. Anchors r_hard at k for the flat spectrum.
  const HARD_EPS = 0.5;

  // --- math (ported from demos-core.js) ------------------------------------
  // buildEquicorr(k, ρ): unit diagonal, ρ everywhere off-diagonal.
  function buildEquicorr(kk: number, r: number): Mat {
    const M: Mat = [];
    for (let i = 0; i < kk; i++) {
      const row: number[] = [];
      for (let j = 0; j < kk; j++) row.push(i === j ? 1 : r);
      M.push(row);
    }
    return M;
  }

  // buildAR1(k, ρ): Σ_ij = ρ^|i−j| — a colored stationary covariance.
  function buildAR1(kk: number, r: number): Mat {
    const M: Mat = [];
    for (let i = 0; i < kk; i++) {
      const row: number[] = [];
      for (let j = 0; j < kk; j++) row.push(Math.pow(r, Math.abs(i - j)));
      M.push(row);
    }
    return M;
  }

  function build(r: number): Mat {
    return model === "ar1" ? buildAR1(k, r) : buildEquicorr(k, r);
  }

  interface Spec {
    eigs: number[]; // eigenvalues, descending (real numerics from Jacobi)
    weights: number[]; // p_i = λ_i / Σλ
    entropy: number; // H(p) = −Σ p_i ln p_i (natural log)
    effRank: number; // exp(H) = effective rank
    hardCount: number; // #{λ_i > HARD_EPS · mean}
    flatness: number; // γ = geom-mean / arith-mean of eigenvalues ∈ (0,1]
  }

  function compute(r: number): Spec {
    const M = build(r);
    const eigs = jacobiEigen(M).values.map((v) => Math.max(v, 0));
    const total = eigs.reduce((a, b) => a + b, 0) || 1;
    const weights = eigs.map((v) => v / total);
    let entropy = 0;
    let logSum = 0;
    let hardCount = 0;
    const meanEig = total / k;
    for (let i = 0; i < k; i++) {
      const p = weights[i];
      if (p > 0) entropy -= p * Math.log(p);
      logSum += Math.log(Math.max(eigs[i], 1e-300));
      if (eigs[i] > HARD_EPS * meanEig) hardCount += 1;
    }
    const geoMean = Math.exp(logSum / k);
    const flatness = clamp(geoMean / meanEig, 0, 1);
    return {
      eigs,
      weights,
      entropy,
      effRank: Math.exp(entropy),
      hardCount,
      flatness,
    };
  }

  // --- precomputed r_eff(ρ) trace (recomputed when k or the model changes) --
  const TRACE_N = 121;
  let traceEff: number[] = [];
  let traceHard: number[] = [];
  function rebuildTrace(): void {
    traceEff = new Array(TRACE_N);
    traceHard = new Array(TRACE_N);
    for (let i = 0; i < TRACE_N; i++) {
      const r = (i / (TRACE_N - 1)) * RHO_MAX;
      const s = compute(r);
      traceEff[i] = s.effRank;
      traceHard[i] = s.hardCount;
    }
  }
  rebuildTrace();

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  const traceHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(mainHost, traceHost);
  root.append(panel);

  // top-left: the effective rank (the answer), nominal rank, and ρ
  const overlay = overlayLayer("tl");
  const eqEff = liveEquation("r_{\\text{eff}}=e^{H(p)}\\;=", theme, theme.gold);
  const eqK = liveEquation("\\text{nominal rank }k\\;=", theme, theme.cream);
  const eqRho = liveEquation("\\rho\\;=", theme, theme.cream);
  overlay.append(eqEff.node, eqK.node, eqRho.node);
  mainHost.append(overlay);

  // top-right: the definition + the spectral-flatness invariant
  const defOverlay = overlayLayer("tr");
  const eqModel = staticEquation("", theme, theme.fg);
  const eqFlat = staticEquation("", theme, theme.green);
  defOverlay.append(
    staticEquation("p_i=\\dfrac{\\lambda_i}{\\sum_j\\lambda_j},\\quad H(p)=-\\textstyle\\sum_i p_i\\ln p_i", theme, theme.blue),
    eqModel,
    eqFlat,
  );
  mainHost.append(defOverlay);

  const traceTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.green};font:12px ${theme.mono};`,
  });
  traceTag.textContent = "effective rank  e^{H}  vs.  ρ";
  traceHost.append(traceTag);

  const rcMain = responsiveCanvas(mainHost, 2.0, () => render());
  const rcTrace = responsiveCanvas(traceHost, 4.9, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:14px;flex-wrap:wrap;" });

  const sweepBtn = el("button", { type: "button", style: btnStyle(theme) });

  const modelBtn = el("button", { type: "button", style: btnStyle(theme) });

  const kRow = sliderRow(theme, "coords  k", K_MIN, K_MAX, 1, k, "number of coordinates k, the nominal rank");
  const rhoRow = sliderRow(theme, "corr  ρ", 0, RHO_MAX, 0.01, rho, "correlation rho");

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ ρ  ·  sweep  ·  ←→ scrub";
  controls.append(sweepBtn, modelBtn, kRow.row, rhoRow.row, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: ρ ---------------------------------------------
  const rhoTracker = valueTracker(rho, (v) => {
    rho = clamp(v, 0, RHO_MAX);
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    const spec = compute(rho);
    drawSpectrum(spec);
    drawTrace(spec);

    eqEff.set(spec.effRank.toFixed(3));
    eqK.set(String(k));
    eqRho.set(rho.toFixed(3));
    eqModel.innerHTML = staticEquation(
      model === "ar1"
        ? "\\Sigma_{ij}=\\rho^{\\,|i-j|}\\quad(\\text{AR}(1))"
        : "\\Sigma=(1-\\rho)I+\\rho\\,\\mathbf{1}\\mathbf{1}^{\\top}",
      theme,
      theme.fg,
    ).innerHTML;
    eqFlat.innerHTML = staticEquation(
      `\\gamma=\\dfrac{\\text{geo}}{\\text{arith}}=${spec.flatness.toFixed(3)}`,
      theme,
      theme.green,
    ).innerHTML;

    if (document.activeElement !== kRow.input) kRow.set(k);
    if (document.activeElement !== rhoRow.input) rhoRow.set(Number(rho.toFixed(2)));
    updateModelBtn();
  }

  // --- main panel: the eigenvalue spectrum (weights p_i) -------------------
  function drawSpectrum(spec: Spec): void {
    const { ctx, width: W, height: H } = rcMain;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 34;
    const mr = 16;
    const mt = 64;
    const mb = 28;
    const x0 = ml;
    const yBase = H - mb;
    const innerW = W - ml - mr;
    const innerH = H - mb - mt;

    const weights = spec.weights;
    // The bars are the normalized weights p_i (a probability distribution): a
    // flat 1/k spectrum at ρ=0, a single spike near 1 as ρ→1. Scale to the max
    // weight so the structure is always legible.
    const wmax = Math.max(...weights, 1 / k) * 1.08;
    const y = (v: number) => yBase - (clamp(v, 0, wmax) / wmax) * innerH;
    const xAt = (i: number) => x0 + ((i + 0.5) / k) * innerW;
    const bw = Math.max(2, (innerW / k) * 0.62);

    // axes
    strokeLine(ctx, [x0, mt], [x0, yBase], theme.axis, 1.4, 1);
    strokeLine(ctx, [x0, yBase], [x0 + innerW, yBase], theme.axis, 1.4, 1);

    // the uniform reference 1/k (cream): the spectrum at ρ=0, where r_eff=k
    const yUnif = y(1 / k);
    polyline(ctx, [[x0, yUnif], [x0 + innerW, yUnif]], {
      color: theme.cream,
      width: 1.2,
      alpha: 0.55,
      dash: [5, 6],
    });
    ctx.fillStyle = theme.cream;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("1/k", x0 + 4, yUnif - 3);

    // bars: color each by how much of the spectrum it carries — a dominant
    // direction glows gold (the "answer" mass), the redundant floor stays blue.
    const wHi = Math.max(...weights);
    for (let i = 0; i < k; i++) {
      const cx = xAt(i);
      const top = y(weights[i]);
      const dominant = weights[i] > 1.5 / k && weights[i] >= wHi - 1e-9;
      const color = dominant ? theme.gold : theme.blue;
      ctx.globalAlpha = dominant ? 0.92 : 0.8;
      ctx.fillStyle = color;
      ctx.fillRect(cx - bw / 2, top, bw, yBase - top);
      ctx.globalAlpha = 1;
      if (dominant) {
        glow(ctx, theme.gold, 6, () => {
          ctx.fillStyle = theme.gold;
          ctx.fillRect(cx - bw / 2, top, bw, Math.min(4, yBase - top));
        });
      }
    }

    // the effective rank as a gold marker sitting between bar #⌈r_eff⌉−1 and the
    // next — a vertical rule showing where the "effective" directions run out.
    const reff = clamp(spec.effRank, 1, k);
    const markX = x0 + (reff / k) * innerW;
    glow(ctx, theme.gold, 7, () => {
      strokeLine(ctx, [markX, mt], [markX, yBase], theme.gold, 1.8, 0.9);
    });
    disc(ctx, [markX, mt + 6], 3.4, theme.gold, 1);
    mathLabel(ctx, "r", [markX + (reff < k - 0.5 ? 6 : -6), mt + 4], {
      color: theme.gold,
      size: 15,
      sub: "eff",
      align: reff < k - 0.5 ? "left" : "right",
      glow: 5,
    });

    // the nominal rank k as a cream marker at the right edge (all k directions)
    const kx = x0 + innerW;
    strokeLine(ctx, [kx, mt], [kx, yBase], theme.cream, 1.3, 0.45);

    // x ticks (eigenvalue index, roman numerals — thinned when k is large)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tickEvery = k > 14 ? 2 : 1;
    for (let i = 0; i < k; i++) {
      if (i % tickEvery !== 0 && i !== k - 1) continue;
      ctx.fillText(String(i + 1), xAt(i), yBase + 5);
    }

    // axis caption
    mathLabel(ctx, "p", [x0 - 22, mt + 12], { color: theme.tick, size: 15, sub: "i" });
  }

  // --- trace panel: r_eff(ρ) across the whole correlation range ------------
  function drawTrace(spec: Spec): void {
    const { ctx, width: W, height: H } = rcTrace;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 30;
    const mr = 14;
    const mt = 26;
    const mb = 22;
    const x = (r: number) => ml + (r / RHO_MAX) * (W - ml - mr);
    const ymax = k * 1.06;
    const y = (v: number) => H - mb - (clamp(v, 0, ymax) / ymax) * (H - mt - mb);

    // axes
    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.2, 1);
    strokeLine(ctx, [ml, H - mb], [ml, mt], theme.axis, 1.2, 1);

    // y grid + ticks at the nominal rank k and the rank-one floor 1
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const g of [1, k]) {
      strokeLine(ctx, [ml, y(g)], [W - mr, y(g)], theme.grid, 1, 0.22);
      ctx.fillText(String(g), ml - 6, y(g));
    }

    // the nominal rank k as a dashed cream ceiling (r_eff can never exceed it)
    polyline(ctx, [[ml, y(k)], [W - mr, y(k)]], { color: theme.cream, width: 1.2, alpha: 0.5, dash: [6, 6] });

    // the hard count #{λ_i>ε} as a faint red step underneath the soft curve
    const hardPts: Pt[] = traceHard.map((v, i) => [x((i / (TRACE_N - 1)) * RHO_MAX), y(v)]);
    polyline(ctx, hardPts, { color: theme.red, width: 1.4, alpha: 0.45 });

    // the soft effective rank e^{H(ρ)} — faint full curve, bright up to ρ
    const curve: Pt[] = traceEff.map((v, i) => [x((i / (TRACE_N - 1)) * RHO_MAX), y(v)]);
    polyline(ctx, curve, { color: theme.green, width: 1.6, alpha: 0.32 });
    const upto = curve.filter((_, i) => (i / (TRACE_N - 1)) * RHO_MAX <= rho);
    if (upto.length > 1) polyline(ctx, upto, { color: theme.green, width: 2.4, alpha: 1 });

    // current ρ marker: a cream rule and the gold answer dot on the curve
    const cx = x(rho);
    strokeLine(ctx, [cx, H - mb], [cx, y(spec.effRank)], theme.cream, 1, 0.5);
    glow(ctx, theme.gold, 6, () => disc(ctx, [cx, y(spec.effRank)], 3.6, theme.gold, 1));

    // x ticks (ρ)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const r of [0, 0.25, 0.5, 0.75, 1]) {
      const rr = r * RHO_MAX;
      ctx.fillText(r.toFixed(2), x(rr), H - mb + 4);
    }
  }

  // --- interaction ---------------------------------------------------------
  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updateSweepBtn();
  }

  function setRho(r: number, animate = false): void {
    stopSweep();
    const t = clamp(r, 0, RHO_MAX);
    if (animate) rhoTracker.set(t, true);
    else rhoTracker.jump(t);
  }

  function setK(nk: number): void {
    k = clamp(Math.round(nk), K_MIN, K_MAX);
    rebuildTrace();
    render();
  }

  function setModel(m: SpectrumModel): void {
    model = m === "ar1" ? "ar1" : "equicorr";
    rebuildTrace();
    render();
  }

  function sweep(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setRho(RHO_MAX, false);
      return;
    }
    sweepActive = true;
    updateSweepBtn();
    // a there-and-back traversal: collapse to ρ=1, then relax back to 0.
    let phase = rho / RHO_MAX; // [0,1] up-leg progress
    let returning = false;
    sweepCancel = ticker((dt) => {
      phase += dt / 3.2;
      if (phase >= 1) {
        phase = 1;
        if (!returning) {
          returning = true;
          phase = 0;
          rhoTracker.jump(RHO_MAX);
          return;
        }
        sweepActive = false;
        sweepCancel = null;
        updateSweepBtn();
        rhoTracker.set(0, true);
        return false;
      }
      rhoTracker.jump((returning ? 1 - phase : phase) * RHO_MAX);
    });
  }

  function updateSweepBtn(): void {
    sweepBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; sweep ρ";
  }
  function updateModelBtn(): void {
    modelBtn.innerHTML = model === "ar1" ? "&#8646;&nbsp; AR(1)" : "&#8862;&nbsp; equicorr";
  }
  updateSweepBtn();
  updateModelBtn();

  sweepBtn.addEventListener("click", sweep);
  modelBtn.addEventListener("click", () => setModel(model === "ar1" ? "equicorr" : "ar1"));
  kRow.input.addEventListener("input", () => setK(Number(kRow.input.value)));
  rhoRow.input.addEventListener("input", () => setRho(Number(rhoRow.input.value), false));

  // Horizontal drag on either canvas scrubs ρ across its whole range.
  const stopDragMain = draggable(
    rcMain.canvas,
    (px) => {
      stopSweep();
      rhoTracker.jump(clamp(px / rcMain.width, 0, 1) * RHO_MAX);
    },
    { onStart: () => (rcMain.canvas.style.cursor = "grabbing"), onEnd: () => (rcMain.canvas.style.cursor = "grab") },
  );
  rcMain.canvas.style.cursor = "grab";

  const stopDragTrace = draggable(
    rcTrace.canvas,
    (px) => {
      stopSweep();
      const ml = 30;
      const mr = 14;
      const f = clamp((px - ml) / (rcTrace.width - ml - mr), 0, 1);
      rhoTracker.jump(f * RHO_MAX);
    },
    { onStart: () => (rcTrace.canvas.style.cursor = "grabbing"), onEnd: () => (rcTrace.canvas.style.cursor = "grab") },
  );
  rcTrace.canvas.style.cursor = "grab";

  rcMain.canvas.tabIndex = 0;
  rcMain.canvas.setAttribute("role", "img");
  rcMain.canvas.setAttribute(
    "aria-label",
    "Effective rank of a covariance spectrum: the eigenvalues normalized to a probability distribution p, with the effective rank exp of the spectral entropy marked in gold and the nominal rank k in cream. Raising the correlation rho concentrates the spectrum into one direction and the effective rank collapses from k toward one.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      sweep();
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      setRho(0, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setRho(RHO_MAX, true);
      return;
    }
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setRho(rho - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setRho(rho + step, true);
    else return;
    e.preventDefault();
  };
  rcMain.canvas.addEventListener("keydown", onKey);

  // Reduced motion: no glide — land on the configured ρ immediately.
  if (prefersReducedMotion()) rho = clamp(options.rho ?? 0.6, 0, RHO_MAX);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setRho,
    setK,
    setModel,
    sweep,
    destroy() {
      destroyed = true;
      rhoTracker.stop();
      sweepCancel?.();
      stopDragMain();
      stopDragTrace();
      rcMain.canvas.removeEventListener("keydown", onKey);
      rcMain.destroy();
      rcTrace.destroy();
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

function sliderRow(
  theme: Theme,
  label: string,
  min: number,
  max: number,
  step: number,
  value: number,
  aria: string,
): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:9px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:6ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(step),
    value: String(value),
    style: "flex:1;min-width:110px;max-width:200px;",
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

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
