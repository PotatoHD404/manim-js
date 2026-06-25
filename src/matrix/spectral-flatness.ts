import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { type Theme, withTheme } from "../core/theme";

export type FlatnessModel = "equi" | "ar1";

export interface SpectralFlatnessOptions {
  /** Covariance model: "equi" = equicorrelated, "ar1" = AR(1) Toeplitz. */
  model?: FlatnessModel;
  /** Matrix size k (number of stationary coordinates / eigenvalues). */
  size?: number;
  /** Initial correlation ρ ∈ [0, 0.98]. */
  rho?: number;
  theme?: Partial<Theme>;
}

export interface SpectralFlatnessApi {
  setRho(rho: number, animate?: boolean): void;
  setModel(model: FlatnessModel): void;
  setSize(k: number): void;
  destroy(): void;
}

const RHO_MAX = 0.98;

/**
 * Scene — "how flat is the spectrum?". For a unit-variance stationary column the
 * covariance is either equicorrelated (Σ_ij = ρ off-diagonal) or AR(1)
 * (Σ_ij = ρ^|i−j|). Its k eigenvalues are the spectrum; the spectral flatness
 *
 *     γ = geometric mean(λ) / arithmetic mean(λ) ∈ (0, 1]
 *
 * is the Wiener entropy of that spectrum. White noise (ρ = 0) has a flat
 * spectrum and γ = 1; as ρ → 1 the spectrum collapses onto one dominant
 * eigenvalue and γ → 0. The per-coordinate redundancy converges, by the
 * Kolmogorov–Szegő limit, to TC_k/k → −½ log₂ γ — the bits each extra column
 * shares with the rest. The gold geometric-mean line sinks below the cream
 * arithmetic-mean line exactly as ρ grows; the meter reads γ off in [0, 1].
 *
 * Eigenvalues come from the same Jacobi routine the post uses (buildEquicorr /
 * buildAR1 → jacobiEigvals); the flatness and redundancy formulas are the post's
 * §11 verbatim.
 */
export function createSpectralFlatness(
  target: HTMLElement,
  options: SpectralFlatnessOptions = {},
): SpectralFlatnessApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let model: FlatnessModel = options.model === "ar1" ? "ar1" : "equi";
  let k = clamp(Math.round(options.size ?? 8), 2, 12);
  let rho = clamp(options.rho ?? 0.6, 0, RHO_MAX);
  let destroyed = false;

  // --- covariance builders (the post's buildEquicorr / buildAR1) -----------
  function buildEquicorr(kk: number, rr: number): Mat {
    const M: Mat = [];
    for (let i = 0; i < kk; i++) {
      const row: number[] = [];
      for (let j = 0; j < kk; j++) row.push(i === j ? 1 : rr);
      M.push(row);
    }
    return M;
  }
  function buildAR1(kk: number, rr: number): Mat {
    const M: Mat = [];
    for (let i = 0; i < kk; i++) {
      const row: number[] = [];
      for (let j = 0; j < kk; j++) row.push(rr ** Math.abs(i - j));
      M.push(row);
    }
    return M;
  }

  interface Spectrum {
    eigs: number[]; // descending
    am: number; // arithmetic mean (= 1 for the unit-diagonal models)
    gm: number; // geometric mean
    gamma: number; // spectral flatness γ = gm / am ∈ (0,1]
    redundancy: number; // −½ log₂ γ  (Kolmogorov–Szegő per-coordinate redundancy)
  }

  // Spectral flatness γ = geometric mean(λ) / arithmetic mean(λ) ∈ (0,1].
  function spectrumAt(rr: number): Spectrum {
    const M = model === "equi" ? buildEquicorr(k, rr) : buildAR1(k, rr);
    const eigs = jacobiEigen(M).values.map((v) => Math.max(v, 1e-300));
    let sum = 0;
    let logSum = 0;
    for (const l of eigs) {
      sum += l;
      logSum += Math.log(l);
    }
    const am = sum / k;
    const gm = Math.exp(logSum / k);
    const gamma = clamp(am > 0 ? gm / am : 1, 0, 1);
    const redundancy = -0.5 * Math.log2(Math.max(gamma, 1e-300));
    return { eigs, am, gm, gamma, redundancy };
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the live flatness and the redundancy it implies
  const overlay = overlayLayer("tl");
  const eqGamma = liveEquation("\\gamma=\\dfrac{\\mathrm{GM}(\\lambda)}{\\mathrm{AM}(\\lambda)}=", theme, theme.gold);
  const eqRed = liveEquation("\\mathrm{TC}_k/k\\to-\\tfrac12\\log_2\\gamma=", theme, theme.green);
  overlay.append(eqGamma.node, eqRed.node);
  panel.append(overlay);

  // top-right: the covariance model restated for the current numbers
  const modelOverlay = overlayLayer("tr");
  const eqModel = staticEquation("", theme, theme.blue);
  modelOverlay.append(eqModel);
  panel.append(modelOverlay);

  const rc = responsiveCanvas(panel, 1.74, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });

  const equiBtn = el("button", { type: "button", style: toggleStyle(theme, model === "equi") });
  equiBtn.textContent = "equicorrelated";
  const ar1Btn = el("button", { type: "button", style: toggleStyle(theme, model === "ar1") });
  ar1Btn.textContent = "AR(1)";

  const rhoLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  rhoLabel.textContent = "correlation  ρ";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: String(RHO_MAX),
    step: "0.01",
    value: String(rho),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "correlation rho",
  });
  const sizeLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  sizeLabel.textContent = "size  k";
  const sizeSlider = el("input", {
    type: "range",
    min: "2",
    max: "12",
    step: "1",
    value: String(k),
    style: "flex:0 1 110px;min-width:80px;",
    "aria-label": "matrix size k",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ρ · ←→";
  controls.append(equiBtn, ar1Btn, rhoLabel, slider, sizeLabel, sizeSlider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the correlation ρ -----------------------------
  const rhoTracker = valueTracker(rho, (v) => {
    rho = clamp(v, 0, RHO_MAX);
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    const s = spectrumAt(rho);
    draw(s);

    eqGamma.set(s.gamma.toFixed(3));
    eqRed.set(`${s.redundancy.toFixed(3)} bits`);
    eqModel.innerHTML = staticEquation(
      model === "equi"
        ? `\\Sigma_{ij}=\\rho^{[i\\neq j]},\\ k=${k}`
        : `\\Sigma_{ij}=\\rho^{|i-j|},\\ k=${k}`,
      theme,
      theme.blue,
    ).innerHTML;

    if (document.activeElement !== slider) slider.value = rho.toFixed(2);
    if (document.activeElement !== sizeSlider) sizeSlider.value = String(k);
  }

  function draw(s: Spectrum): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 40;
    const mr = 132; // room for the flatness meter on the right
    const mt = 64; // room for the overlay equations on top
    const mb = 30;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // A vertical scale tall enough to hold the largest eigenvalue (it is
    // 1+(k−1)ρ for equicorr, up to k). The arithmetic mean is always 1, so the
    // cream AM line sits at a fixed fraction of the scale.
    const maxEig = Math.max(...s.eigs, 1);
    const yMax = maxEig * 1.08;
    const x = (i: number) => ml + ((i + 0.5) / k) * plotW;
    const y = (v: number) => mt + plotH - (clamp(v, 0, yMax) / yMax) * plotH;

    // faint receding number-plane: a few horizontal rules in BLUE_D
    const gstep = niceStep(yMax);
    ctx.lineWidth = 1;
    for (let g = gstep; g <= yMax + 1e-9; g += gstep) {
      strokeLine(ctx, [ml, y(g)], [ml + plotW, y(g)], theme.grid, 1, 0.22);
    }
    // axes
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt + plotH], [ml + plotW, mt + plotH], theme.axis, 1.4, 1);

    // eigenvalue bars (the spectrum) — bright BLUE_C data, faintly glowing
    const bw = (plotW / k) * 0.56;
    glow(ctx, theme.blue, 4, () => {
      for (let i = 0; i < k; i++) {
        const cx = x(i);
        const top = y(s.eigs[i]);
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = theme.blue;
        ctx.fillRect(cx - bw / 2, top, bw, mt + plotH - top);
        ctx.globalAlpha = 1;
      }
    });
    // bar tops as discs so the spectrum reads as a sampled curve too
    for (let i = 0; i < k; i++) disc(ctx, [x(i), y(s.eigs[i])], 2.4, theme.blue, 1);

    // arithmetic mean (cream) — the reference; for unit diagonal it is exactly 1
    const yAm = y(s.am);
    polyline(ctx, [[ml, yAm], [ml + plotW, yAm]], { color: theme.cream, width: 1.6, alpha: 0.9, dash: [2, 4] });
    mathLabel(ctx, "AM", [ml + plotW - 4, yAm - 6], { color: theme.cream, size: 13, align: "right", italic: false });

    // geometric mean (gold) — the "answer": γ is how far this has sunk below AM
    const yGm = y(s.gm);
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[ml, yGm], [ml + plotW, yGm]], { color: theme.gold, width: 2, alpha: 0.95, dash: [7, 5] });
    });
    mathLabel(ctx, "GM", [ml + plotW - 4, yGm + 16], { color: theme.gold, size: 13, align: "right", italic: false });

    // the flatness gap between GM and AM, drawn as a bracket on the left margin
    glow(ctx, theme.gold, 4, () => {
      strokeLine(ctx, [ml + 8, yAm], [ml + 8, yGm], theme.gold, 1.4, 0.85);
      strokeLine(ctx, [ml + 5, yAm], [ml + 11, yAm], theme.gold, 1.4, 0.85);
      strokeLine(ctx, [ml + 5, yGm], [ml + 11, yGm], theme.gold, 1.4, 0.85);
    });

    // x ticks (eigenvalue index) and a y label
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < k; i++) ctx.fillText(String(i + 1), x(i), mt + plotH + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let g = gstep; g <= yMax + 1e-9; g += gstep) ctx.fillText(trim(g), ml - 5, y(g));
    mathLabel(ctx, "λ", [ml + plotW / 2, mt + plotH + 22], { color: theme.tick, size: 15, sub: "i", align: "center" });

    // the flatness meter on the right: γ in [0,1]
    drawFlatnessMeter(ctx, W, mt, mr, plotH, s.gamma);
  }

  /** A vertical [0,1] gauge for γ, gold near 1 (flat), red near 0 (collapsed). */
  function drawFlatnessMeter(
    ctx: CanvasRenderingContext2D,
    W: number,
    mt: number,
    mr: number,
    plotH: number,
    gamma: number,
  ): void {
    const bw = 16;
    const x0 = W - mr + 28;
    const top = mt + 8;
    const bot = mt + plotH - 8;
    const h = bot - top;
    const yOf = (g: number) => bot - clamp(g, 0, 1) * h;

    // track
    strokeLine(ctx, [x0, top], [x0, bot], theme.grid, bw, 0.3);
    // ticks at 0, 0.5, 1
    ctx.fillStyle = theme.tick;
    ctx.font = `10px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const g of [0, 0.5, 1]) {
      strokeLine(ctx, [x0 - bw / 2, yOf(g)], [x0 + bw / 2, yOf(g)], theme.muted, 1, 0.5);
      ctx.fillText(g.toFixed(1), x0 + bw / 2 + 5, yOf(g));
    }
    // fill from the bottom up to γ
    const col = gamma > 0.66 ? theme.gold : gamma > 0.33 ? theme.green : theme.red;
    glow(ctx, col, 6, () => strokeLine(ctx, [x0, bot], [x0, yOf(gamma)], col, bw, 0.95));
    // the read-off cap
    glow(ctx, col, 6, () => disc(ctx, [x0, yOf(gamma)], 4.5, col, 1));

    ctx.fillStyle = theme.muted;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("flatness", x0, top - 8);
    mathLabel(ctx, "γ", [x0, top - 22], { color: col, size: 16, align: "center" });
  }

  // --- interaction ---------------------------------------------------------
  function setRho(r: number, animate = false): void {
    const t = clamp(r, 0, RHO_MAX);
    if (animate) rhoTracker.set(t, true);
    else rhoTracker.jump(t);
  }
  function setModel(m: FlatnessModel): void {
    model = m === "ar1" ? "ar1" : "equi";
    equiBtn.setAttribute("style", toggleStyle(theme, model === "equi"));
    ar1Btn.setAttribute("style", toggleStyle(theme, model === "ar1"));
    render();
  }
  function setSize(nk: number): void {
    k = clamp(Math.round(nk), 2, 12);
    render();
  }

  equiBtn.addEventListener("click", () => setModel("equi"));
  ar1Btn.addEventListener("click", () => setModel("ar1"));
  slider.addEventListener("input", () => rhoTracker.jump(Number(slider.value)));
  sizeSlider.addEventListener("input", () => setSize(Number(sizeSlider.value)));

  // Horizontal drag on the canvas scrubs ρ across the full [0, RHO_MAX] range.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const ml = 40;
      const mr = 132;
      const plotW = rc.width - ml - mr;
      rhoTracker.jump(clamp((px - ml) / plotW, 0, 1) * RHO_MAX);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Spectral flatness: the eigenvalue spectrum of an equicorrelated or AR(1) covariance, with its geometric-mean and arithmetic-mean lines. The flatness gamma, the ratio of geometric to arithmetic mean, falls from one toward zero as the correlation rho grows; a meter reads it off in zero to one.",
  );
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setRho(rho - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setRho(rho + step, true);
    else if (e.key === "Home") setRho(0, true);
    else if (e.key === "End") setRho(RHO_MAX, true);
    else if (e.key === "m" || e.key === "M") setModel(model === "equi" ? "ar1" : "equi");
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: settle on the initial ρ with no glide.
  if (prefersReducedMotion()) rho = clamp(rho, 0, RHO_MAX);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setRho,
    setModel,
    setSize,
    destroy() {
      destroyed = true;
      rhoTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- helpers ---------------------------------------------------------------

/** A round-number grid step roughly dividing `max` into ~5 intervals. */
function niceStep(max: number): number {
  const raw = max / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}

/** Trim a tick number to at most one decimal, dropping a trailing ".0". */
function trim(v: number): string {
  const r = Math.round(v * 10) / 10;
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}

function toggleStyle(theme: Theme, active: boolean): string {
  const bg = active ? "#4a4a4a" : "#2a2a2a";
  const border = active ? theme.cream : "#555555";
  const fg = active ? theme.cream : theme.muted;
  return `font:13px ${theme.mono};color:${fg};background:${bg};border:1px solid ${border};border-radius:8px;padding:7px 12px;cursor:pointer;`;
}
