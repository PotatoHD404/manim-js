import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { type Theme, withTheme } from "../core/theme";

export interface EquicorrelatedOptions {
  /** Block size k of the equicorrelated covariance. Default 8 (clamped 2..12). */
  k?: number;
  /** Equicorrelation ρ ∈ [0, 0.99]. The movable variable. Default 0.6. */
  rho?: number;
  /**
   * Observation resolution Δ for the quantized-Gaussian capacity, exactly the
   * post's `gaussian_capacity_quantized`. Default 1e-2.
   */
  delta?: number;
  theme?: Partial<Theme>;
}

export interface EquicorrelatedApi {
  /** Set the correlation ρ (eased glide unless `animate` is false). */
  setRho(rho: number, animate?: boolean): void;
  /** Set the block size k (integer, clamped to [2, 12]). */
  setK(k: number): void;
  /** Glide ρ all the way to 1 (rank-one limit, C → k). */
  collapseToRankOne(): void;
  destroy(): void;
}

const K_MIN = 2;
const K_MAX = 12;
const RHO_MAX = 0.99;

/**
 * Scene — "all-to-all redundancy is one big direction plus a flat floor". The
 * equicorrelated covariance Σ = (1−ρ)I + ρ·11ᵀ is shown as a k×k heatmap (unit
 * diagonal, ρ off-diagonal) beside its eigenvalue spectrum. That spectrum is
 * exactly two levels: one dominant eigenvalue λ₁ = 1 + (k−1)ρ (the gold
 * "answer") and a degenerate cluster of k−1 equal small eigenvalues λ = 1−ρ
 * (the blue redundant floor). Dialing ρ up drains every coordinate's
 * independent variance into the single common direction: the floor sinks toward
 * 0, λ₁ rises toward k, and the quantized-Gaussian completion capacity C climbs
 * from 1 (independent) toward k (rank one). The green readout k/C — the
 * effective rank — slides from k down toward 1 as the redundancy concentrates.
 *
 * Math ported verbatim from the post's demo library (buildEquicorr, the cyclic
 * Jacobi eigensolver, and quantizedCapacity = the notebook's
 * gaussian_capacity_quantized). The spectrum is computed from the assembled
 * matrix — never the closed form — so the bars are real numerics; the closed
 * forms λ₁ = 1 + (k−1)ρ and λ = 1−ρ are shown only as overlay labels.
 */
export function createMatrixEquicorrelated(
  target: HTMLElement,
  options: EquicorrelatedOptions = {},
): EquicorrelatedApi {
  const theme = withTheme(options.theme);
  const delta = options.delta ?? 1e-2;

  // --- model state ---------------------------------------------------------
  let k = clamp(Math.round(options.k ?? 8), K_MIN, K_MAX);
  let rho = clamp(options.rho ?? 0.6, 0, RHO_MAX); // displayed (eased) ρ
  let destroyed = false;

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

  // The quantized-Gaussian completion capacity, exactly the post's
  // gaussian_capacity_quantized / quantizedCapacity. Each direction with
  // variance λ contributes max(0, ½log₂(2πe·λ) − log₂Δ) bits; S sums over the
  // diagonal variances (all 1 here), Hⱼ over the eigenvalues, and C = S / Hⱼ.
  const bits = (lam: number): number =>
    Math.max(0, 0.5 * Math.log2(2 * Math.PI * Math.E * Math.max(lam, 1e-300)) - Math.log2(delta));

  interface Spec {
    eigs: number[]; // eigenvalues, descending (real numerics from Jacobi)
    lam1: number; // closed form 1 + (k−1)ρ
    lamRest: number; // closed form 1 − ρ
    capacity: number; // quantized capacity C
    nu: number; // implied variance floor ν = Δ²/(2πe) (the post's water level)
  }

  function compute(r: number): Spec {
    const M = buildEquicorr(k, r);
    const eigs = jacobiEigen(M).values.map((v) => Math.max(v, 0));
    const S = k * bits(1); // every diagonal variance is exactly 1
    const Hj = eigs.reduce((acc, l) => acc + bits(l), 0);
    const capacity = Hj < 1e-12 ? Infinity : S / Hj;
    return {
      eigs,
      lam1: 1 + (k - 1) * r,
      lamRest: 1 - r,
      capacity,
      nu: (delta * delta) / (2 * Math.PI * Math.E),
    };
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: capacity and the effective rank it implies
  const overlay = overlayLayer("tl");
  const eqCap = liveEquation("C=\\dfrac{S}{H_{\\text{joint}}}\\;=", theme, theme.gold);
  const eqRank = liveEquation("\\dfrac{k}{C}=r_{\\text{eff}}\\;=", theme, theme.green);
  const eqRho = liveEquation("\\rho\\;=", theme, theme.cream);
  overlay.append(eqCap.node, eqRank.node, eqRho.node);
  mainHost.append(overlay);

  // top-right: the closed-form spectrum the figure is "about"
  const specOverlay = overlayLayer("tr");
  const eqLam1 = staticEquation("", theme, theme.gold);
  const eqLamRest = staticEquation("", theme, theme.blue);
  specOverlay.append(
    staticEquation("\\Sigma=(1-\\rho)I+\\rho\\,\\mathbf{1}\\mathbf{1}^{\\top}", theme, theme.fg),
    eqLam1,
    eqLamRest,
  );
  mainHost.append(specOverlay);

  const rc = responsiveCanvas(mainHost, 2.0, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:14px;flex-wrap:wrap;" });

  const collapseBtn = el("button", { type: "button", style: btnStyle(theme) });
  collapseBtn.innerHTML = "&#9650;&nbsp; rank-one limit";

  const kRow = sliderRow(theme, "block  k", K_MIN, K_MAX, 1, k, "block size k of the equicorrelated covariance");
  const rhoRow = sliderRow(theme, "corr  ρ", 0, RHO_MAX, 0.01, rho, "equicorrelation rho");

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↔ ρ  ·  ←→ scrub  ·  sliders";
  controls.append(collapseBtn, kRow.row, rhoRow.row, hint);
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
    draw(spec);

    const cap = spec.capacity;
    eqCap.set(Number.isFinite(cap) ? cap.toFixed(3) : "∞");
    eqRank.set(Number.isFinite(cap) && cap > 0 ? (k / cap).toFixed(3) : "1");
    eqRho.set(rho.toFixed(3));
    eqLam1.innerHTML = staticEquation(
      `\\lambda_1=1+(k-1)\\rho=${spec.lam1.toFixed(2)}`,
      theme,
      theme.gold,
    ).innerHTML;
    eqLamRest.innerHTML = staticEquation(
      `\\lambda_{2..k}=1-\\rho=${spec.lamRest.toFixed(2)}`,
      theme,
      theme.blue,
    ).innerHTML;

    if (document.activeElement !== kRow.input) kRow.set(k);
    if (document.activeElement !== rhoRow.input) rhoRow.set(Number(rho.toFixed(2)));
  }

  function draw(spec: Spec): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    // Left half: the covariance heatmap. Right half: the eigenvalue spectrum.
    const gap = 26;
    const leftW = Math.min(H * 1.04, (W - gap) * 0.46);
    drawHeatmap(ctx, 14, 18, leftW, H - 36);
    drawSpectrum(ctx, 14 + leftW + gap, 14, W - (14 + leftW + gap) - 14, H - 28, spec);
  }

  /** The k×k matrix Σ as a grid of cells: unit diagonal, ρ off-diagonal. */
  function drawHeatmap(ctx: CanvasRenderingContext2D, ox: number, oy: number, side: number, avail: number): void {
    const sz = Math.min(side, avail);
    const cell = sz / k;
    const x0 = ox;
    const y0 = oy + (avail - sz) / 2;

    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const diag = i === j;
        const val = diag ? 1 : rho; // the entry of Σ
        // diagonal cells are the gold "answer" tint; off-diagonal cells fill
        // with blue whose opacity tracks ρ — the redundancy made visible.
        const color = diag ? theme.gold : theme.blue;
        const alpha = diag ? 0.92 : clamp(0.12 + 0.78 * val, 0, 0.92);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillRect(x0 + j * cell + 0.6, y0 + i * cell + 0.6, cell - 1.2, cell - 1.2);
        ctx.globalAlpha = 1;
      }
    }
    // thin cyan grid framing the matrix (the receding number-plane look)
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= k; i++) {
      strokeLine(ctx, [x0, y0 + i * cell], [x0 + sz, y0 + i * cell], theme.grid, 1, 0.4);
      strokeLine(ctx, [x0 + i * cell, y0], [x0 + i * cell, y0 + sz], theme.grid, 1, 0.4);
    }
    ctx.globalAlpha = 1;

    mathLabel(ctx, "Σ", [x0 + sz / 2, y0 - 6], {
      color: theme.fg,
      size: 15,
      align: "center",
      baseline: "bottom",
    });
  }

  /** Eigenvalue spectrum: one gold bar (λ₁), then the blue degenerate cluster. */
  function drawSpectrum(
    ctx: CanvasRenderingContext2D,
    ox: number,
    oy: number,
    plotW: number,
    plotH: number,
    spec: Spec,
  ): void {
    const ml = 30;
    const mb = 26;
    const mt = 16;
    const x0 = ox + ml;
    const yBase = oy + plotH - mb;
    const innerW = plotW - ml;
    const innerH = plotH - mb - mt;

    const eigs = spec.eigs;
    // shared y-scale: λ₁ can reach k (rank-one limit), so scale to k for stability
    const ymax = Math.max(k, spec.lam1) * 1.06;
    const y = (v: number) => yBase - (clamp(v, 0, ymax) / ymax) * innerH;
    const xAt = (i: number) => x0 + ((i + 0.5) / k) * innerW;
    const bw = (innerW / k) * 0.62;

    // axes
    strokeLine(ctx, [x0, oy + mt], [x0, yBase], theme.axis, 1.4, 1);
    strokeLine(ctx, [x0, yBase], [x0 + innerW, yBase], theme.axis, 1.4, 1);

    // y grid + ticks at integer eigenvalue levels (1, 2, …)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yStep = ymax > 8 ? 2 : 1;
    for (let g = yStep; g <= ymax + 1e-6; g += yStep) {
      strokeLine(ctx, [x0, y(g)], [x0 + innerW, y(g)], theme.grid, 1, 0.22);
      ctx.fillText(String(g), x0 - 6, y(g));
    }

    // the variance floor ν (water level): directions below it are frozen and
    // carry no bits — the operational threshold behind the capacity.
    if (spec.nu < ymax && spec.nu > 0) {
      polyline(ctx, [[x0, y(spec.nu)], [x0 + innerW, y(spec.nu)]], {
        color: theme.red,
        width: 1.2,
        alpha: 0.55,
        dash: [5, 6],
      });
    }

    // bars: index 0 is the dominant eigenvalue (gold answer); the rest are the
    // degenerate redundant cluster (blue floor).
    for (let i = 0; i < k; i++) {
      const cx = xAt(i);
      const top = y(eigs[i]);
      const dominant = i === 0;
      const color = dominant ? theme.gold : theme.blue;
      ctx.globalAlpha = dominant ? 0.92 : 0.78;
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

    // a cream cap tracking the redundant-floor level λ = 1 − ρ (the movable
    // quantity), drawn as a line across the small-eigenvalue cluster.
    if (k > 1) {
      const yFloor = y(spec.lamRest);
      strokeLine(ctx, [xAt(1) - bw / 2 - 3, yFloor], [xAt(k - 1) + bw / 2 + 3, yFloor], theme.cream, 1.4, 0.85);
      disc(ctx, [xAt(Math.max(1, Math.floor((k + 1) / 2))), yFloor], 3, theme.cream, 1);
    }

    // x ticks (eigenvalue index, roman numerals)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < k; i++) ctx.fillText(String(i + 1), xAt(i), yBase + 5);

    // axis caption
    mathLabel(ctx, "λ", [x0 - 22, oy + mt + 10], { color: theme.tick, size: 15, sub: "i" });
  }

  // --- interaction ---------------------------------------------------------
  function setRho(r: number, animate = false): void {
    const t = clamp(r, 0, RHO_MAX);
    if (animate) rhoTracker.set(t, true);
    else rhoTracker.jump(t);
  }

  function setK(nk: number): void {
    k = clamp(Math.round(nk), K_MIN, K_MAX);
    render();
  }

  function collapseToRankOne(): void {
    setRho(RHO_MAX, true);
  }

  collapseBtn.addEventListener("click", collapseToRankOne);
  kRow.input.addEventListener("input", () => setK(Number(kRow.input.value)));
  rhoRow.input.addEventListener("input", () => setRho(Number(rhoRow.input.value), false));

  // Horizontal drag on the canvas scrubs ρ across its whole range.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const f = clamp(px / rc.width, 0, 1);
      rhoTracker.jump(f * RHO_MAX);
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Equicorrelated covariance: a k-by-k heatmap with unit diagonal and rho off-diagonal beside its eigenvalue spectrum — one dominant eigenvalue one plus (k minus one) rho, and k minus one equal small eigenvalues one minus rho. Raising rho drains the floor toward zero and the completion capacity rises from one toward k.",
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setRho(0, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      collapseToRankOne();
      return;
    }
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setRho(rho - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setRho(rho + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

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
    collapseToRankOne,
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
