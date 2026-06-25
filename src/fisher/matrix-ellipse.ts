import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { arrow, disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { jacobiEigen, type Mat } from "../core/math/linalg";
import { Rng } from "../core/math/rng";
import { ellipseWorld, fitView, numberPlane, type View } from "../core/plane";
import { type Theme, withTheme } from "../core/theme";

export interface FisherMatrixEllipseOptions {
  /** Fisher matrix entry I₁₁ (curvature in the μ direction). Default 1/σ² = 0.25. */
  i11?: number;
  /** Fisher matrix entry I₂₂ (curvature in the σ² direction). Default 1/(2σ⁴) = 0.03125. */
  i22?: number;
  /** Off-diagonal coupling I₁₂ = I₂₁. Default 0 (μ and σ² are orthogonal). */
  i12?: number;
  /** Sample size n; the MLE covariance is I⁻¹/n. Default 200 (the post). */
  n?: number;
  /** Confidence-level constant c in ½δᵀIδ = c. χ²₂(0.95)/2 = 2.9957. */
  c?: number;
  /** Points in the asymptotic-MLE scatter. Default 600. */
  points?: number;
  seed?: number;
  theme?: Partial<Theme>;
}

export interface FisherMatrixEllipseApi {
  /** Set I₁₁ (clamped to a positive range). */
  setI11(v: number): void;
  /** Set I₂₂ (clamped to a positive range). */
  setI22(v: number): void;
  /** Set the off-diagonal I₁₂. */
  setI12(v: number): void;
  /** Set the level constant c (eased). */
  setC(c: number, animate?: boolean): void;
  /** Draw a fresh asymptotic-MLE cloud from the next seed. */
  reseed(): void;
  destroy(): void;
}

// χ²₂(0.95) = 5.991, so the 95% Fisher level set ½δᵀIδ = c uses c = 5.991/2.
const C95 = 5.991 / 2;

/**
 * Scene — "the Fisher matrix is a confidence ellipse (its inverse)". Ported from
 * the post's Gaussian 𝒩(μ,σ²) cell: the 2×2 information
 * I = [[1/σ², 0],[0, 1/(2σ⁴)]] defines the quadratic form ½δᵀIδ = c, whose level
 * set is the gold confidence ellipse on the (δμ, δσ²) plane. Its principal axes
 * are the eigenvectors vᵢ of I, with semi-lengths √(2c/λᵢ); along a high-curvature
 * (large-λ) direction the ellipse is short — that direction is sharply pinned —
 * and along a flat (small-λ) direction it is long. Because I⁻¹ is the asymptotic
 * covariance of the MLE, scaling z ↦ I^{-1/2}z/√n draws the seeded MLE scatter the
 * ellipse hugs, exactly as in the figure.
 *
 * Three sliders dial the free entries of the symmetric matrix (I₁₁, I₂₂, the
 * off-diagonal I₁₂); a draggable cursor / arrow keys grow and shrink the level c;
 * a re-seed control redraws the Monte-Carlo cloud. Live eigenvalue and ellipse-
 * area readouts track the matrix the whole time.
 */
export function createFisherMatrixEllipse(
  target: HTMLElement,
  options: FisherMatrixEllipseOptions = {},
): FisherMatrixEllipseApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  // Defaults are the notebook's I = diag(1/σ², 1/(2σ⁴)) at σ² = 4.
  let i11 = clamp(options.i11 ?? 0.25, 0.04, 1.2);
  let i22 = clamp(options.i22 ?? 0.03125, 0.01, 0.4);
  let i12 = clamp(options.i12 ?? 0, -0.18, 0.18);
  const n = options.n ?? 200;
  const nPts = options.points ?? 600;
  let seed = options.seed ?? 0;
  let destroyed = false;

  // --- derived: eigendecomposition + a seeded asymptotic-MLE cloud ----------
  let v1: Pt = [1, 0];
  let v2: Pt = [0, 1];
  let l1 = i11;
  let l2 = i22;
  // z-cloud (white noise) is drawn once per seed; the displayed cloud is
  // z pushed through I^{-1/2} so it always matches the current matrix.
  let zCloud: Pt[] = [];

  function drawCloud(): void {
    const rng = new Rng(seed >>> 0 || 1);
    zCloud = Array.from({ length: nPts }, () => [rng.gauss(), rng.gauss()]);
  }

  function decompose(): void {
    const I: Mat = [
      [i11, i12],
      [i12, i22],
    ];
    const eig = jacobiEigen(I);
    v1 = eig.vectors[0] as Pt;
    v2 = eig.vectors[1] as Pt;
    l1 = Math.max(eig.values[0], 1e-6);
    l2 = Math.max(eig.values[1], 1e-6);
  }

  drawCloud();
  decompose();

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the live eigenvalues of I and the level constant
  const overlay = overlayLayer("tl");
  const eqL1 = liveEquation("\\lambda_1\\;=", theme, theme.gold);
  const eqL2 = liveEquation("\\lambda_2\\;=", theme, theme.gold);
  const eqC = liveEquation("c\\;=", theme, theme.cream);
  overlay.append(eqL1.node, eqL2.node, eqC.node);
  mainHost.append(overlay);

  // top-right: the matrix itself and the level-set equation, restated live
  const matOverlay = overlayLayer("tr");
  const eqMat = staticEquation("", theme, theme.blue);
  matOverlay.append(
    staticEquation("\\tfrac12\\,\\delta^{\\top} I\\,\\delta = c", theme, theme.gold),
    eqMat,
    staticEquation(`\\delta=\\sqrt{n}\\,(\\hat\\theta-\\theta),\\ n=${n}`, theme, theme.muted),
  );
  mainHost.append(matOverlay);

  const rc = responsiveCanvas(mainHost, 1.3, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:8px 18px;align-items:center;",
  });
  const i11Row = sliderRow(theme, "I₁₁", 0.04, 1.2, 0.01, i11, "Fisher entry I one one (curvature in mu)");
  const i22Row = sliderRow(theme, "I₂₂", 0.01, 0.4, 0.005, i22, "Fisher entry I two two (curvature in sigma squared)");
  const i12Row = sliderRow(theme, "I₁₂", -0.18, 0.18, 0.005, i12, "Fisher off-diagonal I one two (coupling)");
  controls.append(i11Row.row, i22Row.row, i12Row.row);
  root.append(controls);

  const buttons = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag to grow c · ←→ · sliders set I · reseed";
  buttons.append(reseedBtn, hint);
  root.append(buttons);

  target.append(root);

  // --- the one eased scalar: the level constant c --------------------------
  let c = clamp(options.c ?? C95, 0.4, 9);
  const cTracker = valueTracker(c, (v) => {
    c = clamp(v, 0.4, 9);
    render();
  });

  // --- render --------------------------------------------------------------
  function render(): void {
    decompose();
    draw();

    eqL1.set(l1.toFixed(3));
    eqL2.set(l2.toFixed(3));
    eqC.set(c.toFixed(2));
    eqMat.innerHTML = staticEquation(
      `I=\\begin{pmatrix}${fmt(i11)} & ${fmt(i12)}\\\\ ${fmt(i12)} & ${fmt(i22)}\\end{pmatrix}`,
      theme,
      theme.blue,
    ).innerHTML;

    if (document.activeElement !== i11Row.input) i11Row.set(i11);
    if (document.activeElement !== i22Row.input) i22Row.set(i22);
    if (document.activeElement !== i12Row.input) i12Row.set(i12);
  }

  // Semi-axis along eigenvector vᵢ of the level set ½δᵀIδ = c: length √(2c/λᵢ).
  // Large λ ⇒ short axis (sharply pinned); small λ ⇒ long axis (flat, uncertain).
  const semiAxes = (): [number, number] => [Math.sqrt((2 * c) / l1), Math.sqrt((2 * c) / l2)];

  // Fit the view to the larger semi-axis (plus headroom), recomputed each frame
  // so the ellipse stays on-screen as the matrix and c vary; the cloud (covariance
  // I⁻¹) is a touch wider than the 95% ring, so allow a little more room.
  const currentHalf = (): number => {
    const [sa1, sa2] = semiAxes();
    return Math.max(sa1, sa2, 2 / Math.sqrt(Math.min(l1, l2))) * 1.18;
  };

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    const half = currentHalf();
    const view = fitView(W, H, half);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);
    numberPlane(ctx, view, theme, { step: niceStep(half) });

    // --- asymptotic-MLE scatter: z pushed through I^{-1/2} --------------------
    // On the rescaled-error plane δ = √n(θ̂−θ), the limit law is √n(θ̂−θ) → 𝒩(0,
    // I⁻¹); sampling x = I^{-1/2} z gives a cloud with exactly covariance I⁻¹ —
    // the spread the ellipse describes, the same frame as the post's figure.
    const ra1 = 1 / Math.sqrt(l1); // I^{-1/2} stretch along v1 (per unit z)
    const ra2 = 1 / Math.sqrt(l2);
    const dotR = Math.max(1.6, view.scale * 0.012);
    glow(ctx, theme.blue, 4, () => {
      for (const z of zCloud) {
        const a = z[0] * ra1;
        const b = z[1] * ra2;
        const p: Pt = [a * v1[0] + b * v2[0], a * v1[1] + b * v2[1]];
        disc(ctx, view.toScreen(p), dotR, theme.blue, 0.4);
      }
    });

    // --- the confidence ellipse ½δᵀIδ = c (the answer) ------------------------
    const [sa1, sa2] = semiAxes();
    const ringPts = ellipseWorld([0, 0], v1, v2, sa1, sa2).map((p) => view.toScreen(p));
    glow(ctx, theme.gold, 7, () => {
      polyline(ctx, ringPts, {
        color: theme.gold,
        width: 2.4,
        alpha: 0.95,
        closed: true,
        fill: hexA(theme.gold, 0.07),
      });
    });

    // --- gold eigen-axis arrows -----------------------------------------------
    drawEigenArrow(ctx, view, v1, sa1, "1");
    drawEigenArrow(ctx, view, v2, sa2, "2");

    // --- the MLE / true value at the origin -----------------------------------
    glow(ctx, theme.cream, 6, () => disc(ctx, view.toScreen([0, 0]), 3.4, theme.cream, 1));
    drawCross(ctx, view.toScreen([0, 0]), 6, theme.fg);

    // --- axis labels: δμ (horizontal), δσ² (vertical) -------------------------
    mathLabel(ctx, "δμ", [W - 14, view.cy - 8], { color: theme.tick, size: 16, align: "right" });
    mathLabel(ctx, "δσ", [view.cx + 10, 20], { color: theme.tick, size: 16, sub: "2" });
  }

  function drawEigenArrow(ctx: CanvasRenderingContext2D, view: View, v: Pt, lenWorld: number, sub: string): void {
    const tip: Pt = [v[0] * lenWorld, v[1] * lenWorld];
    glow(ctx, theme.gold, 7, () => {
      arrow(ctx, view.toScreen([0, 0]), view.toScreen(tip), { color: theme.gold, width: 3.2 });
    });
    const s = view.toScreen(tip);
    mathLabel(ctx, "v", [s[0] + (v[0] >= 0 ? 8 : -8), s[1] - 7], {
      color: theme.gold,
      size: 17,
      sub,
      align: v[0] >= 0 ? "left" : "right",
      glow: 5,
    });
  }

  function drawCross(ctx: CanvasRenderingContext2D, p: Pt, r: number, color: string): void {
    strokeLine(ctx, [p[0] - r, p[1] - r], [p[0] + r, p[1] + r], color, 1.4, 0.9);
    strokeLine(ctx, [p[0] - r, p[1] + r], [p[0] + r, p[1] - r], color, 1.4, 0.9);
  }

  // --- interaction ---------------------------------------------------------
  function setI11(v: number): void {
    i11 = clamp(v, 0.04, 1.2);
    render();
  }
  function setI22(v: number): void {
    i22 = clamp(v, 0.01, 0.4);
    render();
  }
  function setI12(v: number): void {
    i12 = clamp(v, -0.18, 0.18);
    render();
  }
  function setC(level: number, animate = false): void {
    const target = clamp(level, 0.4, 9);
    if (animate) cTracker.set(target, true);
    else cTracker.jump(target);
  }
  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    drawCloud();
    render();
  }

  i11Row.input.addEventListener("input", () => setI11(Number(i11Row.input.value)));
  i22Row.input.addEventListener("input", () => setI22(Number(i22Row.input.value)));
  i12Row.input.addEventListener("input", () => setI12(Number(i12Row.input.value)));
  reseedBtn.addEventListener("click", reseed);

  // Radial drag on the canvas grows / shrinks the level c: the cursor's distance
  // from the origin, read against the current ellipse boundary, sets c.
  const stopDrag = draggable(
    rc.canvas,
    (px, py) => {
      const view = fitView(rc.width, rc.height, currentHalf());
      const w = view.toWorld([px, py]);
      // Quadratic form q = δᵀIδ at the pointer; c = q/2 places the ellipse there.
      const q = i11 * w[0] * w[0] + 2 * i12 * w[0] * w[1] + i22 * w[1] * w[1];
      cTracker.jump(clamp(q / 2, 0.4, 9));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "The Fisher information matrix as a confidence ellipse: the quadratic form one-half delta-transpose-I-delta equals c, drawn with gold eigen-axes whose lengths are the square root of two c over each eigenvalue. The seeded scatter of rescaled maximum-likelihood errors, with asymptotic covariance I-inverse, is hugged by the ellipse. Sliders set the matrix entries; drag or arrow keys grow the level c.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setC(C95, true);
      return;
    }
    const step = e.shiftKey ? 1 : 0.25;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setC(c - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setC(c + step, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the 95% level immediately, no glide.
  if (prefersReducedMotion()) c = C95;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setI11,
    setI22,
    setI12,
    setC,
    reseed,
    destroy() {
      destroyed = true;
      cTracker.stop();
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
  const row = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:3.5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: String(step),
    value: String(value),
    style: "flex:1;min-width:80px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:4ch;text-align:right;` });
  out.textContent = value.toFixed(3);
  input.addEventListener("input", () => (out.textContent = Number(input.value).toFixed(3)));
  row.append(lab, input, out);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
      out.textContent = v.toFixed(3);
    },
  };
}

// --- helpers ---------------------------------------------------------------

/** Compact fixed-precision string for the matrix display. */
function fmt(v: number): string {
  return v.toFixed(3);
}

/** A round-number grid step roughly dividing `half` into ~5 intervals. */
function niceStep(half: number): number {
  const raw = half / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}

/** Append an alpha byte to a #rrggbb color. */
function hexA(hex: string, alpha: number): string {
  const a = Math.round(clamp(alpha, 0, 1) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
