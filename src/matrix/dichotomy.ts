import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface DichotomyOptions {
  /** Initial matrix size k (the operating point). Default 16. */
  k?: number;
  /** Smallest matrix size on the axis. Default 2. */
  kMin?: number;
  /** Largest matrix size on the axis. Default 128. */
  kMax?: number;
  theme?: Partial<Theme>;
}

export interface DichotomyApi {
  /** Set the matrix size k (the operating point); eased unless animate=false. */
  setK(k: number, animate?: boolean): void;
  /** Sweep the operating point k across the whole axis (Manim ticker). */
  play(): void;
  destroy(): void;
}

/**
 * Scene — "the structure dichotomy". For a square structured family of k×k
 * matrices the completion capacity is the dimension ratio C = N/dim𝓕: the
 * number of informative entries over the family's degrees of freedom. Plotted
 * against the size k (log capacity axis), the families split sharply into two
 * regimes about the gold boundary C = 2:
 *
 *   • linear-redundancy families — circulant (C=k), Toeplitz (C=k²/(2k−1)),
 *     low rank-2 (C=k²/[4(k−1)]) — rise without bound, roughly linearly in k;
 *   • involutive families — orthogonal (C=2k/(k−1)) and symmetric (C=2k/(k+1))
 *     — only pair entries (i,j)↔(j,i), saving one constant factor, so C → 2;
 *   • general and diagonal sit on the C = 1 baseline.
 *
 * The cream operating line marks the chosen k; every family's capacity is read
 * off it and classified live as "diverging" (above the C=2 boundary) or
 * "bounded" (converging to 2). Slide / drag / sweep k across the axis and watch
 * the curves separate — the phase transition is the C=2 line, and which side a
 * family lands on is fixed by whether its redundancy is linear or involutive.
 *
 * Ported verbatim from the post's notebook dichotomy cell (§5 `family(name, k)`
 * with n = k): general k²/k², diagonal k/k, rank2 k²/[2(2k−2)], toeplitz
 * k²/(2k−1), circulant k²/k, symmetric k²/[k(k+1)/2], orthogonal k²/[k(k−1)/2].
 */
export function createMatrixDichotomy(
  target: HTMLElement,
  options: DichotomyOptions = {},
): DichotomyApi {
  const theme = withTheme(options.theme);
  const kMin = clamp(Math.round(options.kMin ?? 2), 2, 8);
  const kMax = clamp(Math.round(options.kMax ?? 128), 16, 512);
  let k = clamp(Math.round(options.k ?? 16), kMin, kMax);

  // --- the structured families (notebook §5 `family`, square so n = k) -------
  // C = N / dim𝓕. Linear-redundancy families diverge; involutive ones → 2.
  const families: Family[] = [
    { key: "circulant", label: "circulant", color: theme.blue, dash: [], regime: "diverging", cap: (kk) => kk },
    { key: "toeplitz", label: "Toeplitz", color: theme.green, dash: [], regime: "diverging", cap: (kk) => (kk * kk) / (2 * kk - 1) },
    { key: "rank2", label: "rank 2", color: theme.red, dash: [], regime: "diverging", cap: (kk) => (kk * kk) / (4 * (kk - 1)) },
    { key: "orthogonal", label: "orthogonal", color: theme.gold, dash: [6, 5], regime: "bounded", cap: (kk) => (2 * kk) / (kk - 1) },
    { key: "symmetric", label: "symmetric", color: theme.cream, dash: [6, 5], regime: "bounded", cap: (kk) => (2 * kk) / (kk + 1) },
    { key: "general", label: "general / diag", color: theme.muted, dash: [2, 4], regime: "baseline", cap: () => 1 },
  ];

  // Capacity (log) axis bounds. C runs from 1 (the floor) up to the circulant's
  // value at kMax, with a little headroom; the gold C = 2 dichotomy line lives
  // in between.
  const cFloor = 1;
  const cTop = Math.max(kMax * 1.25, 8);
  const logFloor = Math.log10(cFloor); // 0
  const logTop = Math.log10(cTop);
  const C_STAR = 2; // the dichotomy boundary

  // A dense log-spaced sampling of k for the smooth curves.
  const SAMPLES = 200;
  const sampleK: number[] = Array.from({ length: SAMPLES }, (_, i) => {
    const t = i / (SAMPLES - 1);
    return Math.pow(10, Math.log10(kMin) + t * (Math.log10(kMax) - Math.log10(kMin)));
  });

  // --- DOM -------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  // top-left: the operating point and how many families are on each side
  const overlay = overlayLayer("tl");
  const eqK = liveEquation("k\\;=", theme, theme.cream);
  const eqDiverge = liveEquation("C>2\\;\\text{(linear)}:", theme, theme.blue);
  const eqBounded = liveEquation("C\\to2\\;\\text{(involutive)}:", theme, theme.gold);
  overlay.append(eqK.node, eqDiverge.node, eqBounded.node);
  mainHost.append(overlay);

  // top-right: the capacity-as-dimension-count definition
  const defOverlay = overlayLayer("tr");
  defOverlay.append(
    staticEquation("C=\\dfrac{N_{\\text{informative}}}{\\dim\\mathcal{F}}", theme, theme.blue),
    staticEquation("\\text{boundary}\\;\\;C=2", theme, theme.gold),
  );
  mainHost.append(defOverlay);

  const rc = responsiveCanvas(mainHost, 1.5, () => render());

  // --- controls --------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "matrix size  k";
  // The slider is in log-k so each decade gets equal travel, like the plot.
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "1000",
    step: "1",
    value: String(Math.round((kToLog(k) * 1000))),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "matrix size k on a logarithmic scale; the operating point swept across the structure families",
  });
  const kOut = el("output", { style: `font:14px ${ROMAN_FONT};color:${theme.fg};min-width:4ch;` });
  kOut.textContent = String(k);
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · sweep · ←→";
  controls.append(playBtn, label, slider, kOut, hint);
  root.append(controls);

  target.append(root);

  // --- state -----------------------------------------------------------------
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  // The displayed k (continuous, in log space) eases toward its target so every
  // interaction glides; the readout snaps to the nearest integer size.
  const kTracker = valueTracker(k, (v) => {
    k = clamp(v, kMin, kMax);
    render();
  });

  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  // --- render ----------------------------------------------------------------
  function render(): void {
    draw();

    const kr = Math.round(k);
    eqK.set(String(kr));
    // classify each family at the current k against the C = 2 boundary
    let nDiv = 0;
    let nBnd = 0;
    for (const f of families) {
      if (f.regime === "baseline") continue;
      if (f.cap(k) > C_STAR + 1e-9) nDiv++;
      else nBnd++;
    }
    eqDiverge.set(String(nDiv));
    eqBounded.set(String(nBnd));

    if (document.activeElement !== slider) slider.value = String(Math.round(kToLog(k) * 1000));
    kOut.textContent = String(kr);
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 16;
    const mt = 16;
    const mb = 34;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;

    // x: log10 k → screen; y: log10 C → screen.
    const X = (kk: number) =>
      ml + ((Math.log10(clamp(kk, kMin, kMax)) - Math.log10(kMin)) / (Math.log10(kMax) - Math.log10(kMin))) * plotW;
    const Y = (cc: number) => mt + plotH - ((Math.log10(clamp(cc, cFloor, cTop)) - logFloor) / (logTop - logFloor)) * plotH;

    // --- the two phase regions, shaded ---------------------------------------
    // Above C = 2: the linear-redundancy regime (curves diverge). Below, down to
    // C = 1: the involutive band where the bounded families live and converge.
    const yStar = Y(C_STAR);
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = theme.blue;
    ctx.fillRect(ml, mt, plotW, yStar - mt); // C > 2 region
    ctx.fillStyle = theme.gold;
    ctx.fillRect(ml, yStar, plotW, mt + plotH - yStar); // 1 ≤ C ≤ 2 region
    ctx.globalAlpha = 1;

    // faint number plane (the cyan grid, receding): vertical rules at powers of
    // two in k, horizontal rules at decades of C.
    for (let p = Math.ceil(Math.log2(kMin)); Math.pow(2, p) <= kMax + 1e-9; p++) {
      const xx = X(Math.pow(2, p));
      strokeLine(ctx, [xx, mt], [xx, mt + plotH], theme.grid, 1, 0.18);
    }
    for (const c of decadeTicks(cFloor, cTop)) {
      const yy = Y(c);
      strokeLine(ctx, [ml, yy], [W - mr, yy], theme.grid, 1, c === cFloor ? 0.26 : 0.14);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.5, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.5, 1);

    // --- the C = 2 dichotomy boundary (the gold "answer") --------------------
    glow(ctx, theme.gold, 6, () => {
      polyline(ctx, [[ml, yStar], [W - mr, yStar]], { color: theme.gold, width: 2, alpha: 0.95, dash: [7, 6] });
    });
    mathLabel(ctx, "C", [ml + 6, yStar - 6], { color: theme.gold, size: 14, sub: "*", glow: 4 });
    ctx.fillStyle = theme.gold;
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("= 2", W - mr - 2, yStar - 4);

    // region tags
    ctx.fillStyle = theme.blue;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.globalAlpha = 0.85;
    ctx.fillText("linear redundancy  ·  C → ∞", ml + 8, mt + 6);
    ctx.fillStyle = theme.gold;
    ctx.textBaseline = "bottom";
    ctx.fillText("involutive  ·  C → 2     baseline  ·  C = 1", ml + 8, mt + plotH - 6);
    ctx.globalAlpha = 1;

    // --- the family curves ---------------------------------------------------
    for (const f of families) {
      const pts: Pt[] = sampleK.map((kk) => [X(kk), Y(f.cap(kk))]);
      glow(ctx, f.color, f.regime === "baseline" ? 0 : 3, () => {
        polyline(ctx, pts, { color: f.color, width: f.regime === "baseline" ? 1.6 : 2.2, alpha: 0.92, dash: f.dash });
      });
    }

    // --- the operating point: a cream vertical line at the chosen k ----------
    const opx = X(k);
    glow(ctx, theme.cream, 5, () => {
      strokeLine(ctx, [opx, mt], [opx, mt + plotH], theme.cream, 1.4, 0.7);
    });

    // read each family off the operating line; mark the dot, colour-tagged by
    // which side of the C = 2 boundary it lands on.
    for (const f of families) {
      const cv = f.cap(k);
      const py = Y(cv);
      const onTop = cv > C_STAR + 1e-9;
      glow(ctx, f.color, 5, () => disc(ctx, [opx, py], f.regime === "baseline" ? 2.6 : 3.4, f.color, 1));
      // a tiny "+" above / "−" below tick to signal the side, only for the
      // structured families (the baseline is unambiguous)
      if (f.regime !== "baseline") {
        const side = onTop ? "diverging" : "→2";
        ctx.fillStyle = f.color;
        ctx.font = `10px ${ROMAN_FONT}`;
        ctx.textAlign = k > geoMid(kMin, kMax) ? "right" : "left";
        ctx.textBaseline = "middle";
        const dx = k > geoMid(kMin, kMax) ? -7 : 7;
        ctx.fillText(`${f.label} ${side}`, opx + dx, py);
      }
    }

    // k label on the operating line
    mathLabel(ctx, "k", [opx + (k > geoMid(kMin, kMax) ? -8 : 8), mt + 12], {
      color: theme.cream,
      size: 15,
      align: k > geoMid(kMin, kMax) ? "right" : "left",
      glow: 5,
    });

    // --- axis labels (KaTeX serif) and ticks (roman numerals) ----------------
    mathLabel(ctx, "k", [W - mr - 4, mt + plotH + 22], { color: theme.tick, size: 16, align: "right" });
    mathLabel(ctx, "C", [ml - 32, mt + 10], { color: theme.tick, size: 16 });

    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let p = Math.ceil(Math.log2(kMin)); Math.pow(2, p) <= kMax + 1e-9; p++) {
      const kk = Math.pow(2, p);
      ctx.fillText(String(kk), X(kk), mt + plotH + 6);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const c of decadeTicks(cFloor, cTop)) ctx.fillText(fmtTick(c), ml - 6, Y(c));

    drawLegend(ctx, W - mr - 8, mt + 52);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = top;
    for (const f of families) {
      const lx = right - 118;
      ctx.strokeStyle = f.color;
      ctx.lineWidth = f.regime === "baseline" ? 1.6 : 2.2;
      ctx.setLineDash(f.dash);
      ctx.beginPath();
      ctx.moveTo(lx, yy);
      ctx.lineTo(lx + lineW, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = theme.fg;
      ctx.fillText(f.label, lx + lineW + 6, yy + 0.5);
      yy += 15;
    }
    ctx.restore();
  }

  // --- interaction -----------------------------------------------------------
  function setK(nk: number, animate = false): void {
    stopSweep();
    const t = clamp(nk, kMin, kMax);
    if (animate) kTracker.set(t, true);
    else kTracker.jump(t);
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setK(kMax, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    // traverse k across the whole (log) axis, left to right, then settle at kMax
    let s = (Math.log10(k) - Math.log10(kMin)) / (Math.log10(kMax) - Math.log10(kMin));
    sweepCancel = ticker((dt) => {
      s += dt / 5;
      if (s >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        kTracker.jump(kMax);
        return false;
      }
      const kk = Math.pow(10, Math.log10(kMin) + s * (Math.log10(kMax) - Math.log10(kMin)));
      kTracker.jump(kk);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; sweep k";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  slider.addEventListener("input", () => {
    stopSweep();
    kTracker.jump(logToK(Number(slider.value) / 1000));
  });

  // Horizontal drag on the canvas scrubs k along the log axis.
  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      stopSweep();
      const ml = 46;
      const mr = 16;
      const plotW = rc.width - ml - mr;
      const t = clamp((px - ml) / plotW, 0, 1);
      kTracker.jump(Math.pow(10, Math.log10(kMin) + t * (Math.log10(kMax) - Math.log10(kMin))));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "The structure dichotomy: completion capacity C equals the number of informative entries over the family's degrees of freedom, plotted against matrix size k on a logarithmic capacity axis. Linear-redundancy families — circulant, Toeplitz and low rank — rise without bound above the gold C equals two boundary, while involutive families — orthogonal and symmetric — converge to two, and the general and diagonal families sit at one. Slide, drag, or sweep the matrix size and read each family off the operating line, classified by which side of the C equals two boundary it lands on.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      setK(kMin, true);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setK(kMax, true);
      return;
    }
    // step by a multiplicative factor so travel feels even on the log axis
    const factor = e.shiftKey ? 2 : Math.pow(2, 0.25);
    let nk = k;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") nk = k / factor;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") nk = k * factor;
    else return;
    e.preventDefault();
    setK(nk, true);
  };
  rc.canvas.addEventListener("keydown", onKey);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setK,
    play,
    destroy() {
      destroyed = true;
      kTracker.stop();
      sweepCancel?.();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };

  // --- log-axis mapping for the slider (closures over kMin/kMax) -------------
  function kToLog(kk: number): number {
    return (Math.log10(clamp(kk, kMin, kMax)) - Math.log10(kMin)) / (Math.log10(kMax) - Math.log10(kMin));
  }
  function logToK(t: number): number {
    return Math.pow(10, Math.log10(kMin) + clamp(t, 0, 1) * (Math.log10(kMax) - Math.log10(kMin)));
  }
}

// --- the structured families -----------------------------------------------

interface Family {
  key: string;
  label: string;
  color: string;
  dash: number[];
  /** Which side of the C = 2 boundary the family ends up on as k grows. */
  regime: "diverging" | "bounded" | "baseline";
  /** Completion capacity C = N/dim𝓕 at square size k (notebook §5 `family`). */
  cap: (k: number) => number;
}

// --- helpers ----------------------------------------------------------------

/** Geometric midpoint of [a,b] — the log-axis centre, for label-side flipping. */
function geoMid(a: number, b: number): number {
  return Math.sqrt(a * b);
}

/** Decade tick positions (1, 2, 5 per decade) within [lo, hi]. */
function decadeTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const start = Math.floor(Math.log10(lo));
  const end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo - 1e-9 && v <= hi + 1e-9) out.push(v);
    }
  }
  return out;
}

/** Compact tick label (integers without a trailing .0). */
function fmtTick(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(1);
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
