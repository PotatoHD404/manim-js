import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface CommonBitOptions {
  /** Row counts whose capacity curves are drawn. Defaults to the post's [2,3,4,6]. */
  ks?: number[];
  /** Which k-curve is highlighted / read off (must be one of `ks`). Default 4. */
  active?: number;
  /** Initial private-noise level q ∈ [0, 1/2]. Default 0.1. */
  q?: number;
  theme?: Partial<Theme>;
}

export interface CommonBitApi {
  /** Move the private-noise cursor q ∈ [0, 1/2] (eased when `animate`). */
  setNoise(q: number, animate?: boolean): void;
  /** Highlight the capacity curve for `k` rows. */
  setActiveK(k: number): void;
  /** Sweep q from clean (0) to fully noisy (1/2), watching the shared bit wash out. */
  play(): void;
  destroy(): void;
}

/**
 * Scene — "the shared bit and its capacity". The common-bit column law from the
 * post: every row is the same latent bit B ~ Bern(½) seen through its own
 * private Bern(q) noise, Xℓ = B ⊕ Zℓ. The marginals stay uniform (S = k), so the
 * completion capacity is C = k / H(X₁,…,X_k), and the redundancy the rows share
 * is the total correlation TC = k − H(X₁,…,X_k) — the "common bits".
 *
 * The main panel plots C against the private-noise level q ∈ [0, ½] for several
 * row counts k; each curve falls from C = k at q = 0 (a clean shared bit, marked
 * by a gold dotted rule) to C = 1 at q = ½ (the bit is washed out, the rows are
 * independent). A cream cursor is the movable variable q: drag it, slide it, or
 * arrow-key it and the active curve's C, joint entropy, and shared bits TC update
 * live, with a small meter tracing the common information as it grows toward the
 * clean limit. Reducing the noise raises the correlation, and the shared bits
 * climb — capacity is monotone in redundancy.
 *
 * Ported verbatim from the notebook's `common_bit(k, q)` / `capacity(joint)`
 * cell: H(joint) is summed over Hamming-weight classes with
 * P_w = ½·(q^w(1−q)^{k−w} + q^{k−w}(1−q)^w), which reproduces the brute-force
 * 2^k enumeration to machine precision, and S = k since every marginal is
 * uniform. C(q=0) = k and C(q=½) = 1 exactly, as in the figure.
 */
export function createCommonBit(target: HTMLElement, options: CommonBitOptions = {}): CommonBitApi {
  const theme = withTheme(options.theme);
  const ks = (options.ks ?? [2, 3, 4, 6]).map((k) => Math.max(2, Math.round(k)));
  const kMax = Math.max(...ks);
  let activeK = ks.includes(Math.round(options.active ?? 4)) ? Math.round(options.active ?? 4) : ks[ks.length - 1];
  let q = clamp(options.q ?? 0.1, 0, 0.5);
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  // Per-k accent colors for the curve family; the active curve is drawn bright,
  // the rest receding like the cyan number plane.
  const kColor = (k: number): string => {
    const order = ks.indexOf(k);
    const palette = [theme.blue, theme.green, theme.gold, theme.red, "#9b8cff", "#e08fc0"];
    return palette[order % palette.length];
  };

  // Pre-sampled C(q) curves (the math is deterministic, so these never change).
  const SAMPLES = 241;
  const qGrid = Array.from({ length: SAMPLES }, (_, i) => (i / (SAMPLES - 1)) * 0.5);
  const curves = new Map<number, number[]>();
  for (const k of ks) curves.set(k, qGrid.map((qq) => capacityCommonBit(k, qq)));

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  const traceHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(mainHost, traceHost);
  root.append(panel);

  // top-left: the live readouts (capacity = the answer; shared bits = the traced quantity)
  const overlay = overlayLayer("tl");
  const eqCap = liveEquation("C=\\dfrac{S}{H(X_1,\\dots,X_k)}\\;=", theme, theme.gold);
  const eqTc = liveEquation("\\mathrm{TC}=S-H(\\mathbf{X})\\;=", theme, theme.green);
  const eqQ = liveEquation("q\\;=", theme, theme.cream);
  overlay.append(eqCap.node, eqTc.node, eqQ.node);
  mainHost.append(overlay);

  // top-right: the column law, restated
  const lawOverlay = overlayLayer("tr");
  lawOverlay.append(
    staticEquation("X_\\ell = B \\oplus Z_\\ell,\\quad Z_\\ell\\sim\\mathrm{Bern}(q)", theme, theme.muted),
    staticEquation("S=\\textstyle\\sum_\\ell H(X_\\ell)=k", theme, theme.blue),
  );
  mainHost.append(lawOverlay);

  const traceTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.green};font:12px ${theme.mono};`,
  });
  traceTag.textContent = "shared bits  TC  vs.  q   (correlation grows as noise falls)";
  traceHost.append(traceTag);

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "500",
    step: "1",
    value: String(Math.round((q / 0.5) * 500)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "private-noise level q from 0 to one half",
  });
  const kLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  kLabel.textContent = "rows k";
  const kGroup = el("div", { style: "display:flex;gap:6px;" });
  const kButtons = new Map<number, HTMLButtonElement>();
  for (const k of ks) {
    const b = el("button", { type: "button", style: kBtnStyle(theme, kColor(k)) }) as HTMLButtonElement;
    b.textContent = String(k);
    b.addEventListener("click", () => setActiveK(k));
    kButtons.set(k, b);
    kGroup.append(b);
  }
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · sweep · ←→";
  controls.append(playBtn, slider, kLabel, kGroup, hint);
  root.append(controls);

  target.append(root);

  const rcMain = responsiveCanvas(mainHost, 1.62, () => render());
  const rcTrace = responsiveCanvas(traceHost, 5.0, () => render());

  // The displayed noise eases toward its target, so every interaction glides.
  const qTracker = valueTracker(q, (v) => {
    q = clamp(v, 0, 0.5);
    render();
  });

  // --- math ----------------------------------------------------------------
  function activeCurve(): number[] {
    return curves.get(activeK) ?? [];
  }
  function capNow(): number {
    return capacityCommonBit(activeK, q);
  }
  function hJointNow(): number {
    return hJointCommonBit(activeK, q);
  }
  function tcNow(): number {
    return activeK - hJointNow();
  }

  // --- render --------------------------------------------------------------
  function render(): void {
    drawMain();
    drawTrace();
    eqCap.set(capNow().toFixed(3));
    eqTc.set(tcNow().toFixed(3));
    eqQ.set(q.toFixed(3));
    for (const [k, b] of kButtons) b.style.outline = k === activeK ? `2px solid ${kColor(k)}` : "none";
    if (document.activeElement !== slider) slider.value = String(Math.round((q / 0.5) * 500));
  }

  function drawMain(): void {
    const { ctx, width: W, height: H } = rcMain;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 44;
    const mr = 18;
    const mt = 16;
    const mb = 32;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const cTop = kMax * 1.06; // top of the C axis (just above C = kMax)
    const x = (qq: number) => ml + clamp(qq / 0.5, 0, 1) * plotW;
    const y = (c: number) => mt + plotH - (clamp(c, 0, cTop) / cTop) * plotH;

    // faint number-plane: vertical q gridlines + horizontal integer-C lines
    for (let i = 0; i <= 5; i++) {
      const gx = ml + (i / 5) * plotW;
      strokeLine(ctx, [gx, mt], [gx, mt + plotH], theme.grid, 1, i === 0 ? 0.28 : 0.14);
    }
    for (let c = 1; c <= kMax; c++) {
      strokeLine(ctx, [ml, y(c)], [W - mr, y(c)], theme.grid, 1, 0.12);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    // the noiseless C = k rules (gold dotted), the answer values each curve heads to
    for (const k of ks) {
      const isActive = k === activeK;
      polyline(ctx, [[ml, y(k)], [W - mr, y(k)]], {
        color: theme.gold,
        width: 1,
        alpha: isActive ? 0.6 : 0.22,
        dash: [4, 6],
      });
    }
    // the C = 1 floor (full independence at q = 1/2)
    polyline(ctx, [[ml, y(1)], [W - mr, y(1)]], { color: theme.muted, width: 1, alpha: 0.4, dash: [2, 5] });

    // the capacity curves: inactive recede, the active one is bright and glows
    for (const k of ks) {
      if (k === activeK) continue;
      const pts: Pt[] = (curves.get(k) ?? []).map((c, i) => [x(qGrid[i]), y(c)]);
      polyline(ctx, pts, { color: kColor(k), width: 1.5, alpha: 0.3 });
    }
    const aPts: Pt[] = activeCurve().map((c, i) => [x(qGrid[i]), y(c)]);
    glow(ctx, kColor(activeK), 5, () => {
      polyline(ctx, aPts, { color: kColor(activeK), width: 2.6, alpha: 1 });
    });

    // curve labels at the clean end (q = 0, C = k)
    for (const k of ks) {
      mathLabel(ctx, "k", [x(0) + 6, y(k) - 5], {
        color: kColor(k),
        size: k === activeK ? 16 : 13,
        sub: String(k),
        align: "left",
        glow: k === activeK ? 5 : 0,
      });
    }

    // the movable variable q: a vertical cream cursor reading the active curve
    const cx = x(q);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.3, 0.55);
    const cNow = capNow();
    glow(ctx, theme.cream, 8, () => disc(ctx, [cx, y(cNow)], 4.2, theme.cream, 1));
    // a faint tick on every other curve at this q, so the spread across k reads
    for (const k of ks) {
      if (k === activeK) continue;
      disc(ctx, [cx, y(capacityCommonBit(k, q))], 2.4, kColor(k), 0.85);
    }
    mathLabel(ctx, "q", [cx + (q < 0.42 ? 7 : -7), mt + 13], {
      color: theme.cream,
      size: 15,
      align: q < 0.42 ? "left" : "right",
      glow: 5,
    });

    // axis labels (KaTeX serif) + ticks (roman numerals)
    mathLabel(ctx, "C", [ml - 30, mt + 4], { color: theme.tick, size: 16 });
    mathLabel(ctx, "q", [W - mr, mt + plotH + 22], { color: theme.tick, size: 16, align: "right" });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= 5; i++) {
      const qq = (i / 5) * 0.5;
      ctx.fillText(qq.toFixed(2), ml + (i / 5) * plotW, mt + plotH + 5);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let c = 1; c <= kMax; c++) ctx.fillText(String(c), ml - 6, y(c));
  }

  function drawTrace(): void {
    const { ctx, width: W, height: H } = rcTrace;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 16;
    const mr = 14;
    const mt = 26;
    const mb = 22;
    const tcTop = (activeK - 1) * 1.08 || 1; // TC ranges over [0, k-1]
    // The x-axis here is rising correlation: noise q runs RIGHT→LEFT so that
    // "more correlation" reads left-to-right, matching the shared bits growing.
    const x = (qq: number) => ml + (1 - qq / 0.5) * (W - ml - mr);
    const y = (tc: number) => H - mb - (clamp(tc, 0, tcTop) / tcTop) * (H - mt - mb);

    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.2, 1);
    strokeLine(ctx, [ml, H - mb], [ml, mt], theme.axis, 1.2, 1);

    // full TC(q) curve faint, the swept-in (clean) part up to the cursor bright
    const tcCurve: Pt[] = qGrid.map((qq) => [x(qq), y(activeK - hJointCommonBit(activeK, qq))]);
    polyline(ctx, tcCurve, { color: theme.green, width: 1.6, alpha: 0.32 });
    // bright from the noisy end (q=0.5) down to the current q (the cleaner side)
    const bright = tcCurve.filter((_, i) => qGrid[i] >= q);
    if (bright.length > 1) polyline(ctx, bright, { color: theme.green, width: 2.4, alpha: 1 });

    // the clean-limit marker: TC = k - 1 at q = 0 (the answer the curve climbs to)
    strokeLine(ctx, [x(0), y(0)], [x(0), y(activeK - 1)], theme.gold, 1, 0.5);
    disc(ctx, [x(0), y(activeK - 1)], 3, theme.gold, 1);

    // current q marker
    const tcNowV = activeK - hJointCommonBit(activeK, q);
    strokeLine(ctx, [x(q), H - mb], [x(q), y(tcNowV)], theme.cream, 1, 0.5);
    glow(ctx, theme.cream, 6, () => disc(ctx, [x(q), y(tcNowV)], 3.4, theme.cream, 1));

    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const d of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) ctx.fillText(d.toFixed(1), x(d), H - mb + 4);
  }

  // --- interaction ---------------------------------------------------------
  function setNoise(qq: number, animate = false): void {
    stopSweep();
    const t = clamp(qq, 0, 0.5);
    if (animate) qTracker.set(t, true);
    else qTracker.jump(t);
  }

  function setActiveK(k: number): void {
    if (!ks.includes(k)) return;
    activeK = k;
    render();
  }

  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      qTracker.jump(0.5);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    let s = q / 0.5;
    sweepCancel = ticker((dt) => {
      s += dt / 4.5;
      if (s >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        qTracker.jump(0.5);
        return false;
      }
      qTracker.jump(s * 0.5);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; add noise";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  slider.addEventListener("input", () => {
    stopSweep();
    qTracker.jump((Number(slider.value) / 500) * 0.5);
  });

  // Horizontal drag on the main canvas scrubs q.
  const stopDrag = draggable(
    rcMain.canvas,
    (px) => {
      stopSweep();
      const ml = 44;
      const mr = 18;
      const plotW = rcMain.width - ml - mr;
      qTracker.jump(clamp((px - ml) / plotW, 0, 1) * 0.5);
    },
    { onStart: () => (rcMain.canvas.style.cursor = "grabbing"), onEnd: () => (rcMain.canvas.style.cursor = "grab") },
  );
  rcMain.canvas.style.cursor = "grab";

  rcMain.canvas.tabIndex = 0;
  rcMain.canvas.setAttribute("role", "img");
  rcMain.canvas.setAttribute(
    "aria-label",
    "Common-bit model: each row is a shared latent bit seen through private Bernoulli-q noise. The completion capacity C is plotted against the noise level q for several row counts k; every curve falls from C equal to k at zero noise to C equal to one at noise one half. A cursor reads the active curve, and a meter below traces the shared bits, the total correlation, which grows as the noise falls and the correlation rises.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const step = e.shiftKey ? 0.02 : 0.005;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setNoise(q - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setNoise(q + step, true);
    else if (e.key === "Home") setNoise(0, true);
    else if (e.key === "End") setNoise(0.5, true);
    else return;
    e.preventDefault();
  };
  rcMain.canvas.addEventListener("keydown", onKey);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setNoise,
    setActiveK,
    play,
    destroy() {
      destroyed = true;
      qTracker.stop();
      sweepCancel?.();
      stopDrag();
      rcMain.canvas.removeEventListener("keydown", onKey);
      rcMain.destroy();
      rcTrace.destroy();
      root.remove();
    },
  };
}

// --- math: ported from the notebook's common_bit(k,q) / capacity(joint) ------

/**
 * Joint entropy H(X₁,…,X_k) of the common-bit law, in bits. The notebook
 * enumerates all 2^k outcomes; here we group by Hamming weight w, since every
 * outcome of weight w shares the same probability
 *   P_w = ½·(q^w (1−q)^{k−w} + q^{k−w} (1−q)^w),
 * and there are C(k,w) of them. This reproduces the brute-force value exactly
 * while staying O(k).
 */
function hJointCommonBit(k: number, q: number): number {
  let h = 0;
  let comb = 1; // C(k, 0)
  for (let w = 0; w <= k; w++) {
    const a = Math.pow(q, w) * Math.pow(1 - q, k - w);
    const b = Math.pow(q, k - w) * Math.pow(1 - q, w);
    const p = 0.5 * (a + b);
    if (p > 0) h += -comb * p * Math.log2(p);
    comb = (comb * (k - w)) / (w + 1); // C(k, w+1)
  }
  return h;
}

/**
 * Completion capacity C = S / H_joint. Every marginal Xℓ is uniform Bernoulli,
 * so S = Σℓ H(Xℓ) = k; the capacity is therefore k / H(X₁,…,X_k). At q = 0 the
 * joint entropy is 1 bit (the shared bit), giving C = k; at q = ½ the rows are
 * independent uniform bits, H_joint = k and C = 1.
 */
function capacityCommonBit(k: number, q: number): number {
  const hj = hJointCommonBit(k, q);
  return hj < 1e-12 ? Number.POSITIVE_INFINITY : k / hj;
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}

function kBtnStyle(theme: Theme, color: string): string {
  return `font:13px ${theme.mono};color:${color};background:#1e1e1e;border:1px solid #555555;border-radius:7px;padding:5px 10px;min-width:30px;cursor:pointer;`;
}
