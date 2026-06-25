import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { texInto } from "../core/katex";
import { type Theme, withTheme } from "../core/theme";

export interface ZeroCountOptions {
  /** Category counts; one of them should be zero to show the problem. */
  counts?: number[];
  /** Category labels (one per count). */
  labels?: string[];
  /** Symmetric Dirichlet concentration α (per category). α=1 → MLE, α=2 → add-one. */
  alpha?: number;
  /** Maximum count a single category can reach via slider/drag. */
  maxCount?: number;
  theme?: Partial<Theme>;
}

export interface ZeroCountApi {
  /** Set the symmetric Dirichlet concentration α. */
  setAlpha(alpha: number, animate?: boolean): void;
  /** Set the integer count of one category. */
  setCount(index: number, count: number, animate?: boolean): void;
  /** Focus a category for keyboard interaction. */
  select(index: number): void;
  destroy(): void;
}

/**
 * Scene — the zero-count problem. The bars are the categorical probability
 * estimates: the cool MLE `π̂ᵢ = xᵢ/n` (which collapses to *exactly zero* for an
 * unseen category) versus the gold MAP estimate under a symmetric Dirichlet(α)
 * prior, `π̂ᵢ = (xᵢ+α−1)/(n+m(α−1))`, which stays strictly positive. Slide the
 * prior strength α from 1 (pure MLE) up through 2 (add-one / Laplace) and watch
 * the gold lollipop lift the unseen category off the floor — while the readout
 * shows the future-observation log-likelihood going from −∞ (MLE) to finite
 * (MAP). Drag any bar to change its count, including dragging the zero category
 * back up.
 */
export function createZeroCount(target: HTMLElement, options: ZeroCountOptions = {}): ZeroCountApi {
  const theme = withTheme(options.theme);
  const labels = (options.labels ?? ["A", "B", "C", "D"]).slice();
  const maxCount = options.maxCount ?? 16;
  const m = labels.length;

  // State: integer counts (the data), and the symmetric prior strength α.
  const counts = (options.counts ?? [8, 5, 0, 3]).slice(0, m).map((c) => clamp(Math.round(c), 0, maxCount));
  while (counts.length < m) counts.push(0);
  let alpha = clamp(options.alpha ?? 2, 1, 6);
  // The unseen category is the one that starts at zero (falls back to the last).
  let focus = counts.indexOf(0);
  if (focus < 0) focus = m - 1;

  let destroyed = false;

  // --- the math (ported verbatim from the notebook's zero-count cell) -------
  const n = () => counts.reduce((a, c) => a + c, 0);
  // MLE: observed frequency. Unseen category → exactly 0.
  const mle = (i: number): number => {
    const N = n();
    return N > 0 ? counts[i] / N : 0;
  };
  // MAP under symmetric Dirichlet(α): (xᵢ+α−1)/(n+m(α−1)).
  // α=1 reproduces the MLE; α=2 is add-one (Laplace) smoothing (xᵢ+1)/(n+m).
  const map = (i: number, a = alpha): number => {
    const denom = n() + m * (a - 1);
    return denom > 0 ? (counts[i] + a - 1) / denom : 1 / m;
  };

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // overlay: the MAP formula with the live α, and the kept legend
  const overlay = overlayLayer("tl");
  const eqAlpha = liveEquation("\\alpha\\;=", theme, theme.gold);
  overlay.append(
    staticEquation("\\hat\\pi_i^{\\,\\mathrm{MLE}}=\\dfrac{x_i}{n}", theme, theme.blue),
    staticEquation("\\hat\\pi_i^{\\,\\mathrm{MAP}}=\\dfrac{x_i+\\alpha-1}{\\,n+m(\\alpha-1)\\,}", theme, theme.gold),
    eqAlpha.node,
  );
  panel.append(overlay);

  // top-right: future-observation log-likelihood for the unseen category
  const llOverlay = overlayLayer("tr");
  const llTitle = el("div", { style: `color:${theme.muted};font:12px ${theme.mono};line-height:1.7;` });
  const llMle = el("div", { style: "line-height:1.9;" });
  const llMap = el("div", { style: "line-height:1.9;" });
  llOverlay.append(llTitle, llMle, llMap);
  panel.append(llOverlay);

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });

  const alphaLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  alphaLabel.textContent = "prior strength  α";
  const alphaSlider = el("input", {
    type: "range",
    min: "1",
    max: "6",
    step: "0.05",
    value: String(alpha),
    style: "flex:1;min-width:150px;max-width:260px;accent-color:" + theme.gold + ";",
    "aria-label": "symmetric Dirichlet prior strength alpha",
  });
  alphaSlider.addEventListener("input", () => alphaTracker.jump(Number(alphaSlider.value)));

  const catLabel = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  catLabel.textContent = "count  xᵢ";
  const countSlider = el("input", {
    type: "range",
    min: "0",
    max: String(maxCount),
    step: "1",
    value: String(counts[focus]),
    style: "flex:1;min-width:130px;max-width:200px;accent-color:" + theme.blue + ";",
    "aria-label": "count of the selected category",
  });
  countSlider.addEventListener("input", () => setCount(focus, Number(countSlider.value), false));

  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag bars · α slider · [ ] pick · ←→";

  controls.append(alphaLabel, alphaSlider, catLabel, countSlider, hint);
  root.append(controls);
  target.append(root);

  const rc = responsiveCanvas(panel, 1.78, () => render());

  // --- animation: α is the eased main scalar; per-bar count trackers glide --
  let alphaShown = alpha;
  const alphaTracker = valueTracker(alpha, (v) => {
    alpha = v;
    alphaShown = v;
    render();
  });
  // Displayed counts ease toward the integer state so bars grow/shrink smoothly.
  const countShown = counts.slice();
  const countTrackers = counts.map((c, i) =>
    valueTracker(c, (v) => {
      countShown[i] = v;
      render();
    }),
  );

  // --- render --------------------------------------------------------------
  function render(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 46;
    const mr = 18;
    const mt = 22;
    const mb = 40;
    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const baseY = H - mb;

    // y maps a probability in [0,1] (the y-axis is "estimated probability").
    const yMax = 1;
    const y = (pr: number) => baseY - clamp(pr, 0, 1) * (plotH / yMax);

    // axes
    strokeLine(ctx, [ml, mt], [ml, baseY], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, baseY], [W - mr, baseY], theme.axis, 1.4, 1);

    // y gridlines + ticks at 0, .25, .5, .75, 1
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      const gy = y(g);
      polyline(ctx, [[ml, gy], [W - mr, gy]], { color: theme.grid, width: 1, alpha: g === 0 ? 0 : 0.14 });
      ctx.fillText(g.toFixed(2), ml - 7, gy);
    }
    ctx.save();
    ctx.translate(13, mt + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = theme.muted;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("estimated probability", 0, 0);
    ctx.restore();

    // per-category slot geometry
    const slot = plotW / m;
    const stemHalf = Math.min(slot * 0.17, 16);
    const cx = (i: number) => ml + (i + 0.5) * slot;

    // draw each category: MLE lollipop (left, blue) + MAP lollipop (right, gold)
    for (let i = 0; i < m; i++) {
      const center = cx(i);
      const xMle = center - stemHalf;
      const xMap = center + stemHalf;
      const isFocus = i === focus;
      const isUnseen = Math.round(countShown[i]) === 0;

      // focus band behind the slot
      if (isFocus) {
        ctx.fillStyle = theme.cream;
        ctx.globalAlpha = 0.05;
        ctx.fillRect(ml + i * slot, mt, slot, plotH);
        ctx.globalAlpha = 1;
      }

      // recompute estimates from the *displayed* (eased) counts so bars glide
      const Nshown = countShown.reduce((a, c) => a + c, 0);
      const prMle = Nshown > 0 ? countShown[i] / Nshown : 0;
      const denom = Nshown + m * (alphaShown - 1);
      const prMap = denom > 0 ? (countShown[i] + alphaShown - 1) / denom : 1 / m;

      // MLE lollipop — the spiky estimate; for an unseen category it is exactly
      // 0, drawn as a hollow ring on the floor (the "hole" in the model).
      lollipop(ctx, xMle, baseY, y(prMle), theme.blue, isUnseen && prMle === 0);
      // MAP lollipop — the gold "answer"; never reaches the floor for α>1.
      lollipop(ctx, xMap, baseY, y(prMap), theme.gold, false);

      // value chips above each head
      chip(ctx, xMle, y(prMle), prMle.toFixed(2), theme.blue);
      chip(ctx, xMap, y(prMap), prMap.toFixed(2), theme.gold);

      // x label: "A (8)" — label and current integer count
      const cInt = Math.round(countShown[i]);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = isFocus ? theme.cream : theme.tick;
      ctx.font = `${isFocus ? "600 " : ""}13px ${ROMAN_FONT}`;
      ctx.fillText(`${labels[i]} (${cInt})`, center, baseY + 8);
      if (cInt === 0) {
        ctx.fillStyle = theme.red;
        ctx.font = `10px ${ROMAN_FONT}`;
        ctx.fillText("unseen", center, baseY + 24);
      }
    }

    // legend (top, under the formulae) — small swatches
    legend(ctx, W - mr, baseY - plotH - 2);

    // --- live readouts ------------------------------------------------------
    eqAlpha.set(alphaShown.toFixed(2));

    // future observation in the unseen (focused) category
    const fi = focus;
    const piMle = mle(fi);
    const piMap = map(fi);
    llTitle.textContent = `future obs in ${labels[fi]}:  log-likelihood`;
    texInto(
      llMle,
      `\\color{${theme.blue}}\\mathrm{MLE}:\\ \\log\\hat\\pi_{${labels[fi]}}=` +
        (piMle <= 0 ? "-\\infty" : `\\log ${piMle.toFixed(3)}=${Math.log(piMle).toFixed(2)}`),
    );
    texInto(
      llMap,
      `\\color{${theme.gold}}\\mathrm{MAP}:\\ \\log\\hat\\pi_{${labels[fi]}}=` +
        `\\log ${piMap.toFixed(3)}=${Math.log(piMap).toFixed(2)}`,
    );

    // keep controls in sync (without fighting an in-progress drag/type)
    if (document.activeElement !== alphaSlider) alphaSlider.value = alphaShown.toFixed(2);
    if (document.activeElement !== countSlider) {
      countSlider.value = String(Math.round(countShown[focus]));
    }
  }

  /** A Manim-style lollipop: a stem from the baseline to a glowing head. */
  function lollipop(
    ctx: CanvasRenderingContext2D,
    x: number,
    y0: number,
    yHead: number,
    color: string,
    hollow: boolean,
  ): void {
    if (hollow) {
      // zero estimate: nothing rises off the floor — a hollow ring marks the hole
      glow(ctx, color, 4, () => {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(x, y0 - 1, 4.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
      return;
    }
    strokeLine(ctx, [x, y0], [x, yHead], color, 2.4, 0.9);
    glow(ctx, color, 7, () => disc(ctx, [x, yHead], 4.4, color, 1));
  }

  /** A small value chip floating just above a lollipop head. */
  function chip(ctx: CanvasRenderingContext2D, x: number, yHead: number, text: string, color: string): void {
    mathLabel(ctx, text, [x, yHead - 9], {
      color,
      size: 11,
      italic: false,
      align: "center",
      baseline: "bottom",
      glow: 4,
    });
  }

  function legend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const items: [string, string][] = [
      [theme.blue, "MLE"],
      [theme.gold, "MAP"],
    ];
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = `11px ${ROMAN_FONT}`;
    let x = right - 118;
    const yy = top + 2;
    for (const [c, name] of items) {
      glow(ctx, c, 5, () => disc(ctx, [x, yy], 4, c, 1));
      ctx.fillStyle = c;
      ctx.fillText(name, x + 8, yy);
      x += 58;
    }
  }

  // --- interaction ---------------------------------------------------------
  function setAlpha(a: number, animate = false): void {
    const v = clamp(a, 1, 6);
    if (animate) alphaTracker.set(v, true);
    else alphaTracker.jump(v);
  }

  function setCount(i: number, c: number, animate = false): void {
    if (i < 0 || i >= m) return;
    const v = clamp(Math.round(c), 0, maxCount);
    counts[i] = v;
    if (animate && !prefersReducedMotion()) countTrackers[i].set(v, true);
    else countTrackers[i].jump(v);
    // a count change alters n, so every bar moves — nudge the others to redraw
    if (!destroyed) render();
  }

  function select(i: number): void {
    focus = clamp(i, 0, m - 1);
    if (document.activeElement !== countSlider) countSlider.value = String(counts[focus]);
    render();
  }

  // map a canvas x to the category slot under the pointer
  function slotAt(px: number): number {
    const ml = 46;
    const mr = 18;
    const plotW = rc.width - ml - mr;
    const i = Math.floor(((px - ml) / plotW) * m);
    return clamp(i, 0, m - 1);
  }

  // drag: pick the slot under the pointer and set its count from the y-position
  const stopDrag = draggable(
    rc.canvas,
    (px, py) => {
      const i = slotAt(px);
      if (i !== focus) select(i);
      const mt = 22;
      const mb = 40;
      const plotH = rc.height - mt - mb;
      const baseY = rc.height - mb;
      // vertical position → probability → integer count consistent with the others
      const pr = clamp((baseY - py) / plotH, 0, 1);
      // invert MLE π=x/n with the OTHER categories' counts held fixed:
      // x = pr * (rest)/(1-pr); clamp to range.
      const rest = counts.reduce((a, c, j) => (j === i ? a : a + c), 0);
      let target: number;
      if (pr >= 0.999) target = maxCount;
      else if (rest === 0) target = Math.round(pr * maxCount); // degenerate: scale directly
      else target = Math.round((pr / (1 - pr)) * rest);
      setCount(i, target, false);
    },
    {
      onStart: () => (rc.canvas.style.cursor = "ns-resize"),
      onEnd: () => (rc.canvas.style.cursor = "crosshair"),
    },
  );
  rc.canvas.style.cursor = "crosshair";

  // a11y + keyboard
  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Zero-count problem: bar chart of categorical probability estimates. The MLE assigns exactly zero probability to an unseen category; the MAP estimate under a Dirichlet prior keeps it positive. Use square brackets to pick a category, arrow keys to change the prior strength, and shift with arrows to change the selected count.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "[" || e.key === "]") {
      e.preventDefault();
      const next = (focus + (e.key === "]" ? 1 : -1) + m) % m;
      select(next);
      return;
    }
    const isCount = e.shiftKey;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      if (isCount) setCount(focus, counts[focus] - 1, true);
      else setAlpha(alpha - 0.25, true);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      if (isCount) setCount(focus, counts[focus] + 1, true);
      else setAlpha(alpha + 0.25, true);
    }
  };
  rc.canvas.addEventListener("keydown", onKey);

  render();
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setAlpha,
    setCount,
    select,
    destroy() {
      destroyed = true;
      alphaTracker.stop();
      for (const t of countTrackers) t.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}
