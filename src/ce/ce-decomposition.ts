import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface CeDecompositionOptions {
  /** True Bernoulli bias p (the data-generating distribution). Default 0.7. */
  p?: number;
  /** Initial model probability q (the read-off cursor). Default p so it starts at the floor. */
  q?: number;
  theme?: Partial<Theme>;
}

export interface CeDecompositionApi {
  /** Set the true bias p ∈ (0,1). */
  setP(p: number): void;
  /** Move the model-probability cursor q (eased when animate). */
  setQ(q: number, animate?: boolean): void;
  /** Snap the cursor onto q = p, where KL = 0 and cross-entropy bottoms out. */
  snapToFloor(): void;
  destroy(): void;
}

/**
 * Scene — "cross-entropy = entropy + KL". For a Bernoulli true bias p the model
 * probability q is swept across (0,1) and three quantities are drawn in nats:
 * the cross-entropy H(p,q) = −[p log q + (1−p) log(1−q)] (the bowl, the answer),
 * the entropy floor H(p) = H(p,p) (flat, the irreducible cost), and the forward
 * divergence D_KL(p‖q) = H(p,q) − H(p) (gold). A draggable cursor over q shades
 * the identity directly on the chart — a green band 0→H(p) for the entropy and a
 * gold band H(p)→H(p,q) for the KL, so the stack literally sums to the blue
 * cross-entropy. A side stacked bar restates H + KL = CE with live values. Both
 * the bowl and the KL bottom out at q = p, where KL = 0 and H(p,q) = H(p).
 *
 * Ported verbatim from the post's notebook cell (p=0.7, q grid 1e-3..1−1e-3,
 * 500 points; bern_xent and KL = H_pq − H_p exactly as there).
 */
export function createCeDecomposition(
  target: HTMLElement,
  options: CeDecompositionOptions = {},
): CeDecompositionApi {
  const theme = withTheme(options.theme);

  // --- model state ---------------------------------------------------------
  let p = clamp(options.p ?? 0.7, 0.01, 0.99);
  let destroyed = false;

  // q grid — matches the notebook: 500 points on (1e-3, 1−1e-3).
  const G = 500;
  const eps = 1e-12; // keeps logs finite, exactly as the notebook's `eps`
  const qg = new Float64Array(G);
  for (let i = 0; i < G; i++) qg[i] = 1e-3 + (1 - 2e-3) * (i / (G - 1));

  // Reusable curve buffers (nats), recomputed in place each frame.
  const ce = new Float64Array(G); // H(p, q)  cross-entropy
  const kl = new Float64Array(G); // D_KL(p || q)
  let hP = 0; // H(p)  the entropy floor
  const yMax = 3; // nats; the notebook clips the axis at ax.set_ylim(0, 3)

  // --- math (the notebook's bern_xent, term-for-term) ----------------------
  const bernXent = (pp: number, qq: number): number =>
    -(pp * Math.log(qq + eps) + (1 - pp) * Math.log(1 - qq + eps));

  function recompute(): void {
    hP = bernXent(p, p);
    for (let i = 0; i < G; i++) {
      const v = bernXent(p, qg[i]);
      ce[i] = v;
      kl[i] = v - hP; // D_KL = H(p,q) − H(p), exactly as the notebook
    }
  }

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the live decomposition, term by term
  const overlay = overlayLayer("tl");
  const eqCe = liveEquation("H(p,q)=", theme, theme.blue);
  const eqHp = liveEquation("H(p)=", theme, theme.green);
  const eqKl = liveEquation("D_{\\mathrm{KL}}(p\\,\\|\\,q)=", theme, theme.gold);
  overlay.append(eqCe.node, eqHp.node, eqKl.node);
  panel.append(overlay);

  // top-right: the identity, restated as a sum of the live numbers
  const sumOverlay = overlayLayer("tr");
  const eqSum = staticEquation("", theme, theme.muted);
  sumOverlay.append(eqSum);
  panel.append(sumOverlay);

  const rc = responsiveCanvas(panel, 1.66, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", {
    style: "display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:8px 18px;align-items:center;",
  });
  const pRow = sliderRow(theme, "true  p", 1, 99, Math.round(p * 100), "true Bernoulli bias p, in percent");
  const qRow = sliderRow(theme, "model  q", 1, 99, Math.round((options.q ?? p) * 100), "model probability q, in percent");
  controls.append(pRow.row, qRow.row);
  root.append(controls);

  const actions = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const floorBtn = el("button", { type: "button", style: btnStyle(theme) });
  floorBtn.innerHTML = "&#9660;&nbsp; snap q = p";
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag the cursor to read off q  ·  ←→ scrub  ·  p sets the data";
  actions.append(floorBtn, hint);
  root.append(actions);

  target.append(root);

  // --- the one eased scalar: the model-probability cursor q ----------------
  let cursor = clamp(options.q ?? p, 1e-3, 1 - 1e-3);
  const cursorTracker = valueTracker(cursor, (v) => {
    cursor = clamp(v, 1e-3, 1 - 1e-3);
    render();
  });

  // continuous read-offs at the cursor (closed form, not grid interpolation)
  const ceAt = (q: number) => bernXent(p, clamp(q, 1e-3, 1 - 1e-3));
  const klAt = (q: number) => ceAt(q) - hP;

  // --- plot frame (shared by render + drag math) ---------------------------
  const ml = 52;
  const mr = 130; // leaves room for the stacked-bar inset on the right
  const mt = 18;
  const mb = 34;

  // --- render --------------------------------------------------------------
  function render(): void {
    recompute();
    draw();

    const ceV = ceAt(cursor);
    const klV = klAt(cursor);
    eqCe.set(`${ceV.toFixed(3)} nats`);
    eqHp.set(`${hP.toFixed(3)} nats`);
    eqKl.set(`${klV.toFixed(3)} nats`);
    eqSum.innerHTML = staticEquation(
      `${hP.toFixed(3)}+${klV.toFixed(3)}=${ceV.toFixed(3)}`,
      theme,
      theme.muted,
    ).innerHTML;

    if (document.activeElement !== pRow.input) pRow.set(Math.round(p * 100));
    if (document.activeElement !== qRow.input) qRow.set(Math.round(cursor * 100));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const x = (q: number) => ml + clamp(q, 0, 1) * plotW;
    const y = (v: number) => mt + plotH - (clamp(v, 0, yMax) / yMax) * plotH;

    // faint vertical grid (the cyan plane, receding) at q tenths
    for (let g = 0; g <= 10; g++) {
      const gx = g / 10;
      strokeLine(ctx, [x(gx), mt], [x(gx), mt + plotH], theme.grid, 1, g % 5 === 0 ? 0.26 : 0.13);
    }
    // faint horizontal grid at half-nat steps
    for (let v = 0; v <= yMax + 1e-9; v += 0.5) {
      strokeLine(ctx, [ml, y(v)], [ml + plotW, y(v)], theme.grid, 1, 0.13);
    }
    // axes
    strokeLine(ctx, [ml, mt + plotH], [ml + plotW, mt + plotH], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.4, 1);

    const toScreen = (arr: Float64Array): Pt[] => {
      const out: Pt[] = [];
      for (let i = 0; i < G; i++) {
        const v = arr[i];
        if (v > yMax) continue; // clip to the [0,3] window like the notebook
        out.push([x(qg[i]), y(v)]);
      }
      return out;
    };

    // --- the stacked decomposition at the cursor (the figure's whole idea) ---
    // A green band 0 → H(p) (entropy) and a gold band H(p) → H(p,q) (KL),
    // drawn as a slim shaded column at the cursor's q. The two stack to the
    // blue cross-entropy curve: H(p,q) = H(p) + D_KL(p‖q), made literal.
    const cx = x(cursor);
    const ceCur = ceAt(cursor);
    const klCur = klAt(cursor);
    const bandHalf = Math.max(7, plotW * 0.014);
    // entropy band (green): the irreducible floor
    ctx.fillStyle = hexA(theme.green, 0.22);
    ctx.fillRect(cx - bandHalf, y(hP), bandHalf * 2, mt + plotH - y(hP));
    // KL band (gold): the avoidable surplus above the floor
    ctx.fillStyle = hexA(theme.gold, 0.24);
    ctx.fillRect(cx - bandHalf, y(Math.min(ceCur, yMax)), bandHalf * 2, y(hP) - y(Math.min(ceCur, yMax)));

    // --- the three curves ----------------------------------------------------
    // KL (gold) — the avoidable divergence; zero at q = p
    polyline(ctx, toScreen(kl), { color: theme.gold, width: 2, alpha: 0.95 });

    // H(p) entropy floor (green, dashed) — flat, model-independent
    polyline(ctx, [[ml, y(hP)], [ml + plotW, y(hP)]], {
      color: theme.green,
      width: 1.6,
      alpha: 0.85,
      dash: [7, 6],
    });

    // H(p,q) cross-entropy (bright blue, glowing) — the answer object
    glow(ctx, theme.blue, 6, () => {
      polyline(ctx, toScreen(ce), { color: theme.blue, width: 2.6, alpha: 1 });
    });

    // the optimum: q = p, where the bowl meets the floor and KL = 0
    strokeLine(ctx, [x(p), mt], [x(p), mt + plotH], theme.muted, 1, 0.45);
    glow(ctx, theme.gold, 6, () => disc(ctx, [x(p), y(hP)], 4, theme.gold, 1));
    mathLabel(ctx, "q", [x(p) + (p < 0.85 ? 7 : -7), y(hP) - 10], {
      color: theme.muted,
      size: 13,
      sub: "*",
      align: p < 0.85 ? "left" : "right",
    });

    // the read-off cursor (the movable variable): vertical rule + ticks on each curve
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.55);
    glow(ctx, theme.cream, 7, () => {
      disc(ctx, [cx, y(Math.min(ceCur, yMax))], 3.6, theme.cream, 1);
    });
    disc(ctx, [cx, y(hP)], 2.6, theme.green, 0.95);
    disc(ctx, [cx, y(Math.min(klCur, yMax))], 2.6, theme.gold, 0.95);
    mathLabel(ctx, "q", [cx + (cursor < 0.9 ? 6 : -6), mt + 14], {
      color: theme.cream,
      size: 15,
      align: cursor < 0.9 ? "left" : "right",
      glow: 5,
    });

    // axis ticks (roman numerals)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = 0; g <= 10; g += 2) ctx.fillText((g / 10).toFixed(1), x(g / 10), mt + plotH + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = 0; v <= yMax + 1e-9; v += 1) ctx.fillText(v.toFixed(1), ml - 7, y(v));

    // axis titles (KaTeX serif)
    mathLabel(ctx, "q", [ml + plotW, mt + plotH + 24], { color: theme.tick, size: 14, align: "right" });
    ctx.save();
    ctx.translate(16, mt + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    mathLabel(ctx, "nats", [0, 0], { color: theme.tick, size: 13, align: "center", italic: false });
    ctx.restore();

    // legend (top, under the overlay) + the stacked-bar inset on the right
    drawLegend(ctx, ml + plotW, mt);
    drawStackedBar(ctx, W, H, ceCur);
  }

  /** A vertical stacked bar restating H (green) + KL (gold) = CE (blue), live. */
  function drawStackedBar(ctx: CanvasRenderingContext2D, W: number, H: number, ceCur: number): void {
    const bw = 30;
    const bx = W - mr + 34;
    const top = mt + 30;
    const bot = H - mb;
    const barH = bot - top;
    const Y = (v: number) => bot - (clamp(v, 0, yMax) / yMax) * barH;

    // track
    ctx.fillStyle = hexA(theme.muted, 0.1);
    ctx.fillRect(bx, top, bw, barH);

    // entropy segment (green) at the bottom — the floor
    const yFloor = Y(hP);
    glow(ctx, theme.green, 5, () => {
      ctx.fillStyle = hexA(theme.green, 0.85);
      ctx.fillRect(bx, yFloor, bw, bot - yFloor);
    });
    // KL segment (gold) stacked on top — the surplus
    const yTop = Y(Math.min(ceCur, yMax));
    glow(ctx, theme.gold, 5, () => {
      ctx.fillStyle = hexA(theme.gold, 0.85);
      ctx.fillRect(bx, yTop, bw, yFloor - yTop);
    });
    // the total reaches the cross-entropy: a bright blue cap line
    glow(ctx, theme.blue, 6, () => strokeLine(ctx, [bx - 4, yTop], [bx + bw + 4, yTop], theme.blue, 2.4, 1));

    // segment labels
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `italic 11px ${ROMAN_FONT}`;
    if (bot - yFloor > 12) {
      ctx.fillStyle = theme.green;
      ctx.fillText("H(p)", bx + bw / 2, (yFloor + bot) / 2);
    }
    if (yFloor - yTop > 12) {
      ctx.fillStyle = theme.gold;
      ctx.fillText("KL", bx + bw / 2, (yTop + yFloor) / 2);
    }
    // CE total above the cap
    ctx.fillStyle = theme.blue;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textBaseline = "bottom";
    ctx.fillText(`${ceCur.toFixed(2)}`, bx + bw / 2, yTop - 4);
  }

  function drawLegend(ctx: CanvasRenderingContext2D, right: number, top: number): void {
    const rows: Array<[string, string, number[]]> = [
      ["H(p,q)", theme.blue, []],
      ["H(p)", theme.green, [7, 6]],
      ["D_KL", theme.gold, []],
    ];
    ctx.save();
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lineW = 20;
    let yy = top + 8;
    for (const [name, color, dash] of rows) {
      const lx = right - 92;
      ctx.strokeStyle = color;
      ctx.lineWidth = name === "H(p,q)" ? 2.4 : 1.8;
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

  // --- interaction ---------------------------------------------------------
  function setQ(q: number, animate = false): void {
    if (animate) cursorTracker.set(clamp(q, 1e-3, 1 - 1e-3), true);
    else cursorTracker.jump(clamp(q, 1e-3, 1 - 1e-3));
  }

  function setP(np: number): void {
    p = clamp(np, 0.01, 0.99);
    render();
  }

  function snapToFloor(): void {
    setQ(p, true);
  }

  pRow.input.addEventListener("input", () => setP(Number(pRow.input.value) / 100));
  qRow.input.addEventListener("input", () => setQ(Number(qRow.input.value) / 100, false));
  floorBtn.addEventListener("click", snapToFloor);

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const plotW = rc.width - ml - mr;
      cursorTracker.jump(clamp((px - ml) / plotW, 1e-3, 1 - 1e-3));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Cross-entropy decomposition: for a Bernoulli true bias p, the cross-entropy H(p,q), the entropy floor H(p), and the KL divergence are plotted against the model probability q in nats. A cursor over q shades the entropy band and the KL band that stack to the cross-entropy, and a side bar restates the same sum. The cross-entropy bottoms out and the KL vanishes at q equals p.",
  );

  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setQ(cursor - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setQ(cursor + step, true);
    else if (e.key === "Home" || e.key === "End") setQ(p, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: land on the floor (q = p) immediately so there's no glide.
  if (prefersReducedMotion()) cursor = clamp(p, 1e-3, 1 - 1e-3);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setP,
    setQ,
    snapToFloor,
    destroy() {
      destroyed = true;
      cursorTracker.stop();
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

function sliderRow(theme: Theme, label: string, min: number, max: number, value: number, aria: string): Row {
  const row = el("div", { style: "display:flex;align-items:center;gap:10px;" });
  const lab = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};min-width:6.5ch;` });
  lab.textContent = label;
  const input = el("input", {
    type: "range",
    min: String(min),
    max: String(max),
    step: "1",
    value: String(value),
    style: "flex:1;min-width:90px;",
    "aria-label": aria,
  }) as HTMLInputElement;
  const out = el("output", { style: `font:13px ${ROMAN_FONT};color:${theme.fg};min-width:3.5ch;text-align:right;` });
  out.textContent = (value / 100).toFixed(2);
  input.addEventListener("input", () => (out.textContent = (Number(input.value) / 100).toFixed(2)));
  row.append(lab, input, out);
  return {
    row,
    input,
    set(v: number) {
      input.value = String(v);
      out.textContent = (v / 100).toFixed(2);
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
