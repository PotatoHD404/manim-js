import { clamp, prefersReducedMotion, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { type Theme, withTheme } from "../core/theme";

export interface FisherReparamOptions {
  /** Initial probability point p ∈ (0,1). Defaults to the post's worked point 0.30. */
  p?: number;
  theme?: Partial<Theme>;
}

export interface FisherReparamApi {
  /** Move the read-off point p (eased when animate). */
  setP(p: number, animate?: boolean): void;
  /** Snap to the symmetric centre p = 1/2, where both curves are flattest. */
  snapToCentre(): void;
  destroy(): void;
}

/**
 * Scene — "a tensor, not a number". The same Bernoulli model carries different
 * Fisher information in different coordinates. In the probability coordinate
 * `p` it is `I(p)=1/(p(1-p))`, which blows up at the edges; in the natural
 * parameter `η=logit(p)` the Jacobian `g'(η)=dp/dη=p(1-p)` rescales it to the
 * tame `I(η)=(g'(η))² I(p)=p(1-p)`. The slider moves a point over `p`; both
 * Fisher curves are drawn on a log axis (the gold one is the reparameterized
 * "answer"), a cursor reads each off, and a vertical bracket between them shows
 * the live `(g'(η))²` rescaling factor that carries one curve to the other.
 *
 * Ported from the post's notebook cell 15: `ps∈[0.02,0.98]`, `Ip=1/(p(1-p))`,
 * `Ieta=p(1-p)`, `J=p(1-p)`, with the tensor law `I_η=J² I_p` and a log y-axis
 * clipped to `[0.05, 60]`.
 */
export function createFisherReparam(
  target: HTMLElement,
  options: FisherReparamOptions = {},
): FisherReparamApi {
  const theme = withTheme(options.theme);

  // --- model (exactly the notebook grid) -----------------------------------
  const P_LO = 0.02;
  const P_HI = 0.98;
  const G = 400;
  const ps = new Float64Array(G);
  const ip = new Float64Array(G); // I(p)   = 1 / (p(1-p))   — diverges at edges
  const ieta = new Float64Array(G); // I(η) = J² I(p) = p(1-p) — bounded
  for (let i = 0; i < G; i++) {
    const p = P_LO + (P_HI - P_LO) * (i / (G - 1));
    ps[i] = p;
    ip[i] = ipOf(p);
    ieta[i] = ietaOf(p);
  }

  // log y-axis bounds, matching the post (ax.set_ylim(0.05, 60))
  const Y_LO = 0.05;
  const Y_HI = 60;
  const logLo = Math.log10(Y_LO);
  const logHi = Math.log10(Y_HI);

  // --- state ---------------------------------------------------------------
  let p = clamp(options.p ?? 0.3, P_LO, P_HI);
  let destroyed = false;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  // top-left: the transformation law, live for the current p
  const overlay = overlayLayer("tl");
  const eqLaw = staticEquation("I_{\\eta}=\\bigl(g'(\\eta)\\bigr)^{2}\\,I_{\\theta}", theme, theme.gold);
  const eqIp = liveEquation("I(p)=\\dfrac{1}{p(1-p)}=", theme, theme.blue);
  const eqJ = liveEquation("g'(\\eta)=\\dfrac{dp}{d\\eta}=p(1-p)=", theme, theme.cream);
  const eqIeta = liveEquation("I(\\eta)=\\bigl(g'(\\eta)\\bigr)^{2}I(p)=", theme, theme.gold);
  overlay.append(eqLaw, eqIp.node, eqJ.node, eqIeta.node);
  panel.append(overlay);

  // top-right: the worked point
  const pointOverlay = overlayLayer("tr");
  const eqPoint = liveEquation("p\\;=", theme, theme.cream);
  pointOverlay.append(eqPoint.node);
  panel.append(pointOverlay);

  const rc = responsiveCanvas(panel, 1.62, () => render());

  // --- controls ------------------------------------------------------------
  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const centreBtn = el("button", { type: "button", style: btnStyle(theme) });
  centreBtn.innerHTML = "&#9679;&nbsp; centre p=½";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "point  p";
  const slider = el("input", {
    type: "range",
    min: String(Math.round(P_LO * 1000)),
    max: String(Math.round(P_HI * 1000)),
    step: "1",
    value: String(Math.round(p * 1000)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "probability point p",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag the curve · centre · ←→";
  controls.append(centreBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- the one eased scalar: the read-off point over p ---------------------
  const pTracker = valueTracker(p, (v) => {
    p = clamp(v, P_LO, P_HI);
    render();
  });

  // --- plot geometry (set per draw) ----------------------------------------
  const ml = 50;
  const mr = 18;
  const mt = 16;
  const mb = 30;

  function render(): void {
    draw();
    const Ip = ipOf(p);
    const J = jacOf(p);
    const Ieta = ietaOf(p);
    eqIp.set(Ip.toFixed(2));
    eqJ.set(J.toFixed(3));
    eqIeta.set(Ieta.toFixed(3));
    eqPoint.set(p.toFixed(3));
    if (document.activeElement !== slider) slider.value = String(Math.round(p * 1000));
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const plotW = W - ml - mr;
    const plotH = H - mt - mb;
    const X = (pp: number) => ml + clamp((pp - P_LO) / (P_HI - P_LO), 0, 1) * plotW;
    const Y = (v: number) => {
      const l = (Math.log10(clamp(v, Y_LO, Y_HI)) - logLo) / (logHi - logLo);
      return mt + plotH - l * plotH;
    };

    // faint number-plane backdrop: log decade grid (cyan, receding) + p grid
    for (const dec of [-1, 0, 1]) {
      const yy = Y(Math.pow(10, dec));
      strokeLine(ctx, [ml, yy], [W - mr, yy], theme.grid, 1, 0.32);
      // minor log subdivisions within the decade
      for (let k = 2; k <= 9; k++) {
        const ym = Y(Math.pow(10, dec) * k);
        if (ym > mt && ym < mt + plotH) strokeLine(ctx, [ml, ym], [W - mr, ym], theme.grid, 1, 0.12);
      }
    }
    for (const g of [0.2, 0.4, 0.6, 0.8]) {
      strokeLine(ctx, [X(g), mt], [X(g), mt + plotH], theme.grid, 1, 0.16);
    }

    // axes
    strokeLine(ctx, [ml, mt + plotH], [W - mr, mt + plotH], theme.axis, 1.3, 1);
    strokeLine(ctx, [ml, mt], [ml, mt + plotH], theme.axis, 1.3, 1);

    const toScreen = (arr: Float64Array): Pt[] => {
      const out: Pt[] = new Array(G);
      for (let i = 0; i < G; i++) out[i] = [X(ps[i]), Y(arr[i])];
      return out;
    };

    // I(p): the original coordinate — bright blue, diverging at the edges
    glow(ctx, theme.blue, 5, () => {
      polyline(ctx, toScreen(ip), { color: theme.blue, width: 2.4, alpha: 1 });
    });
    // I(η): the reparameterized "answer" — gold, bounded and tame
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, toScreen(ieta), { color: theme.gold, width: 2.4, alpha: 1 });
    });

    // the read-off cursor (the movable variable): a vertical rule at p, with the
    // (g'(η))² rescaling shown as a bracket joining I(p) to I(η).
    const Ip = ipOf(p);
    const Ieta = ietaOf(p);
    const cx = X(p);
    const yIp = Y(Ip);
    const yIeta = Y(Ieta);
    strokeLine(ctx, [cx, mt], [cx, mt + plotH], theme.cream, 1.2, 0.5);

    // the rescaling segment: green (the traced quantity carrying one to the other)
    glow(ctx, theme.green, 5, () => {
      strokeLine(ctx, [cx, yIp], [cx, yIeta], theme.green, 2.6, 0.95);
    });
    // small bracket ticks at each end of the rescaling segment
    strokeLine(ctx, [cx - 5, yIp], [cx + 5, yIp], theme.green, 2, 0.95);
    strokeLine(ctx, [cx - 5, yIeta], [cx + 5, yIeta], theme.green, 2, 0.95);
    // the factor label, placed beside the bracket midpoint
    const midY = (yIp + yIeta) / 2;
    mathLabel(ctx, "×", [cx + (p < 0.85 ? 9 : -9), midY + 4], {
      color: theme.green,
      size: 13,
      align: p < 0.85 ? "left" : "right",
      glow: 4,
    });
    mathLabel(ctx, "(g'(η))²", [cx + (p < 0.85 ? 22 : -22), midY + 4], {
      color: theme.green,
      size: 13,
      align: p < 0.85 ? "left" : "right",
      glow: 4,
    });

    // read-off markers on each curve
    glow(ctx, theme.blue, 6, () => disc(ctx, [cx, yIp], 4, theme.blue, 1));
    glow(ctx, theme.gold, 6, () => disc(ctx, [cx, yIeta], 4, theme.gold, 1));

    // curve labels in KaTeX serif, hugging the curves at the right edge
    mathLabel(ctx, "I", [X(0.86) + 2, Y(ipOf(0.86)) - 8], { color: theme.blue, size: 16, sub: "θ", glow: 5 });
    mathLabel(ctx, "I", [X(0.86) + 2, Y(ietaOf(0.86)) + 18], { color: theme.gold, size: 16, sub: "η", glow: 5 });

    // p axis ticks (roman numerals) and a y-axis decade label set
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      if (g < P_LO || g > P_HI) {
        // still show 0 and 1 anchored at the axis bounds for readability
        ctx.fillText(g.toFixed(1), clamp(X(g), ml, W - mr), mt + plotH + 5);
      } else {
        ctx.fillText(g.toFixed(1), X(g), mt + plotH + 5);
      }
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const dec of [-1, 0, 1]) {
      const v = Math.pow(10, dec);
      ctx.fillText(dec === 0 ? "1" : `10${sup(dec)}`, ml - 7, Y(v));
    }

    // axis titles (KaTeX serif): p along the bottom, "Fisher information" rotated
    mathLabel(ctx, "p", [ml + plotW / 2, mt + plotH + 18], { color: theme.tick, size: 15, align: "center" });
  }

  // --- interaction ---------------------------------------------------------
  function setP(np: number, animate = false): void {
    const t = clamp(np, P_LO, P_HI);
    if (animate) pTracker.set(t, true);
    else pTracker.jump(t);
  }

  function snapToCentre(): void {
    setP(0.5, true);
  }

  slider.addEventListener("input", () => pTracker.jump(Number(slider.value) / 1000));
  centreBtn.addEventListener("click", snapToCentre);

  const stopDrag = draggable(
    rc.canvas,
    (px) => {
      const plotW = rc.width - ml - mr;
      pTracker.jump(clamp(P_LO + ((px - ml) / plotW) * (P_HI - P_LO), P_LO, P_HI));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Fisher information under reparameterization: the Bernoulli information in the probability coordinate, I(p)=1/(p(1-p)), diverges at the edges, while in the natural parameter eta=logit(p) it is rescaled by the squared Jacobian (dp/deta)^2 to the bounded I(eta)=p(1-p). Drag to read both off at a chosen p.",
  );

  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 0.05 : 0.01;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setP(p - step, true);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setP(p + step, true);
    else if (e.key === "Home") setP(0.5, true);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  // Reduced motion: settle on the centre immediately so there's no glide.
  if (prefersReducedMotion() && options.p === undefined) p = 0.5;

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setP,
    snapToCentre,
    destroy() {
      destroyed = true;
      pTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- math (the notebook, verbatim) -----------------------------------------

/** Bernoulli Fisher information in the probability coordinate: 1 / (p(1-p)). */
function ipOf(p: number): number {
  return 1 / (p * (1 - p));
}

/** Jacobian g'(η) = dp/dη = p(1-p) for the logit reparameterization η=logit(p). */
function jacOf(p: number): number {
  return p * (1 - p);
}

/** Fisher information in the natural parameter: I(η) = (g'(η))² I(p) = p(1-p). */
function ietaOf(p: number): number {
  return p * (1 - p);
}

/** Render a small integer exponent with Unicode superscript glyphs (−1, 1). */
function sup(n: number): string {
  const map: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³" };
  return String(n)
    .split("")
    .map((c) => map[c] ?? c)
    .join("");
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
