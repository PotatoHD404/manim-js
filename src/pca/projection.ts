import { clamp, prefersReducedMotion, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { arrow, disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { covariance, jacobiEigen, varianceAlong, type Vec } from "../core/math/linalg";
import { correlatedCloud, Rng } from "../core/math/rng";
import { ellipseWorld, fitView, numberPlane, type View } from "../core/plane";
import { type Theme, withTheme } from "../core/theme";

export interface ProjectionOptions {
  points?: number;
  seed?: number;
  /** Initial direction angle, degrees. */
  angle?: number;
  /** 2×2 mixing matrix `A`; data covariance is `A Aᵀ`. */
  mix?: [[number, number], [number, number]];
  theme?: Partial<Theme>;
}

export interface ProjectionApi {
  setAngle(deg: number, animate?: boolean): void;
  snapToPrincipalAxis(): void;
  play(): void;
  destroy(): void;
}

/**
 * Scene — "maximize the variance". The covariance ellipse shows the shape of the
 * data; the gold arrows are its eigenvectors. Spin the cream direction arrow and
 * the variance along it, `uᵀΣu`, is traced out as a curve below — its single
 * peak sits exactly on the long eigenvector. Best-linear-approximation and
 * maximum-variance are then visibly the same statement.
 */
export function createProjection(
  target: HTMLElement,
  options: ProjectionOptions = {},
): ProjectionApi {
  const theme = withTheme(options.theme);
  const n = options.points ?? 220;
  const mix = options.mix ?? [
    [1.75, 0.0],
    [0.95, 0.62],
  ];

  const rng = new Rng(options.seed ?? 20260621);
  const raw = correlatedCloud(rng, n, mix);
  const mx = raw.reduce((a, p) => a + p[0], 0) / n;
  const my = raw.reduce((a, p) => a + p[1], 0) / n;
  const pts: Vec[] = raw.map((p) => [p[0] - mx, p[1] - my]);

  const cov = covariance(pts);
  const eigen = jacobiEigen(cov);
  const v1 = eigen.vectors[0] as Pt;
  const v2 = eigen.vectors[1] as Pt;
  const l1 = eigen.values[0];
  const l2 = eigen.values[1];
  const v1Angle = Math.atan2(v1[1], v1[0]);
  const cloudMaxR = Math.max(...pts.map((p) => Math.hypot(p[0], p[1])));
  const half = Math.max(cloudMaxR * 1.04, 2 * Math.sqrt(l1) * 1.08);

  const traceN = 181;
  const traceVar = Array.from({ length: traceN }, (_, i) => varianceAlong(cov, dir((i / (traceN - 1)) * Math.PI)));
  const maxVar = l1;

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });

  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  const traceHost = el("div", { style: `position:relative;border-top:1px solid ${theme.grid};` });
  panel.append(mainHost, traceHost);

  const overlay = overlayLayer("tl");
  const eqVar = liveEquation("\\operatorname{Var}(u)=u^{\\top}\\Sigma\\,u\\;=", theme, theme.blue);
  const eqAng = liveEquation("\\theta\\;=", theme, theme.cream);
  overlay.append(eqVar.node, eqAng.node);
  mainHost.append(overlay);

  const eigOverlay = overlayLayer("tr");
  eigOverlay.append(
    staticEquation(`\\lambda_1=${l1.toFixed(2)}`, theme, theme.gold),
    staticEquation(`\\lambda_2=${l2.toFixed(2)}`, theme, theme.gold),
  );
  mainHost.append(eigOverlay);

  const traceTag = el("div", {
    style: `position:absolute;top:8px;left:14px;pointer-events:none;color:${theme.green};font:12px ${theme.mono};`,
  });
  traceTag.textContent = "variance along u  vs.  θ";
  traceHost.append(traceTag);

  root.append(panel);

  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "180",
    step: "1",
    value: String(Math.round(options.angle ?? 18)),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "axis angle in degrees",
  });
  slider.addEventListener("input", () => {
    stopSweep();
    angleTracker.jump((Number(slider.value) * Math.PI) / 180);
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag · sweep · ←→";
  controls.append(playBtn, slider, hint);
  root.append(controls);

  target.append(root);

  // --- state ---------------------------------------------------------------
  let angle = ((options.angle ?? 18) * Math.PI) / 180;
  let sweepActive = false;
  let sweepCancel: (() => void) | null = null;
  let destroyed = false;

  const rcMain = responsiveCanvas(mainHost, 1.34, () => render());
  const rcTrace = responsiveCanvas(traceHost, 4.9, () => render());

  // The displayed angle eases toward its target, so every interaction glides.
  const angleTracker = valueTracker(angle, (v) => {
    angle = v;
    render();
  });

  function stopSweep(): void {
    if (!sweepActive) return;
    sweepCancel?.();
    sweepCancel = null;
    sweepActive = false;
    updatePlayBtn();
  }

  function render(): void {
    drawMain();
    drawTrace();
    const u = dir(angle);
    eqVar.set(varianceAlong(cov, u).toFixed(2));
    let deg = (angle * 180) / Math.PI;
    while (deg < 0) deg += 180;
    while (deg >= 180) deg -= 180;
    eqAng.set(`${deg.toFixed(0)}°`);
    if (document.activeElement !== slider) slider.value = String(Math.round(deg));
  }

  function drawMain(): void {
    const { ctx, width: W, height: H } = rcMain;
    const view = fitView(W, H, half);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);
    numberPlane(ctx, view, theme, { step: 1 });

    const u = dir(angle);

    // covariance ellipses at 1σ and 2σ
    drawEllipse(ctx, view, 1, 0.12, 0.55);
    drawEllipse(ctx, view, 2, 0.05, 0.85);

    // residual drops + projected spread along u
    for (let i = 0; i < pts.length; i += 3) {
      const p = pts[i] as Pt;
      const t = p[0] * u[0] + p[1] * u[1];
      const proj: Pt = [t * u[0], t * u[1]];
      strokeLine(ctx, view.toScreen(p), view.toScreen(proj), theme.red, 1, 0.4);
    }

    // data points (the figure: bright, faintly glowing). Radius tracks the view
    // so dots read like Manim's world-unit Dot rather than a fixed pixel size.
    const dotR = Math.max(2.2, view.scale * 0.04);
    glow(ctx, theme.blue, 4, () => {
      for (const p of pts) disc(ctx, view.toScreen(p as Pt), dotR, theme.blue, 0.85);
    });

    // the candidate axis: dashed full line through the origin
    const axEnd: Pt = [u[0] * half * 1.5, u[1] * half * 1.5];
    const axEndN: Pt = [-u[0] * half * 1.5, -u[1] * half * 1.5];
    polyline(ctx, [view.toScreen(axEndN), view.toScreen(axEnd)], {
      color: theme.cream,
      width: 1,
      alpha: 0.3,
      dash: [6, 7],
    });

    // projected points sitting on u
    for (let i = 0; i < pts.length; i += 3) {
      const p = pts[i] as Pt;
      const t = p[0] * u[0] + p[1] * u[1];
      disc(ctx, view.toScreen([t * u[0], t * u[1]]), 2.1, theme.cream, 0.9);
    }

    // eigenvector arrows (the answer), scaled to one standard deviation
    drawEigenArrow(ctx, view, v1, Math.sqrt(l1), "v", "1");
    drawEigenArrow(ctx, view, v2, Math.sqrt(l2), "v", "2");

    // the movable direction arrow
    const uTip: Pt = [u[0] * half * 0.92, u[1] * half * 0.92];
    glow(ctx, theme.cream, 8, () => {
      arrow(ctx, view.toScreen([0, 0]), view.toScreen(uTip), { color: theme.cream, width: 3 });
    });
    const us = view.toScreen(uTip);
    const lab: Pt = [us[0] + (u[0] >= 0 ? 9 : -9), us[1] - 9];
    mathLabel(ctx, "u", lab, {
      color: theme.cream,
      size: 19,
      align: u[0] >= 0 ? "left" : "right",
      glow: 6,
    });
  }

  function drawEllipse(ctx: CanvasRenderingContext2D, view: View, k: number, fillA: number, strokeA: number): void {
    const pts2 = ellipseWorld([0, 0], v1, v2, k * Math.sqrt(l1), k * Math.sqrt(l2)).map((p) => view.toScreen(p));
    glow(ctx, theme.blue, 5, () => {
      polyline(ctx, pts2, { color: theme.blue, width: 1.8, alpha: strokeA, closed: true, fill: hexA(theme.blue, fillA) });
    });
  }

  function drawEigenArrow(ctx: CanvasRenderingContext2D, view: View, v: Pt, lenWorld: number, base: string, sub: string): void {
    const tip: Pt = [v[0] * lenWorld, v[1] * lenWorld];
    glow(ctx, theme.gold, 7, () => {
      arrow(ctx, view.toScreen([0, 0]), view.toScreen(tip), { color: theme.gold, width: 3.4 });
    });
    const s = view.toScreen(tip);
    mathLabel(ctx, base, [s[0] + (v[0] >= 0 ? 8 : -8), s[1] - 7], {
      color: theme.gold,
      size: 17,
      sub,
      align: v[0] >= 0 ? "left" : "right",
      glow: 5,
    });
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
    const x = (deg: number) => ml + (deg / 180) * (W - ml - mr);
    const y = (vv: number) => H - mb - (vv / (maxVar * 1.08)) * (H - mt - mb);

    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.2, 1);
    strokeLine(ctx, [ml, H - mb], [ml, mt], theme.axis, 1.2, 1);

    let degNow = (angle * 180) / Math.PI;
    while (degNow < 0) degNow += 180;
    while (degNow >= 180) degNow -= 180;

    // full curve (faint) and the swept-in part (bright) up to the current angle
    const curve: Pt[] = traceVar.map((vv, i) => [x((i / (traceN - 1)) * 180), y(vv)]);
    polyline(ctx, curve, { color: theme.green, width: 1.6, alpha: 0.32 });
    const upto = curve.filter((_, i) => (i / (traceN - 1)) * 180 <= degNow);
    if (upto.length > 1) polyline(ctx, upto, { color: theme.green, width: 2.4, alpha: 1 });

    // peak marker at the optimum
    let peakDeg = (v1Angle * 180) / Math.PI;
    while (peakDeg < 0) peakDeg += 180;
    while (peakDeg >= 180) peakDeg -= 180;
    strokeLine(ctx, [x(peakDeg), y(0)], [x(peakDeg), y(maxVar)], theme.gold, 1, 0.5);
    disc(ctx, [x(peakDeg), y(maxVar)], 3, theme.gold, 1);

    // current angle marker
    const vNow = varianceAlong(cov, dir(angle));
    strokeLine(ctx, [x(degNow), H - mb], [x(degNow), y(vNow)], theme.cream, 1, 0.5);
    disc(ctx, [x(degNow), y(vNow)], 3.4, theme.cream, 1);

    ctx.fillStyle = theme.tick;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const d of [0, 45, 90, 135, 180]) ctx.fillText(`${d}°`, x(d), H - mb + 4);
  }

  // --- interaction ---------------------------------------------------------
  function setAngle(deg: number, animate = false): void {
    stopSweep();
    const targetRad = (deg * Math.PI) / 180;
    if (animate) angleTracker.set(targetRad, true);
    else angleTracker.jump(targetRad);
  }

  function play(): void {
    if (sweepActive) {
      stopSweep();
      return;
    }
    if (prefersReducedMotion()) {
      setAngle((v1Angle * 180) / Math.PI, false);
      return;
    }
    sweepActive = true;
    updatePlayBtn();
    let sweep = (angle % Math.PI) / Math.PI;
    sweepCancel = ticker((dt) => {
      sweep += dt / 4.5;
      if (sweep >= 1) {
        sweepActive = false;
        sweepCancel = null;
        updatePlayBtn();
        angleTracker.set(v1Angle, true);
        return false;
      }
      angleTracker.jump(sweep * Math.PI);
    });
  }

  function updatePlayBtn(): void {
    playBtn.innerHTML = sweepActive ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; sweep θ";
  }
  updatePlayBtn();
  playBtn.addEventListener("click", play);

  const stopDrag = draggable(
    rcMain.canvas,
    (px, py) => {
      stopSweep();
      const cx = rcMain.width / 2;
      const cy = rcMain.height / 2;
      let a = Math.atan2(-(py - cy), px - cx);
      if (a < 0) a += Math.PI;
      if (a >= Math.PI) a -= Math.PI;
      angleTracker.jump(a);
    },
    { onStart: () => (rcMain.canvas.style.cursor = "grabbing"), onEnd: () => (rcMain.canvas.style.cursor = "grab") },
  );
  rcMain.canvas.style.cursor = "grab";

  rcMain.canvas.tabIndex = 0;
  rcMain.canvas.setAttribute("role", "img");
  rcMain.canvas.setAttribute(
    "aria-label",
    "Maximize the variance: rotate the candidate axis u through the data cloud; the variance u-transpose-Sigma-u is plotted below and peaks at the first principal axis.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const stepDeg = e.shiftKey ? 5 : 1;
    let deg = (angle * 180) / Math.PI;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") deg -= stepDeg;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") deg += stepDeg;
    else return;
    e.preventDefault();
    stopSweep();
    deg = ((deg % 180) + 180) % 180;
    angleTracker.set((deg * Math.PI) / 180, true);
  };
  rcMain.canvas.addEventListener("keydown", onKey);
  const stopKeys = () => rcMain.canvas.removeEventListener("keydown", onKey);

  render();
  // Math labels use KaTeX's font; redraw once it is guaranteed loaded.
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) render();
    });
  }

  return {
    setAngle,
    snapToPrincipalAxis: () => setAngle((v1Angle * 180) / Math.PI, true),
    play,
    destroy() {
      destroyed = true;
      angleTracker.stop();
      sweepCancel?.();
      stopDrag();
      stopKeys();
      rcMain.destroy();
      rcTrace.destroy();
      root.remove();
    },
  };
}

function dir(theta: number): Vec {
  return [Math.cos(theta), Math.sin(theta)];
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
