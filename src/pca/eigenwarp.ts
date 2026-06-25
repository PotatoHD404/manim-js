import { clamp, prefersReducedMotion, smooth, ticker, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { arrow, disc, glow, mathLabel, type Pt, polyline, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { covariance, jacobiEigen, type Mat, type Vec } from "../core/math/linalg";
import { correlatedCloud, Rng } from "../core/math/rng";
import { fitView, numberPlane } from "../core/plane";
import { type Theme, withTheme } from "../core/theme";

export interface EigenwarpOptions {
  points?: number;
  seed?: number;
  mix?: [[number, number], [number, number]];
  theme?: Partial<Theme>;
}

export interface EigenwarpApi {
  setT(t: number): void;
  play(): void;
  destroy(): void;
}

/**
 * Scene — "the covariance is a transformation". A round, isotropic cloud and its
 * grid are pushed through `Σ^{1/2}`, morphing into the data ellipse. The gold
 * eigenvector lines never change direction — points on them only slide in and
 * out — while the cream test vector swings off its original span. That is what an
 * eigenvector *is*, and the principal axes are exactly those of the covariance.
 */
export function createEigenwarp(target: HTMLElement, options: EigenwarpOptions = {}): EigenwarpApi {
  const theme = withTheme(options.theme);
  const n = options.points ?? 240;
  const mix = options.mix ?? [
    [1.75, 0.0],
    [0.95, 0.62],
  ];

  const rng = new Rng(options.seed ?? 11);
  const raw = correlatedCloud(rng, n, mix);
  const mx = raw.reduce((a, p) => a + p[0], 0) / n;
  const my = raw.reduce((a, p) => a + p[1], 0) / n;
  const data: Pt[] = raw.map((p) => [p[0] - mx, p[1] - my]);

  const cov = covariance(data);
  const eigen = jacobiEigen(cov);
  const v1 = eigen.vectors[0] as Pt;
  const v2 = eigen.vectors[1] as Pt;
  const l1 = eigen.values[0];
  const l2 = eigen.values[1];

  // Floor the eigenvalues so a (near-)degenerate cloud can't send Σ^{-1/2} to
  // infinity and blank the whole scene with NaNs.
  const eps = Math.max(l1, 1) * 1e-9;
  const sqrtM = symFromEig(v1, v2, Math.sqrt(l1), Math.sqrt(l2));
  const invSqrt = symFromEig(v1, v2, 1 / Math.sqrt(Math.max(l1, eps)), 1 / Math.sqrt(Math.max(l2, eps)));
  // whitened (round) cloud — Σ^{-1/2} x; at t=1 the morph returns exactly `data`
  const round: Pt[] = data.map((p) => applyMat(invSqrt, p));

  const half = Math.max(Math.max(...data.map((p) => Math.hypot(p[0], p[1]))) * 1.05, 2 * Math.sqrt(l1) * 1.1);
  const K = Math.ceil(half);

  const v1Ang = Math.atan2(v1[1], v1[0]);
  const testBase: Pt = [Math.cos(v1Ang + 0.62) * 1.9, Math.sin(v1Ang + 0.62) * 1.9];

  // --- DOM -----------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const host = el("div", { style: "position:relative;" });
  panel.append(host);

  const overlay = overlayLayer("tl");
  overlay.append(staticEquation("x=\\Sigma^{1/2}\\,z", theme, theme.cream));
  const eqT = liveEquation("t\\;=", theme, theme.fg);
  overlay.append(eqT.node);
  host.append(overlay);

  const eigOverlay = overlayLayer("tr");
  eigOverlay.append(
    staticEquation(`\\lambda_1=${l1.toFixed(2)}`, theme, theme.gold),
    staticEquation(`\\lambda_2=${l2.toFixed(2)}`, theme, theme.gold),
  );
  host.append(eigOverlay);

  const legend = el("div", {
    style: `position:absolute;bottom:10px;left:14px;pointer-events:none;font:11px ${theme.mono};line-height:1.7;`,
  });
  legend.innerHTML =
    `<span style="color:${theme.gold}">— eigenvectors keep direction</span><br>` +
    `<span style="color:${theme.cream}">— a non-eigen vector turns</span>`;
  host.append(legend);

  root.append(panel);

  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;" });
  const playBtn = el("button", { type: "button", style: btnStyle(theme) });
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "100",
    step: "1",
    value: "0",
    style: "flex:1;max-width:260px;",
  });
  const tag = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  tag.textContent = "round  →  data";
  controls.append(playBtn, slider, tag);
  root.append(controls);

  target.append(root);

  // --- state ---------------------------------------------------------------
  let t = 0;
  let playing = false;
  let cancel: (() => void) | null = null;
  let destroyed = false;

  const rc = responsiveCanvas(host, 1.2, () => draw());

  // Displayed t eases toward its target, so drag/slider/keyboard glide.
  const tTracker = valueTracker(0, (v) => {
    t = v;
    draw();
  });

  function stopPlay(): void {
    if (!playing) return;
    cancel?.();
    cancel = null;
    playing = false;
    updateBtn();
  }

  function mAt(tt: number): Mat {
    return [
      [(1 - tt) + tt * sqrtM[0][0], tt * sqrtM[0][1]],
      [tt * sqrtM[1][0], (1 - tt) + tt * sqrtM[1][1]],
    ];
  }

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    const view = fitView(W, H, half);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);
    // faint static reference grid (the "before"), then the live transforming
    // grid on top — the deformation is the whole point, so it must read clearly.
    numberPlane(ctx, view, theme, { step: 1, labels: false, alpha: 0.35 });

    const M = mAt(t);

    // the transformed grid (parallelogram lines) — the moving figure
    for (let i = -K; i <= K; i++) {
      const va = view.toScreen(applyMat(M, [i, -K]));
      const vb = view.toScreen(applyMat(M, [i, K]));
      const ha = view.toScreen(applyMat(M, [-K, i]));
      const hb = view.toScreen(applyMat(M, [K, i]));
      const op = i === 0 ? 0.7 : 0.42;
      strokeLine(ctx, va, vb, theme.blue, i === 0 ? 1.6 : 1, op);
      strokeLine(ctx, ha, hb, theme.blue, i === 0 ? 1.6 : 1, op);
    }

    // morphing cloud (the figure: bright, faintly glowing)
    const dotR = Math.max(2.2, view.scale * 0.04);
    glow(ctx, theme.blue, 4, () => {
      for (const p of round) disc(ctx, view.toScreen(applyMat(M, p)), dotR, theme.blue, 0.82);
    });

    // eigen-direction spans (fixed lines) + sliding arrows
    for (const [v, lam, sub] of [
      [v1, l1, "1"],
      [v2, l2, "2"],
    ] as [Pt, number, string][]) {
      const ext = half * 1.4;
      strokeLine(ctx, view.toScreen([-v[0] * ext, -v[1] * ext]), view.toScreen([v[0] * ext, v[1] * ext]), theme.gold, 1, 0.34);
      const s = (1 - t) + t * Math.sqrt(lam);
      const tip: Pt = [v[0] * 1.6 * s, v[1] * 1.6 * s];
      glow(ctx, theme.gold, 7, () => {
        arrow(ctx, view.toScreen([0, 0]), view.toScreen(tip), { color: theme.gold, width: 3.2 });
      });
      const sp = view.toScreen(tip);
      mathLabel(ctx, "v", [sp[0] + (v[0] >= 0 ? 8 : -8), sp[1] - 7], {
        color: theme.gold,
        size: 17,
        sub,
        align: v[0] >= 0 ? "left" : "right",
        glow: 5,
      });
    }

    // the non-eigen test vector: starts on a span, swings off it
    polyline(ctx, [view.toScreen([-testBase[0] * 1.4, -testBase[1] * 1.4]), view.toScreen([testBase[0] * 1.4, testBase[1] * 1.4])], {
      color: theme.cream,
      width: 1,
      alpha: 0.18,
      dash: [5, 6],
    });
    const tv = applyMat(M, testBase);
    glow(ctx, theme.cream, 7, () => {
      arrow(ctx, view.toScreen([0, 0]), view.toScreen(tv), { color: theme.cream, width: 2.8 });
    });

    eqT.set(t.toFixed(2));
    if (document.activeElement !== slider) slider.value = String(Math.round(t * 100));
  }

  function setT(tt: number, animate = true): void {
    const c = clamp(tt, 0, 1);
    if (animate) tTracker.set(c, true);
    else tTracker.jump(c);
  }

  slider.addEventListener("input", () => {
    stopPlay();
    setT(Number(slider.value) / 100, false);
  });

  function play(): void {
    if (playing) {
      stopPlay();
      return;
    }
    if (prefersReducedMotion()) {
      setT(1, false);
      return;
    }
    playing = true;
    updateBtn();
    const from = t >= 0.999 ? 0 : t;
    if (from === 0) tTracker.jump(0);
    let elapsed = 0;
    const dur = 2.6;
    cancel = ticker((dt) => {
      elapsed += dt;
      const e = smooth(clamp(elapsed / dur, 0, 1));
      tTracker.jump(from + (1 - from) * e);
      if (elapsed >= dur) {
        playing = false;
        cancel = null;
        updateBtn();
        return false;
      }
    });
  }

  function updateBtn(): void {
    playBtn.innerHTML = playing ? "&#10073;&#10073;&nbsp; pause" : "&#9654;&nbsp; morph";
  }
  updateBtn();
  playBtn.addEventListener("click", play);

  const stopDrag = draggable(rc.canvas, (px) => {
    stopPlay();
    setT(px / rc.width, false);
  });

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "The covariance as a transformation: morph a round cloud through Sigma-to-the-half into the data ellipse; eigenvector directions stay fixed while a non-eigen vector turns.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      play();
      return;
    }
    const d = e.shiftKey ? 0.1 : 0.02;
    let nt: number;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") nt = t - d;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") nt = t + d;
    else return;
    e.preventDefault();
    stopPlay();
    setT(nt);
  };
  rc.canvas.addEventListener("keydown", onKey);

  draw();
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) draw();
    });
  }

  return {
    setT,
    play,
    destroy() {
      destroyed = true;
      tTracker.stop();
      cancel?.();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

function symFromEig(v1: Pt, v2: Pt, a: number, b: number): Mat {
  return [
    [a * v1[0] * v1[0] + b * v2[0] * v2[0], a * v1[0] * v1[1] + b * v2[0] * v2[1]],
    [a * v1[0] * v1[1] + b * v2[0] * v2[1], a * v1[1] * v1[1] + b * v2[1] * v2[1]],
  ];
}

function applyMat(m: Mat, p: Vec): Pt {
  return [m[0][0] * p[0] + m[0][1] * p[1], m[1][0] * p[0] + m[1][1] * p[1]];
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
