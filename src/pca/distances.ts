import { clamp, valueTracker } from "../core/anim";
import { draggable, el, responsiveCanvas } from "../core/dom";
import { disc, glow, mathLabel, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { liveEquation, overlayLayer, staticEquation } from "../core/equation";
import { covariance, jacobiEigen, type Mat, type Vec } from "../core/math/linalg";
import { Rng } from "../core/math/rng";
import { type Theme, withTheme } from "../core/theme";

export interface DistancesOptions {
  /** Number of 3-D samples (pairwise distances scale as n^2/2). */
  points?: number;
  seed?: number;
  /** Near-planar 3-D spectrum, descending. The post uses [6, 3, 0.15]. */
  spectrum?: [number, number, number];
  /** Initial tilt of the projection plane off the principal plane, degrees [0,90]. */
  tilt?: number;
  theme?: Partial<Theme>;
}

export interface DistancesApi {
  /** Tilt the projection plane off the principal plane, in degrees [0,90]. */
  setTilt(deg: number, animate?: boolean): void;
  /** Snap back to the principal plane (tilt = 0). */
  alignToPrincipalPlane(): void;
  /** Draw a fresh Monte-Carlo cloud from the next seed. */
  reseed(): void;
  destroy(): void;
}

/**
 * Scene -- "PCA preserves distances". Near-planar 3-D Gaussian data (spectrum
 * ~ [6,3,0.15]) is projected to 2-D; each dot compares one pair's original 3-D
 * distance with its projected 2-D distance. On the principal plane the dots sit
 * on the gold identity line and the correlation is ~ 1. Tilt the projection
 * plane off that plane and the dots peel below the line -- the map starts
 * trading the second principal axis for the near-null third axis, distances
 * collapse, and the live correlation falls. The MDS objective made visible.
 */
export function createDistances(target: HTMLElement, options: DistancesOptions = {}): DistancesApi {
  const theme = withTheme(options.theme);
  const n = options.points ?? 120;
  const spectrum = options.spectrum ?? [6.0, 3.0, 0.15];
  let seed = options.seed ?? 1;

  // --- model (ported from the notebook's distance-preservation cell) --------
  // A random orthonormal frame Q3 rotates a diagonal covariance diag(spectrum);
  // x = A z with A = Q3 diag(sqrt(spectrum)) gives covariance Q3 diag(spectrum) Q3^T.
  let pts3: Vec[] = [];
  let v1: Vec = [1, 0, 0];
  let v2: Vec = [0, 1, 0];
  let v3: Vec = [0, 0, 1];
  let pairs: Array<[number, number]> = [];
  let dHigh: number[] = []; // original 3-D pairwise distances (fixed)
  let dMax = 1;

  function buildCloud(): void {
    const rng = new Rng(seed);
    const Q = orthoFrame(rng); // 3x3 orthonormal columns
    const sq = spectrum.map((s) => Math.sqrt(s));
    const A: Mat = [
      [Q[0][0] * sq[0], Q[0][1] * sq[1], Q[0][2] * sq[2]],
      [Q[1][0] * sq[0], Q[1][1] * sq[1], Q[1][2] * sq[2]],
      [Q[2][0] * sq[0], Q[2][1] * sq[1], Q[2][2] * sq[2]],
    ];
    const raw: Vec[] = [];
    for (let i = 0; i < n; i++) {
      const z = rng.gaussVec(3);
      raw.push([
        A[0][0] * z[0] + A[0][1] * z[1] + A[0][2] * z[2],
        A[1][0] * z[0] + A[1][1] * z[1] + A[1][2] * z[2],
        A[2][0] * z[0] + A[2][1] * z[1] + A[2][2] * z[2],
      ]);
    }
    const m = [0, 0, 0];
    for (const p of raw) for (let k = 0; k < 3; k++) m[k] += p[k];
    for (let k = 0; k < 3; k++) m[k] /= n;
    pts3 = raw.map((p) => [p[0] - m[0], p[1] - m[1], p[2] - m[2]]);

    // sample-covariance eigenbasis (v1, v2 span the principal plane; v3 is near-null)
    const cov = covariance(pts3);
    const eig = jacobiEigen(cov);
    v1 = eig.vectors[0];
    v2 = eig.vectors[1];
    v3 = eig.vectors[2];

    pairs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
    dHigh = pairs.map(([i, j]) => dist3(pts3[i], pts3[j]));
    dMax = Math.max(...dHigh) * 1.04;
  }
  buildCloud();

  // --- DOM ------------------------------------------------------------------
  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  const mainHost = el("div", { style: "position:relative;" });
  panel.append(mainHost);
  root.append(panel);

  const overlay = overlayLayer("tl");
  const eqRho = liveEquation("\\rho\\,(d_{3},d_{2})\\;=", theme, theme.green);
  const eqTilt = liveEquation("\\angle\\;=", theme, theme.cream);
  overlay.append(eqRho.node, eqTilt.node);
  mainHost.append(overlay);

  const idOverlay = overlayLayer("tr");
  idOverlay.append(
    staticEquation("\\|y_i-y_j\\|=\\|X_i-X_j\\|", theme, theme.gold),
    staticEquation(`\\lambda=[${spectrum.join(",\\,")}]`, theme, theme.muted),
  );
  mainHost.append(idOverlay);

  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const alignBtn = el("button", { type: "button", style: btnStyle(theme) });
  alignBtn.innerHTML = "&#9650;&nbsp; align plane";
  const reseedBtn = el("button", { type: "button", style: btnStyle(theme) });
  reseedBtn.innerHTML = "&#8635;&nbsp; reseed";
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "tilt off principal plane";
  const slider = el("input", {
    type: "range",
    min: "0",
    max: "90",
    step: "1",
    value: String(Math.round(clamp(options.tilt ?? 0, 0, 90))),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "tilt of the projection plane off the principal plane, in degrees",
  });
  const hint = el("span", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  hint.textContent = "drag ↕ · reseed · ←→";
  controls.append(alignBtn, reseedBtn, label, slider, hint);
  root.append(controls);

  target.append(root);

  // --- state ----------------------------------------------------------------
  let tilt = (clamp(options.tilt ?? 0, 0, 90) * Math.PI) / 180; // radians, displayed
  let destroyed = false;

  const rc = responsiveCanvas(mainHost, 1.12, () => render());

  // The displayed tilt eases toward its target, so every interaction glides.
  const tiltTracker = valueTracker(tilt, (v) => {
    tilt = v;
    render();
  });

  // --- projection at a given tilt ------------------------------------------
  // b1 = v1 stays the principal axis; b2 rotates from v2 (tilt 0) toward v3
  // (tilt 90deg). Projected coords y = [x.b1, x.b2]; distances follow.
  function projectedDistances(t: number): { dLow: number[]; rho: number } {
    const ct = Math.cos(t);
    const st = Math.sin(t);
    const b1 = v1;
    const b2: Vec = [
      ct * v2[0] + st * v3[0],
      ct * v2[1] + st * v3[1],
      ct * v2[2] + st * v3[2],
    ];
    const ys: Pt[] = pts3.map((p) => [dot3(p, b1), dot3(p, b2)]);
    const dLow = pairs.map(([i, j]) => Math.hypot(ys[i][0] - ys[j][0], ys[i][1] - ys[j][1]));
    return { dLow, rho: pearson(dHigh, dLow) };
  }

  function render(): void {
    const { dLow, rho } = projectedDistances(tilt);
    draw(dLow, rho);
    eqRho.set(rho.toFixed(4));
    const deg = (tilt * 180) / Math.PI;
    eqTilt.set(`${deg.toFixed(0)}°`);
    if (document.activeElement !== slider) slider.value = String(Math.round(deg));
  }

  function draw(dLow: number[], rho: number): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    // A first-quadrant number plane: only the positive corner is meaningful
    // (distances are non-negative), so shift the origin to the bottom-left.
    const ml = 46;
    const mb = 34;
    const mt = 18;
    const mr = 16;
    const sx = (W - ml - mr) / dMax;
    const sy = (H - mt - mb) / dMax;
    const s = Math.min(sx, sy);
    const ox = ml;
    const oy = H - mb;
    const X = (d: number) => ox + d * s;
    const Y = (d: number) => oy - d * s;

    // faint grid in BLUE_D, like a receded number plane
    const gstep = niceStep(dMax);
    ctx.lineWidth = 1;
    for (let g = gstep; g <= dMax + 1e-6; g += gstep) {
      strokeLine(ctx, [X(g), oy], [X(g), Y(dMax)], theme.grid, 1, 0.28);
      strokeLine(ctx, [ox, Y(g)], [X(dMax), Y(g)], theme.grid, 1, 0.28);
    }
    // axes
    strokeLine(ctx, [ox, oy], [X(dMax), oy], theme.axis, 1.6, 1);
    strokeLine(ctx, [ox, oy], [ox, Y(dMax)], theme.axis, 1.6, 1);

    // identity line d2 = d3 -- the "answer": a perfect linear map lands here
    glow(ctx, theme.gold, 5, () => {
      polyline(ctx, [[X(0), Y(0)], [X(dMax), Y(dMax)]], {
        color: theme.gold,
        width: 1.8,
        alpha: 0.9,
        dash: [7, 7],
      });
    });

    // best-fit slope through the origin (projected vs original) -- a contraction
    // line that lies on the identity at tilt 0 and rotates down as it tilts
    let num = 0;
    let den = 0;
    for (let i = 0; i < dHigh.length; i++) {
      num += dHigh[i] * dLow[i];
      den += dHigh[i] * dHigh[i];
    }
    const slope = den > 0 ? num / den : 1;
    strokeLine(ctx, [X(0), Y(0)], [X(dMax), Y(slope * dMax)], theme.cream, 1.3, 0.6);

    // the scatter: every pair is one glowing BLUE_C dot
    const r = Math.max(1.4, Math.min(2.6, 2600 / dHigh.length));
    glow(ctx, theme.blue, 4, () => {
      for (let i = 0; i < dHigh.length; i++) {
        disc(ctx, [X(dHigh[i]), Y(dLow[i])], r, theme.blue, 0.5);
      }
    });

    // axis labels (KaTeX serif) and ticks (roman numerals)
    mathLabel(ctx, "d", [X(dMax) - 4, oy + 22], { color: theme.tick, size: 16, sub: "3", align: "right" });
    mathLabel(ctx, "d", [ox - 30, Y(dMax) + 14], { color: theme.tick, size: 16, sub: "2" });
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let g = gstep; g <= dMax + 1e-6; g += gstep) ctx.fillText(String(g), X(g), oy + 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let g = gstep; g <= dMax + 1e-6; g += gstep) ctx.fillText(String(g), ox - 6, Y(g));

    // a correlation strength bar in the lower-right: green when locked on identity
    drawRhoMeter(ctx, W, H, rho);

    // a small schematic of the tilting plane through the 3-D cloud
    drawSchematic(ctx, W, H);
  }

  /** A compact strength meter for the live correlation, gold when ~ 1. */
  function drawRhoMeter(ctx: CanvasRenderingContext2D, W: number, H: number, rho: number): void {
    const bw = Math.min(150, W * 0.34);
    const bh = 7;
    const x0 = W - 18 - bw;
    const y0 = H - 22;
    strokeLine(ctx, [x0, y0], [x0 + bw, y0], theme.grid, bh, 0.35);
    // map [0.7, 1] -> [0,1] so the meaningful range fills the bar
    const f = clamp((rho - 0.7) / 0.3, 0, 1);
    const col = f > 0.93 ? theme.gold : f > 0.6 ? theme.green : theme.red;
    glow(ctx, col, 6, () => strokeLine(ctx, [x0, y0], [x0 + bw * f, y0], col, bh, 1));
    ctx.fillStyle = theme.muted;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("distance correlation", x0 + bw, y0 - 8);
  }

  /**
   * Schematic inset: the near-planar 3-D cloud edge-on, with the cyan principal
   * plane and the cream projection plane tilting away from it by the live angle.
   */
  function drawSchematic(ctx: CanvasRenderingContext2D, W: number, H: number): void {
    const cw = Math.min(132, W * 0.3);
    const cx = 16 + cw / 2;
    const cy = H - 16 - cw / 2;
    const R = cw / 2;
    ctx.save();
    // panel
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = "#101010";
    ctx.fillRect(cx - R - 6, cy - R - 6, cw + 12, cw + 12);
    ctx.globalAlpha = 1;
    // the disc-like cloud seen edge-on: a flat ellipse of dots (the [6,3] plane)
    const tilt0 = -0.32; // fixed isometric tip so the plane reads as a disc
    for (let i = 0; i < pts3.length; i++) {
      const u = dot3(pts3[i], v1);
      const w = dot3(pts3[i], v2);
      const o = dot3(pts3[i], v3); // tiny out-of-plane offset
      const px = cx + (u / dMax) * R * 1.5;
      const py = cy - (w / dMax) * R * 0.55 * Math.cos(tilt0) - (o / dMax) * R * 1.5;
      disc(ctx, [px, py], 1.3, theme.blue, 0.5);
    }
    // principal plane (cyan, flat)
    strokeLine(ctx, [cx - R, cy + R * 0.28], [cx + R, cy - R * 0.28], theme.grid, 1.4, 0.8);
    // projection plane (cream) tilting away by the live angle
    const a = tilt;
    const dx = Math.cos(a) * R;
    const dy = Math.sin(a) * R;
    glow(ctx, theme.cream, 5, () => {
      strokeLine(ctx, [cx - dx, cy + R * 0.28 + dy], [cx + dx, cy - R * 0.28 - dy], theme.cream, 1.8, 0.95);
    });
    mathLabel(ctx, "θ", [cx + R - 4, cy - R + 4], { color: theme.cream, size: 13, italic: true });
    ctx.restore();
  }

  // --- interaction ----------------------------------------------------------
  function setTilt(deg: number, animate = false): void {
    const t = (clamp(deg, 0, 90) * Math.PI) / 180;
    if (animate) tiltTracker.set(t, true);
    else tiltTracker.jump(t);
  }

  function reseed(): void {
    seed = (seed + 1) >>> 0 || 1;
    buildCloud();
    render();
  }

  alignBtn.addEventListener("click", () => setTilt(0, true));
  reseedBtn.addEventListener("click", reseed);
  slider.addEventListener("input", () => tiltTracker.jump((Number(slider.value) * Math.PI) / 180));

  // Vertical drag on the canvas controls the tilt (down = tilt away).
  const stopDrag = draggable(
    rc.canvas,
    (_px, py) => {
      const f = clamp(py / rc.height, 0, 1);
      tiltTracker.jump(f * (Math.PI / 2));
    },
    { onStart: () => (rc.canvas.style.cursor = "grabbing"), onEnd: () => (rc.canvas.style.cursor = "grab") },
  );
  rc.canvas.style.cursor = "grab";

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    "Distance preservation: each dot compares an original 3-D pairwise distance with its 2-D projected distance. On the principal plane the dots sit on the identity line and the correlation is near one; tilting the projection plane away peels the dots below the line and lowers the correlation.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Home") {
      e.preventDefault();
      setTilt(0, true);
      return;
    }
    const step = e.shiftKey ? 10 : 2;
    let deg = (tilt * 180) / Math.PI;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") deg -= step;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") deg += step;
    else return;
    e.preventDefault();
    setTilt(deg, true);
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
    setTilt,
    alignToPrincipalPlane: () => setTilt(0, true),
    reseed,
    destroy() {
      destroyed = true;
      tiltTracker.stop();
      stopDrag();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}

// --- helpers ----------------------------------------------------------------

function dot3(a: Vec, b: Vec): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function dist3(a: Vec, b: Vec): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Pearson correlation of two equal-length samples. */
function pearson(a: number[], b: number[]): number {
  const m = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < m; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= m;
  mb /= m;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < m; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  const den = Math.sqrt(saa * sbb);
  return den > 0 ? sab / den : 0;
}

/**
 * A random 3x3 orthonormal frame via Gram-Schmidt on seeded Gaussian columns --
 * the deterministic stand-in for the notebook's `np.linalg.qr(randn(3,3))`.
 */
function orthoFrame(rng: Rng): Mat {
  const cols: Vec[] = [rng.gaussVec(3), rng.gaussVec(3), rng.gaussVec(3)];
  const q: Vec[] = [];
  for (const c of cols) {
    let v = c.slice();
    for (const u of q) {
      const d = dot3(v, u);
      v = [v[0] - d * u[0], v[1] - d * u[1], v[2] - d * u[2]];
    }
    const len = Math.hypot(v[0], v[1], v[2]) || 1;
    q.push([v[0] / len, v[1] / len, v[2] / len]);
  }
  // store as columns: Q[row][col]
  return [
    [q[0][0], q[1][0], q[2][0]],
    [q[0][1], q[1][1], q[2][1]],
    [q[0][2], q[1][2], q[2][2]],
  ];
}

/** A round-number grid step roughly dividing `max` into ~5 intervals. */
function niceStep(max: number): number {
  const raw = max / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return step * mag;
}

function btnStyle(theme: Theme): string {
  return `font:13px ${theme.mono};color:${theme.fg};background:#3c3c3c;border:1px solid #555555;border-radius:8px;padding:7px 14px;cursor:pointer;`;
}
