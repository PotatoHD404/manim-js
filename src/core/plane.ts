import { type Pt, ROMAN_FONT, strokeLine } from "./draw";
import type { Theme } from "./theme";

export interface View {
  width: number;
  height: number;
  /** Pixels per world unit. */
  scale: number;
  cx: number;
  cy: number;
  toScreen(p: Pt): Pt;
  toWorld(p: Pt): Pt;
}

/**
 * A square, equal-aspect view centred on the world origin, sized so that
 * `[-half, half]` fits inside the canvas with `pad` pixels of margin. Equal
 * aspect matters: a circle of data must read as a circle.
 */
export function fitView(width: number, height: number, half: number, pad = 26): View {
  const scale = Math.max((Math.min(width, height) / 2 - pad) / half, 1);
  const cx = width / 2;
  const cy = height / 2;
  return {
    width,
    height,
    scale,
    cx,
    cy,
    toScreen: ([x, y]) => [cx + x * scale, cy - y * scale],
    toWorld: ([px, py]) => [(px - cx) / scale, (cy - py) / scale],
  };
}

/**
 * Draw a number plane: faint grid at unit steps, brighter axes through the
 * origin, and a few tick labels. This is the stage every scene stands on.
 */
export function numberPlane(
  ctx: CanvasRenderingContext2D,
  view: View,
  theme: Theme,
  opts: { step?: number; labels?: boolean; alpha?: number } = {},
): void {
  const step = opts.step ?? 1;
  const a = opts.alpha ?? 1;
  const xMax = view.cx / view.scale;
  const yMax = view.cy / view.scale;
  const kx = Math.ceil(xMax / step);
  const ky = Math.ceil(yMax / step);

  // Faint half-step lines first (Manim's faded sub-grid), then the major grid
  // on top, then the two bright axes — a layered look that keeps the grid as a
  // backdrop so the bright data objects read as the figure.
  const minor = step / 2;
  const mkx = Math.ceil(xMax / minor);
  const mky = Math.ceil(yMax / minor);
  for (let i = -mkx; i <= mkx; i++) {
    if (i % 2 === 0) continue;
    const x = i * minor;
    strokeLine(ctx, view.toScreen([x, -yMax]), view.toScreen([x, yMax]), theme.grid, 1, 0.3 * a);
  }
  for (let j = -mky; j <= mky; j++) {
    if (j % 2 === 0) continue;
    const y = j * minor;
    strokeLine(ctx, view.toScreen([-xMax, y]), view.toScreen([xMax, y]), theme.grid, 1, 0.3 * a);
  }
  for (let i = -kx; i <= kx; i++) {
    if (i === 0) continue;
    const x = i * step;
    strokeLine(ctx, view.toScreen([x, -yMax]), view.toScreen([x, yMax]), theme.grid, 1.2, 0.6 * a);
  }
  for (let j = -ky; j <= ky; j++) {
    if (j === 0) continue;
    const y = j * step;
    strokeLine(ctx, view.toScreen([-xMax, y]), view.toScreen([xMax, y]), theme.grid, 1.2, 0.6 * a);
  }

  strokeLine(ctx, view.toScreen([-xMax, 0]), view.toScreen([xMax, 0]), theme.axis, 1.6, 1 * a);
  strokeLine(ctx, view.toScreen([0, -yMax]), view.toScreen([0, yMax]), theme.axis, 1.6, 1 * a);

  if (opts.labels !== false) {
    ctx.fillStyle = theme.tick;
    ctx.font = `13px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = -kx; i <= kx; i++) {
      if (i === 0) continue;
      const p = view.toScreen([i * step, 0]);
      ctx.fillText(String(i * step), p[0], p[1] + 5);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let j = -ky; j <= ky; j++) {
      if (j === 0) continue;
      const p = view.toScreen([0, j * step]);
      ctx.fillText(String(j * step), p[0] - 7, p[1]);
    }
  }
}

/** World-space ellipse: centre + two orthonormal axes scaled by given radii. */
export function ellipseWorld(center: Pt, axisA: Pt, axisB: Pt, ra: number, rb: number, segments = 96): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    const c = Math.cos(t) * ra;
    const s = Math.sin(t) * rb;
    pts.push([
      center[0] + c * axisA[0] + s * axisB[0],
      center[1] + c * axisA[1] + s * axisB[1],
    ]);
  }
  return pts;
}
