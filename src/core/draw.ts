export type Pt = [number, number];

/** Computer-Modern math italic, borrowed from KaTeX — the Manim Tex look. */
export const MATH_FONT = '"KaTeX_Math", "Latin Modern Math", "CMU Serif", serif';
/** Upright roman, for numbers and operators. */
export const ROMAN_FONT = '"KaTeX_Main", "Latin Modern Roman", "CMU Serif", serif';

/** Run `body` with a soft additive glow, like chalk on a dark board. */
export function glow(ctx: CanvasRenderingContext2D, color: string, blur: number, body: () => void): void {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  body();
  ctx.restore();
}

/**
 * A math label drawn in KaTeX's Computer-Modern font, so a single symbol on the
 * canvas matches the rendered Tex in the overlays — the difference between
 * "looks like Manim" and "looks like a chart". Variables are italic; pass
 * `italic: false` for upright numbers. An optional subscript is set smaller.
 */
export function mathLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  p: Pt,
  opts: {
    color: string;
    size?: number;
    italic?: boolean;
    sub?: string;
    align?: CanvasTextAlign;
    baseline?: CanvasTextBaseline;
    glow?: number;
  },
): void {
  const size = opts.size ?? 16;
  const italic = opts.italic !== false;
  const family = italic ? MATH_FONT : ROMAN_FONT;
  ctx.save();
  ctx.fillStyle = opts.color;
  ctx.textAlign = opts.align ?? "left";
  ctx.textBaseline = opts.baseline ?? "alphabetic";
  if (opts.glow) {
    ctx.shadowColor = opts.color;
    ctx.shadowBlur = opts.glow;
  }
  ctx.font = `${italic ? "italic " : ""}${size}px ${family}`;
  ctx.fillText(text, p[0], p[1]);
  if (opts.sub) {
    const w = ctx.measureText(text).width;
    ctx.font = `italic ${size * 0.7}px ${MATH_FONT}`;
    const base = opts.baseline === "middle" ? p[1] + size * 0.18 : p[1] + size * 0.16;
    ctx.textAlign = "left";
    ctx.fillText(opts.sub, p[0] + (opts.align === "right" ? 0 : w) + size * 0.04, base);
  }
  ctx.restore();
}

export function strokeLine(
  ctx: CanvasRenderingContext2D,
  a: Pt,
  b: Pt,
  color: string,
  width = 1,
  alpha = 1,
): void {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

export function polyline(
  ctx: CanvasRenderingContext2D,
  pts: Pt[],
  opts: { color: string; width?: number; alpha?: number; closed?: boolean; fill?: string; dash?: number[] },
): void {
  if (pts.length < 2) return;
  ctx.globalAlpha = opts.alpha ?? 1;
  ctx.lineWidth = opts.width ?? 1.5;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  if (opts.dash) ctx.setLineDash(opts.dash);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (opts.closed) ctx.closePath();
  if (opts.fill) {
    ctx.fillStyle = opts.fill;
    ctx.fill();
  }
  ctx.strokeStyle = opts.color;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

export function disc(
  ctx: CanvasRenderingContext2D,
  p: Pt,
  r: number,
  color: string,
  alpha = 1,
): void {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

/** An arrow from `a` to `b` with a filled triangular head, Manim-style. */
export function arrow(
  ctx: CanvasRenderingContext2D,
  a: Pt,
  b: Pt,
  opts: { color: string; width?: number; head?: number; alpha?: number },
): void {
  const w = opts.width ?? 3;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  // Manim's tip is a fixed-size filled triangle, capped at a fraction of the
  // arrow length (max_tip_length_to_length_ratio) so short vectors stay in
  // proportion instead of growing a head from the stroke width.
  const head = Math.min(opts.head ?? 13, len * 0.4);
  const ux = dx / len;
  const uy = dy / len;
  const base: Pt = [b[0] - ux * head, b[1] - uy * head];

  ctx.globalAlpha = opts.alpha ?? 1;
  ctx.strokeStyle = opts.color;
  ctx.fillStyle = opts.color;
  ctx.lineCap = "round";
  ctx.lineWidth = w;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(base[0], base[1]);
  ctx.stroke();

  const nx = -uy;
  const ny = ux;
  const hw = head * 0.5;
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(base[0] + nx * hw, base[1] + ny * hw);
  ctx.lineTo(base[0] - nx * hw, base[1] - ny * hw);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}
