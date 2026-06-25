export type Easing = (t: number) => number;

/**
 * Manim's default rate function (`smooth`) — Perlin's smootherstep,
 * `6t⁵ − 15t⁴ + 10t³`, zero velocity AND acceleration at both ends. This is the
 * single most recognizable thing about how Manim moves; everything eases through
 * it unless told otherwise.
 */
export const smooth: Easing = (t) => {
  const s = clamp(t, 0, 1);
  return s * s * s * (10 + s * (-15 + 6 * s));
};

/** Manim `rush_into` — start slow, end fast (second half of smooth). */
export const rushInto: Easing = (t) => 2 * smooth(0.5 * clamp(t, 0, 1));

/** Manim `rush_from` — start fast, settle slow (first half of smooth). */
export const rushFrom: Easing = (t) => 2 * smooth(0.5 * clamp(t, 0, 1) + 0.5) - 1;

/** Manim `there_and_back` — go to 1 at the midpoint, return to 0. */
export const thereAndBack: Easing = (t) => {
  const s = clamp(t, 0, 1);
  return smooth(s < 0.5 ? 2 * s : 2 * (1 - s));
};

export const easeInOutCubic: Easing = (t) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export const easeOutCubic: Easing = (t) => 1 - Math.pow(1 - t, 3);

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export interface Tracker {
  /** The currently displayed (eased) value. */
  readonly value: number;
  /** Move the target; the displayed value glides to it (Manim's ValueTracker.animate). */
  set(target: number, animate?: boolean): void;
  /** Jump with no easing. */
  jump(target: number): void;
  stop(): void;
}

/**
 * A Manim-style value tracker: an underlying target plus a displayed value that
 * eases toward it every frame. This is what makes interaction feel like Manim —
 * dragging, sliding, snapping all *glide* to the new state instead of jumping,
 * yet stay real-time (the target follows input immediately; only the visual
 * eases, with a short time constant). Auto-respects prefers-reduced-motion.
 */
export function valueTracker(
  initial: number,
  onChange: (value: number) => void,
  opts: { timeConstant?: number; epsilon?: number } = {},
): Tracker {
  const tc = opts.timeConstant ?? 0.09;
  const eps = opts.epsilon ?? 1e-4;
  let cur = initial;
  let tgt = initial;
  let cancel: (() => void) | null = null;

  const ensure = () => {
    if (cancel) return;
    cancel = ticker((dt) => {
      // Frame-rate-independent exponential approach toward the target.
      const k = 1 - Math.exp(-dt / tc);
      cur += (tgt - cur) * k;
      if (Math.abs(tgt - cur) < eps) {
        cur = tgt;
        onChange(cur);
        cancel = null;
        return false;
      }
      onChange(cur);
    });
  };

  return {
    get value() {
      return cur;
    },
    set(target, animate = true) {
      tgt = target;
      if (!animate || prefersReducedMotion()) {
        cancel?.();
        cancel = null;
        cur = tgt;
        onChange(cur);
        return;
      }
      ensure();
    },
    jump(target) {
      cancel?.();
      cancel = null;
      cur = tgt = target;
      onChange(cur);
    },
    stop() {
      cancel?.();
      cancel = null;
    },
  };
}

export function prefersReducedMotion(): boolean {
  return (
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** Shortest signed angular delta from `a` to `b`, both in radians. */
export function angleDelta(a: number, b: number): number {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

/**
 * A requestAnimationFrame ticker. The callback receives elapsed seconds and
 * returns `false` to stop. Returns a cancel handle. Pauses cleanly when the tab
 * is hidden by leaning on rAF's own throttling.
 */
export function ticker(cb: (dt: number, elapsed: number) => boolean | void): () => void {
  let raf = 0;
  let last = 0;
  let start = 0;
  let alive = true;

  const frame = (now: number) => {
    if (!alive) return;
    if (!start) {
      start = now;
      last = now;
    }
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    const cont = cb(dt, (now - start) / 1000);
    if (cont === false) {
      alive = false;
      return;
    }
    raf = requestAnimationFrame(frame);
  };

  raf = requestAnimationFrame(frame);
  return () => {
    alive = false;
    cancelAnimationFrame(raf);
  };
}

/**
 * Tween a scalar from `from` to `to` over `duration` seconds. The 3b1b "watch it
 * settle" move: nothing teleports, everything eases. Returns a cancel handle.
 */
export function tween(opts: {
  from: number;
  to: number;
  duration: number;
  ease?: Easing;
  onUpdate: (value: number) => void;
  onDone?: () => void;
}): () => void {
  const ease = opts.ease ?? smooth;
  if (prefersReducedMotion()) {
    opts.onUpdate(opts.to);
    opts.onDone?.();
    return () => {};
  }
  return ticker((_dt, elapsed) => {
    const t = clamp(elapsed / opts.duration, 0, 1);
    opts.onUpdate(lerp(opts.from, opts.to, ease(t)));
    if (t >= 1) {
      opts.onDone?.();
      return false;
    }
  });
}
