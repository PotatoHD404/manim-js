import { clamp, valueTracker } from "../core/anim";
import { el, responsiveCanvas } from "../core/dom";
import { disc, type Pt, polyline, ROMAN_FONT, strokeLine } from "../core/draw";
import { texInto } from "../core/katex";
import { type Theme, withTheme } from "../core/theme";

export interface SpectrumOptions {
  /** Eigenvalue spectrum, descending. Defaults to the post's 10-d example. */
  spectrum?: number[];
  /** "scree" = eigenvalue bars + cumulative line; "reconstruction" = error/retained vs q. */
  view?: "scree" | "reconstruction";
  /** Retained-variance threshold line (default 0.95). */
  threshold?: number;
  /** Initial number of kept components. */
  q?: number;
  theme?: Partial<Theme>;
}

export interface SpectrumApi {
  setQ(q: number): void;
  destroy(): void;
}

/**
 * Scenes — "how many components?". The eigenvalue spectrum drives both the scree
 * plot and the reconstruction-error curve; a single slider chooses how many
 * components to keep, and the kept/discarded split, retained variance, and the
 * sum-of-discarded-eigenvalues error all update together.
 */
export function createSpectrum(target: HTMLElement, options: SpectrumOptions = {}): SpectrumApi {
  const theme = withTheme(options.theme);
  const spec = (options.spectrum ?? [8, 4.5, 2, 1.1, 0.7, 0.4, 0.25, 0.15, 0.08, 0.03]).slice();
  const view = options.view ?? "scree";
  const thr = options.threshold ?? 0.95;
  const p = spec.length;
  const total = spec.reduce((a, b) => a + b, 0) || 1;
  const cum: number[] = [];
  let acc = 0;
  for (const v of spec) {
    acc += v;
    cum.push(acc / total);
  }
  let q95 = p;
  for (let i = 0; i < p; i++) {
    if (cum[i] >= thr) {
      q95 = i + 1;
      break;
    }
  }
  let q = clamp(options.q ?? q95, 1, p);
  let qf = q; // eased display position of the cutoff marker
  let destroyed = false;
  const cumAt = (pos: number): number => {
    const i = clamp(pos - 1, 0, p - 1);
    const lo = Math.floor(i);
    const hi = Math.min(lo + 1, p - 1);
    return cum[lo] * (1 - (i - lo)) + cum[hi] * (i - lo);
  };

  const root = el("div", { style: "display:flex;flex-direction:column;gap:12px;" });
  const panel = el("div", {
    style: `position:relative;width:100%;border-radius:14px;overflow:hidden;background:${theme.bg};`,
  });
  root.append(panel);

  const controls = el("div", { style: "display:flex;align-items:center;gap:12px;flex-wrap:wrap;" });
  const label = el("label", { style: `font:13px ${theme.mono};color:${theme.muted};` });
  label.textContent = "components kept  q";
  const slider = el("input", {
    type: "range",
    min: "1",
    max: String(p),
    step: "1",
    value: String(q),
    style: "flex:1;min-width:140px;max-width:240px;",
    "aria-label": "number of principal components kept",
  });
  const readout = el("div", { style: `font:14px ${ROMAN_FONT};color:${theme.fg};min-width:8ch;` });
  controls.append(label, slider, readout);
  root.append(controls);
  target.append(root);

  const rc = responsiveCanvas(panel, 1.9, () => draw());

  // The cutoff marker eases to the chosen q (the bars switch kept/discarded at
  // the integer boundary; the marker glides).
  const qTracker = valueTracker(q, (v) => {
    qf = v;
    draw();
  });

  function draw(): void {
    const { ctx, width: W, height: H } = rc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const ml = 40;
    const mr = 18;
    const mt = 18;
    const mb = 30;
    const x = (i: number) => ml + ((i + 0.5) / p) * (W - ml - mr);
    const yFrac = (f: number) => H - mb - clamp(f, 0, 1) * (H - mt - mb);

    // axes
    strokeLine(ctx, [ml, mt], [ml, H - mb], theme.axis, 1.4, 1);
    strokeLine(ctx, [ml, H - mb], [W - mr, H - mb], theme.axis, 1.4, 1);

    // threshold line (retained-variance fraction)
    const yThr = yFrac(thr);
    polyline(ctx, [[ml, yThr], [W - mr, yThr]], { color: theme.gold, width: 1.2, alpha: 0.7, dash: [6, 6] });
    ctx.fillStyle = theme.gold;
    ctx.font = `12px ${ROMAN_FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${Math.round(thr * 100)}%`, ml + 6, yThr - 3);

    if (view === "scree") {
      const maxEig = spec[0];
      const yEig = (v: number) => H - mb - (v / maxEig) * (H - mt - mb);
      const bw = ((W - ml - mr) / p) * 0.55;
      for (let i = 0; i < p; i++) {
        const cx = x(i);
        const top = yEig(spec[i]);
        const kept = i < q;
        ctx.fillStyle = kept ? theme.blue : theme.red;
        ctx.globalAlpha = kept ? 0.85 : 0.5;
        ctx.fillRect(cx - bw / 2, top, bw, H - mb - top);
        ctx.globalAlpha = 1;
      }
      // cumulative variance line on the [0,1] (threshold) scale
      const line: Pt[] = cum.map((c, i) => [x(i), yFrac(c)]);
      polyline(ctx, line, { color: theme.green, width: 2.2, alpha: 1 });
      for (let i = 0; i < p; i++) disc(ctx, [x(i), yFrac(cum[i])], 2.6, theme.green, 1);
    } else {
      // reconstruction: retained variance (green, up) and error fraction (red, down)
      const retained: Pt[] = cum.map((c, i) => [x(i), yFrac(c)]);
      const errorFrac: Pt[] = cum.map((c, i) => [x(i), yFrac(1 - c)]);
      polyline(ctx, errorFrac, { color: theme.red, width: 2.2, alpha: 1 });
      polyline(ctx, retained, { color: theme.green, width: 2.2, alpha: 1 });
      for (let i = 0; i < p; i++) {
        disc(ctx, [x(i), yFrac(cum[i])], 2.4, theme.green, 1);
        disc(ctx, [x(i), yFrac(1 - cum[i])], 2.4, theme.red, 1);
      }
    }

    // q marker (eased position)
    const qx = x(qf - 1);
    strokeLine(ctx, [qx, mt], [qx, H - mb], theme.cream, 1.4, 0.8);
    disc(ctx, [qx, yFrac(cumAt(qf))], 4, theme.cream, 1);

    // x ticks (component index)
    ctx.fillStyle = theme.tick;
    ctx.font = `11px ${ROMAN_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < p; i++) ctx.fillText(String(i + 1), x(i), H - mb + 5);

    const errAbs = total - cum[q - 1] * total;
    texInto(
      readout,
      view === "scree"
        ? `q=${q},\\ \\text{retained}=${(cum[q - 1] * 100).toFixed(1)}\\%`
        : `q=${q},\\ \\varepsilon^2=\\sum_{k>q}\\lambda_k=${errAbs.toFixed(2)}`,
    );
    if (document.activeElement !== slider) slider.value = String(q);
  }

  function setQ(nq: number, animate = true): void {
    q = clamp(Math.round(nq), 1, p);
    if (animate) qTracker.set(q, true);
    else qTracker.jump(q);
  }

  slider.addEventListener("input", () => setQ(Number(slider.value), false));

  rc.canvas.tabIndex = 0;
  rc.canvas.setAttribute("role", "img");
  rc.canvas.setAttribute(
    "aria-label",
    view === "scree"
      ? "Scree plot: eigenvalue bars in descending order with a cumulative-variance curve; a slider chooses how many components to keep."
      : "Reconstruction error and retained variance as a function of the number of kept components.",
  );
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") setQ(q - 1);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") setQ(q + 1);
    else return;
    e.preventDefault();
  };
  rc.canvas.addEventListener("keydown", onKey);

  draw();
  if (typeof document !== "undefined" && document.fonts) {
    document.fonts.ready.then(() => {
      if (!destroyed) draw();
    });
  }

  return {
    setQ,
    destroy() {
      destroyed = true;
      qTracker.stop();
      rc.canvas.removeEventListener("keydown", onKey);
      rc.destroy();
      root.remove();
    },
  };
}
