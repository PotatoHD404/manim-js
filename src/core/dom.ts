type Attrs = Record<string, string | number | boolean | undefined>;

/** Tiny hyperscript helper so interactives can build their own UI without a framework. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs?: Attrs,
  children?: (Node | string)[],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === false) continue;
      if (k === "style") node.setAttribute("style", String(v));
      else if (k in node) (node as Record<string, unknown>)[k] = v;
      else node.setAttribute(k, String(v));
    }
  }
  if (children) {
    for (const c of children) node.append(c);
  }
  return node;
}

export interface ResponsiveCanvas {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  /** CSS-pixel width/height of the drawing surface. */
  width: number;
  height: number;
  /** Re-read size and reset the DPR transform; call before a redraw on resize. */
  sync(): void;
  destroy(): void;
}

/**
 * A canvas that tracks its container's width at a fixed aspect ratio, handles
 * devicePixelRatio for crisp lines, and calls `onResize` after every layout
 * change. The context is pre-transformed to CSS pixels.
 */
export function responsiveCanvas(
  parent: HTMLElement,
  aspect: number,
  onResize: (c: ResponsiveCanvas) => void,
): ResponsiveCanvas {
  const canvas = el("canvas", {
    style: "display:block;width:100%;touch-action:none;",
  });
  parent.append(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas context unavailable");

  const api: ResponsiveCanvas = {
    canvas,
    ctx,
    width: 0,
    height: 0,
    sync() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = parent.clientWidth || 320;
      const h = Math.round(w / aspect);
      api.width = w;
      api.height = h;
      canvas.style.height = `${h}px`;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    },
    destroy() {
      ro.disconnect();
      canvas.remove();
    },
  };

  const ro = new ResizeObserver(() => {
    api.sync();
    onResize(api);
  });
  ro.observe(parent);
  api.sync();
  return api;
}

/**
 * Pointer drag helper. `onMove` receives canvas-local coordinates. Handles
 * pointer capture so a drag started inside the element keeps tracking outside it.
 */
export function draggable(
  target: HTMLElement,
  onMove: (x: number, y: number, ev: PointerEvent) => void,
  hooks?: { onStart?: () => void; onEnd?: () => void },
): () => void {
  let active = false;
  const local = (ev: PointerEvent) => {
    const r = target.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top] as const;
  };
  const down = (ev: PointerEvent) => {
    active = true;
    target.setPointerCapture(ev.pointerId);
    hooks?.onStart?.();
    const [x, y] = local(ev);
    onMove(x, y, ev);
  };
  const move = (ev: PointerEvent) => {
    if (!active) return;
    const [x, y] = local(ev);
    onMove(x, y, ev);
  };
  const up = () => {
    if (!active) return;
    active = false;
    hooks?.onEnd?.();
  };
  target.addEventListener("pointerdown", down);
  target.addEventListener("pointermove", move);
  target.addEventListener("pointerup", up);
  target.addEventListener("pointercancel", up);
  return () => {
    target.removeEventListener("pointerdown", down);
    target.removeEventListener("pointermove", move);
    target.removeEventListener("pointerup", up);
    target.removeEventListener("pointercancel", up);
  };
}
