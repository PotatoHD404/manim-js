# Authoring a scene (contract for per-figure implementation)

Every interactive figure is ONE self-contained scene built on the shared engine,
styled to look exactly like **ManimCommunity** and animated like Manim. Read the
two reference scenes before writing — match their structure exactly:

- `src/pca/projection.ts` (number-plane scene: ellipse, arrows, KaTeX labels, slider+drag+keyboard, sweep, variance trace)
- `src/pca/spectrum.ts` (chart scene: bars/curves, q slider, KaTeX readout)

## Files to create (and ONLY these — never edit `src/index.ts`)

1. `src/<topic>/<name>.ts` — `export function create<Name>(target: HTMLElement, options): <Name>Api`
2. `src/elements/<tag>.ts` — a tiny `HTMLElement` wrapper reading attributes (copy `src/elements/pca-spectrum.ts`)

Return the tag name + factory name in your result; the lead wires registration.

## The engine (import from `../core/...`)

- `theme` (`withTheme`): `bg #000000`, `grid` BLUE_D, `axis` GREY_A, `blue` BLUE_C (data), `gold` GOLD_C (the "answer" object), `red` RED_C (error), `green` GREEN_C (traced quantity), `cream` (the movable variable). NEVER hardcode hex — use theme fields.
- `plane.ts`: `fitView(W,H,half)` → equal-aspect view; `numberPlane(ctx,view,theme,{step,labels,alpha})`; `ellipseWorld(center,axisA,axisB,ra,rb)`.
- `draw.ts`: `glow`, `strokeLine`, `polyline`, `disc`, `arrow` (triangular tip), `mathLabel(ctx,text,p,{color,size,sub,italic,glow})` — renders in KaTeX Computer-Modern; `ROMAN_FONT`/`MATH_FONT`. Use `mathLabel`/`ROMAN_FONT` for ALL on-canvas math/numbers, never mono.
- `equation.ts`: `liveEquation(tex,theme,color)` (overlay formula + live value), `staticEquation`, `overlayLayer('tl'|'tr'|'bl')`.
- `dom.ts`: `el`, `responsiveCanvas(parent,aspect,onResize)`, `draggable(target,onMove,hooks)`.
- `anim.ts`: `valueTracker`, `smooth`, `ticker`, `clamp`, `prefersReducedMotion`.
- `math/linalg.ts` (covariance, jacobiEigen, pca), `math/rng.ts` (`Rng`, `correlatedCloud`).

## Styling rules (ManimCommunity-exact)

- Black board, the cyan number plane recedes, the bright BLUE_C data + GOLD answer objects carry the contrast. Faint chalk `glow` on the key objects only.
- All math labels in KaTeX serif-italic via `mathLabel`/overlay equations. Tick numbers in `ROMAN_FONT`.
- Arrows are filled triangles; one cool accent + one warm "answer" accent; restrained palette.

## Animation rules (Manim, but interactive)

- Build a `valueTracker` for the scene's main scalar; `render()` reads its value.
- **Continuous controls (slider, canvas drag) → `tracker.jump(v)`** (direct, real-time, synchronous render).
- **Discrete/triggered actions (keyboard step, snap, a "play"/sweep/morph button) → `tracker.set(v, true)`** (eased glide), or drive a `ticker` with `smooth` for a full traversal.
- Always provide: a slider, canvas drag, focusable canvas (`tabIndex=0`, `role="img"`, descriptive `aria-label`), arrow-key control, and `prefers-reduced-motion` → jump to the end state.
- First paint is synchronous; redraw on `document.fonts.ready`; guard with a `destroyed` flag; `destroy()` stops trackers/tickers, removes listeners, disconnects observers, removes the root.

## Quality bar

Detailed and beautiful, but focused — one clear idea per scene, no bloat. The math
must match the post's notebook exactly (it is the source of truth). Real
computation, never pre-baked frames. Seeded RNG so the figure is reproducible; a
re-seed control is welcome where the original shows Monte-Carlo randomness.
