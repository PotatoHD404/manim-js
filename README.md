# manim-js

A **Manim-style interactive-graphics engine for the web**. It reproduces the
[ManimCommunity](https://www.manim.community/) look — a black board, a cyan
`BLUE_D` number plane, glowing data, gold "answer" objects, Computer-Modern
(KaTeX) labels, smootherstep motion — but it is **not** a Manim port and does
**not** render video. The scenes are live, draggable, keyboard-controllable, and
real-time. Pure client-side, framework-agnostic [custom elements](https://developer.mozilla.org/en-US/docs/Web/API/Web_components);
drop them into any static page.

> Independent and Manim-*inspired* — not affiliated with or endorsed by 3Blue1Brown or the Manim Community. The palette and rate functions are taken from [`3b1b/manim`](https://github.com/3b1b/manim) / [`ManimCommunity/manim`](https://github.com/ManimCommunity/manim) source (background `#000000`, grid `BLUE_D #29ABCA`, data `BLUE_C #58C4DD`, `smooth(t)=6t⁵−15t⁴+10t³`).

## What's inside

A small reusable **engine** (`src/core/`) plus **scene packs** built on it:

| Pack | Scenes (custom elements) |
| --- | --- |
| PCA | `<pca-projection>` · `<pca-eigenwarp>` · `<pca-spectrum>` · `<pca-distances>` |
| MLE / MAP | `<mle-beta-bernoulli>` · `<mle-gaussian-shrinkage>` · `<mle-prior-washout>` · `<mle-zero-count>` |

Each scene is backed by real, seeded computation (covariance eigendecomposition,
Beta/Normal conjugate updates, …), not pre-baked frames.

## The engine (`src/core/`)

- `theme.ts` — the ManimCommunity palette (exact source values).
- `plane.ts` — `fitView` + `numberPlane` (layered cyan grid) + `ellipseWorld`.
- `draw.ts` — `glow`, `arrow` (triangular tip), `disc`, `polyline`, `mathLabel` (renders in KaTeX's Computer-Modern font on the canvas).
- `equation.ts` — KaTeX overlay equations with a live value.
- `anim.ts` — Manim rate functions (`smooth`, …), `ticker`, and **`valueTracker`** (a Manim `ValueTracker`-style eased follow). Continuous controls `jump()` (direct); discrete actions `set(v, true)` (eased glide).
- `math/` — linear algebra (covariance, Jacobi eigendecomposition, PCA) + seeded RNG.

## Use it

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css" />
<script type="module">
  import "manim-js"; // registers the custom elements
</script>

<pca-projection seed="42" points="220"></pca-projection>
<mle-beta-bernoulli heads="7" tails="3" alpha="2" beta="2"></mle-beta-bernoulli>
```

Or drive the factories directly:

```ts
import { createProjection, createBetaPosterior } from "manim-js";
const api = createProjection(document.querySelector("#fig1")!, { seed: 1 });
api.snapToPrincipalAxis();
// api.destroy() when tearing the page down
```

KaTeX's stylesheet must be present once on the page — it supplies both the overlay equations and the Computer-Modern font used on the canvas.

## Authoring a new scene

See [`AUTHORING.md`](AUTHORING.md) — every figure is one self-contained scene +
element built on the engine, following the styling and jump-vs-ease animation
rules. The two reference scenes are `src/pca/projection.ts` and `src/pca/spectrum.ts`.

## Develop

```bash
pnpm install
pnpm dev        # demo at http://localhost:5174
pnpm typecheck
pnpm build      # ESM + UMD bundle + .d.ts into dist/
```

## License

MIT
