# pca-viz (published as manim-js) — instructions

## What it is

Sibling repo; the local directory is still named `pca-viz` but the package, README and public repo
were rebranded **manim-js** on 2026-06-25. A small framework-agnostic canvas engine (`src/core/`)
that reproduces the ManimCommunity look (black board, cyan number plane, KaTeX labels, eased
`ValueTracker` motion) as real-time, draggable, framework-agnostic web components — not a video
renderer, not a Manim port. Public: `github.com/PotatoHD404/manim-js`, `main`. 32 scenes across 5
scene packs (PCA, MLE/MAP, cross-entropy, Fisher information, matrix completion) power all the
interactive figures on a consumer blog, a separate private repo
(`github.com/PotatoHD404/mmkuznecov-blog-interactive`, `master`) that vendors this library's UMD
build. No host, no backend, no Dokploy — a static library + a static-HTML consumer.

## Build / test / run

```bash
pnpm install                 # pnpm v10 blocks esbuild's postinstall by default —
                              # package.json already sets pnpm.onlyBuiltDependencies:["esbuild"]
pnpm dev                     # vite dev server
pnpm build                   # vite build + tsc -p tsconfig.build.json -> dist/manim-js.{js,umd.cjs}
pnpm build:demo              # vite build --config vite.demo.config.ts
pnpm typecheck               # tsc --noEmit
```

No test suite as of 2026-09-13 — verification is typecheck + build + visual screenshot review (a
headless preview never fires `requestAnimationFrame`, so animated sweeps/morphs must be verified by
scrubbing to an end-state, e.g. dragging the eigenwarp slider, not by watching them play).

## Deploy

None in the CI sense. When the user asks to publish an update that the blog picks up:

```bash
pnpm build
cp dist/manim-js.umd.cjs ../mmkuznecov.github.io/assets/manim-js.umd.js   # or wherever the consumer vendors it
# only once the user has explicitly asked: commit + push BOTH repos (this one and the blog repo)
```

GitHub Pages from the blog's private repo needs GitHub Pro — the consumer repo is not on Pages by
default.

## Traps

- Never edit `src/index.ts` directly when adding a scene — `AUTHORING.md` is the contract: one
  self-contained `src/<topic>/<name>.ts` scene + one `src/elements/<tag>.ts` element wrapper per
  figure; only the two reference scenes (`src/pca/projection.ts`, `src/pca/spectrum.ts`) are meant
  to be copied from.
- Never hardcode a hex color in a scene — use `theme.ts`'s ManimCommunity-exact fields (`bg`, `grid`
  BLUE_D, `blue` BLUE_C data, `gold` GOLD_C answer objects, `red` RED_C error, `green` GREEN_C traced
  quantity, `cream` movable variable). All on-canvas math/numbers go through `mathLabel`/`ROMAN_FONT`
  from `draw.ts`, never a plain/mono canvas font.
- Continuous controls (slider drag, canvas drag) call `tracker.jump(v)` (direct/sync); discrete
  actions (keyboard step, snap, sweep/morph play) call `tracker.set(v, true)` (eased glide) — mixing
  these up produces motion that doesn't match Manim's `.animate` semantics.
- A headless CI/preview environment's `requestAnimationFrame` is either suspended (backgrounded tab)
  or throttled to ~8fps — eased/animated paths look "stuck" there even when they work correctly in
  a real browser.
- Per-scene batches were built by spawning one agent per figure; agents cannot screenshot, so a
  label-overlap/collision pass (e.g. two on-canvas labels drawing on top of each other) is a known
  gap on later batches (cross-entropy, Fisher, matrix-completion scene packs) that has not had a
  full visual review as of the last recorded session.

## Deeper notes

- `project_side_projects` (§ pca-viz / manim-js) — the D3/Three.js → canvas-engine pivot, the
  ManimCommunity palette research, the blog-integration and per-graph agent rollout history, and the
  rebrand to manim-js.
