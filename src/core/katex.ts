import katex from "katex";

/**
 * Render a TeX string to an HTML string. Errors are swallowed (rendered in
 * red inline) so a typo in a label never blanks the whole explorable.
 *
 * Consumers must load KaTeX's stylesheet once for glyph positioning, e.g.
 * `import "katex/dist/katex.min.css"` or the matching CDN `<link>`.
 */
export function tex(src: string, display = false): string {
  return katex.renderToString(src, { throwOnError: false, displayMode: display });
}

export function texInto(target: HTMLElement, src: string, display = false): void {
  target.innerHTML = tex(src, display);
}
