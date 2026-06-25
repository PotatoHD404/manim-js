/**
 * The visual language — Manim's actual palette, pulled from 3b1b/manim source
 * (manimlib/constants.py + default_config.yml). Manim's dark-grey board
 * (#333333), its grid in BLUE_D, the luminous BLUE_C for data, GOLD for the
 * "answer" objects (eigenvectors — the punchline), RED_C for error, GREEN_C for
 * the traced quantity. Everything glows faintly, like chalk on a board.
 */
export interface Theme {
  bg: string;
  /** Faint number-plane grid lines. */
  grid: string;
  /** The two main axes through the origin. */
  axis: string;
  /** Tick / axis number labels. */
  tick: string;
  fg: string;
  muted: string;
  /** Data points and the covariance ellipse — the luminous blue. */
  blue: string;
  /** Eigenvectors / principal axes — the gold "answer". */
  gold: string;
  /** Residuals / reconstruction error. */
  red: string;
  /** The objective being traced out (variance vs angle). */
  green: string;
  /** The movable variable (the candidate direction the user controls). */
  cream: string;
  mono: string;
}

export const defaultTheme: Theme = {
  bg: "#000000", // ManimCommunity default background_color = BLACK
  grid: "#29ABCA", // BLUE_D — NumberPlane background_line_style
  axis: "#DDDDDD", // GREY_A — crisp neutral axes (Manim Axes default ≈ white)
  tick: "#BBBBBB", // GREY_B
  fg: "#DDDDDD", // GREY_A — manim default stroke / text
  muted: "#888888", // GREY_C
  blue: "#58C4DD", // BLUE_C — the canonical manim blue (data)
  gold: "#F0AC5F", // GOLD_C — the "answer" objects (eigenvectors)
  red: "#FC6255", // RED_C — residuals / error
  green: "#83C167", // GREEN_C — the traced quantity
  cream: "#F3EFE2", // the movable candidate direction
  mono: '"SFMono-Regular", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace',
};

export function withTheme(overrides?: Partial<Theme>): Theme {
  return overrides ? { ...defaultTheme, ...overrides } : defaultTheme;
}
