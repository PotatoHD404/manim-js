export * as core from "./core";
export { defaultTheme, withTheme, type Theme } from "./core/theme";

export { createProjection, type ProjectionApi, type ProjectionOptions } from "./pca/projection";
export { createEigenwarp, type EigenwarpApi, type EigenwarpOptions } from "./pca/eigenwarp";
export { createSpectrum, type SpectrumApi, type SpectrumOptions } from "./pca/spectrum";
export { createDistances } from "./pca/distances";
export { createBetaPosterior } from "./mle/beta-posterior";
export { createGaussianShrinkage } from "./mle/gaussian-shrinkage";
export { createPriorWashout } from "./mle/prior-washout";
export { createZeroCount } from "./mle/zero-count";

export { PcaProjectionElement } from "./elements/pca-projection";
export { PcaEigenwarpElement } from "./elements/pca-eigenwarp";
export { PcaSpectrumElement } from "./elements/pca-spectrum";
export { PcaDistancesElement } from "./elements/pca-distances";
export { MleBetaBernoulliElement } from "./elements/mle-beta-bernoulli";
export { MleGaussianShrinkageElement } from "./elements/mle-gaussian-shrinkage";
export { MlePriorWashoutElement } from "./elements/mle-prior-washout";
export { MleZeroCountElement } from "./elements/mle-zero-count";

import { MleBetaBernoulliElement } from "./elements/mle-beta-bernoulli";
import { MleGaussianShrinkageElement } from "./elements/mle-gaussian-shrinkage";
import { MlePriorWashoutElement } from "./elements/mle-prior-washout";
import { MleZeroCountElement } from "./elements/mle-zero-count";
import { PcaDistancesElement } from "./elements/pca-distances";
import { PcaEigenwarpElement } from "./elements/pca-eigenwarp";
import { PcaProjectionElement } from "./elements/pca-projection";
import { PcaSpectrumElement } from "./elements/pca-spectrum";

const REGISTRY: Array<[string, CustomElementConstructor]> = [
  ["pca-projection", PcaProjectionElement],
  ["pca-eigenwarp", PcaEigenwarpElement],
  ["pca-spectrum", PcaSpectrumElement],
  ["pca-distances", PcaDistancesElement],
  ["mle-beta-bernoulli", MleBetaBernoulliElement],
  ["mle-gaussian-shrinkage", MleGaussianShrinkageElement],
  ["mle-prior-washout", MlePriorWashoutElement],
  ["mle-zero-count", MleZeroCountElement],
];

/**
 * Register the custom elements. Called automatically on import in a browser; the
 * guard keeps it idempotent and safe under SSR / repeated imports.
 */
export function registerElements(): void {
  if (typeof customElements === "undefined") return;
  for (const [tag, ctor] of REGISTRY) {
    if (!customElements.get(tag)) customElements.define(tag, ctor);
  }
}

registerElements();
