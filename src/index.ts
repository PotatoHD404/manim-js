export * as core from "./core";
export { defaultTheme, withTheme, type Theme } from "./core/theme";

// PCA pack
export { createProjection, type ProjectionApi, type ProjectionOptions } from "./pca/projection";
export { createEigenwarp, type EigenwarpApi, type EigenwarpOptions } from "./pca/eigenwarp";
export { createSpectrum, type SpectrumApi, type SpectrumOptions } from "./pca/spectrum";
export { createDistances } from "./pca/distances";

// MLE / MAP pack
export { createBetaPosterior } from "./mle/beta-posterior";
export { createGaussianShrinkage } from "./mle/gaussian-shrinkage";
export { createPriorWashout } from "./mle/prior-washout";
export { createZeroCount } from "./mle/zero-count";

// Cross-entropy pack
export { createCrossEntropyLoss } from "./ce/cross-entropy-loss";
export { createCeDecomposition } from "./ce/ce-decomposition";
export { createForwardReverseKl } from "./ce/forward-reverse-kl";
export { createMseVsBceLoss } from "./ce/mse-vs-bce-loss";
export { createMseVsBceGradient } from "./ce/mse-vs-bce-gradient";

// Fisher information pack
export { createFisherTwoForms } from "./fisher/two-forms";
export { createFisherCurvature } from "./fisher/curvature";
export { createFisherCrb } from "./fisher/crb";
export { createFisherKlHessian } from "./fisher/kl-hessian";
export { createFisherMatrixEllipse } from "./fisher/matrix-ellipse";
export { createFisherReparam } from "./fisher/reparam";
export { createFisherGaussNewton } from "./fisher/gauss-newton";

// Matrix-completion pack
export { createMatrixAr1 } from "./matrix/ar1";
export { createBlockVsSymbol } from "./matrix/block-vs-symbol";
export { createCommonBit } from "./matrix/common-bit";
export { createMatrixDempster } from "./matrix/dempster";
export { createMatrixDichotomy } from "./matrix/dichotomy";
export { createMatrixEffectiveRank } from "./matrix/effective-rank";
export { createMatrixEquicorrelated } from "./matrix/equicorrelated";
export { createMatrixNoisyBsc } from "./matrix/noisy-bsc";
export { createMatrixRandomEnsembles } from "./matrix/random-ensembles";
export { createRankQuantization } from "./matrix/rank-quantization";
export { createSpectralFlatness } from "./matrix/spectral-flatness";
export { createSymbolError } from "./matrix/symbol-error";

import { CeCrossEntropyLossElement } from "./elements/ce-cross-entropy-loss";
import { CeDecompositionElement } from "./elements/ce-decomposition";
import { CeForwardReverseKlElement } from "./elements/ce-forward-reverse-kl";
import { CeMseVsBceGradientElement } from "./elements/ce-mse-vs-bce-gradient";
import { CeMseVsBceLossElement } from "./elements/ce-mse-vs-bce-loss";
import { FisherCrbElement } from "./elements/fisher-crb";
import { FisherCurvatureElement } from "./elements/fisher-curvature";
import { FisherGaussNewtonElement } from "./elements/fisher-gauss-newton";
import { FisherKlHessianElement } from "./elements/fisher-kl-hessian";
import { FisherMatrixEllipseElement } from "./elements/fisher-matrix-ellipse";
import { FisherReparamElement } from "./elements/fisher-reparam";
import { FisherTwoFormsElement } from "./elements/fisher-two-forms";
import { MatrixAr1Element } from "./elements/matrix-ar1";
import { MatrixBlockVsSymbolElement } from "./elements/matrix-block-vs-symbol";
import { MatrixCommonBitElement } from "./elements/matrix-common-bit";
import { MatrixDempsterElement } from "./elements/matrix-dempster";
import { MatrixDichotomyElement } from "./elements/matrix-dichotomy";
import { MatrixEffectiveRankElement } from "./elements/matrix-effective-rank";
import { MatrixEquicorrelatedElement } from "./elements/matrix-equicorrelated";
import { MatrixNoisyBscElement } from "./elements/matrix-noisy-bsc";
import { MatrixRandomEnsemblesElement } from "./elements/matrix-random-ensembles";
import { MatrixRankQuantizationElement } from "./elements/matrix-rank-quantization";
import { MatrixSpectralFlatnessElement } from "./elements/matrix-spectral-flatness";
import { MatrixSymbolErrorElement } from "./elements/matrix-symbol-error";
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
  ["ce-cross-entropy-loss", CeCrossEntropyLossElement],
  ["ce-decomposition", CeDecompositionElement],
  ["ce-forward-reverse-kl", CeForwardReverseKlElement],
  ["ce-mse-vs-bce-loss", CeMseVsBceLossElement],
  ["ce-mse-vs-bce-gradient", CeMseVsBceGradientElement],
  ["fisher-two-forms", FisherTwoFormsElement],
  ["fisher-curvature", FisherCurvatureElement],
  ["fisher-crb", FisherCrbElement],
  ["fisher-kl-hessian", FisherKlHessianElement],
  ["fisher-matrix-ellipse", FisherMatrixEllipseElement],
  ["fisher-reparam", FisherReparamElement],
  ["fisher-gauss-newton", FisherGaussNewtonElement],
  ["matrix-ar1", MatrixAr1Element],
  ["matrix-block-vs-symbol", MatrixBlockVsSymbolElement],
  ["matrix-common-bit", MatrixCommonBitElement],
  ["matrix-dempster", MatrixDempsterElement],
  ["matrix-dichotomy", MatrixDichotomyElement],
  ["matrix-effective-rank", MatrixEffectiveRankElement],
  ["matrix-equicorrelated", MatrixEquicorrelatedElement],
  ["matrix-noisy-bsc", MatrixNoisyBscElement],
  ["matrix-random-ensembles", MatrixRandomEnsemblesElement],
  ["matrix-rank-quantization", MatrixRankQuantizationElement],
  ["matrix-spectral-flatness", MatrixSpectralFlatnessElement],
  ["matrix-symbol-error", MatrixSymbolErrorElement],
];

export {
  CeCrossEntropyLossElement,
  CeDecompositionElement,
  CeForwardReverseKlElement,
  CeMseVsBceGradientElement,
  CeMseVsBceLossElement,
  FisherCrbElement,
  FisherCurvatureElement,
  FisherGaussNewtonElement,
  FisherKlHessianElement,
  FisherMatrixEllipseElement,
  FisherReparamElement,
  FisherTwoFormsElement,
  MatrixAr1Element,
  MatrixBlockVsSymbolElement,
  MatrixCommonBitElement,
  MatrixDempsterElement,
  MatrixDichotomyElement,
  MatrixEffectiveRankElement,
  MatrixEquicorrelatedElement,
  MatrixNoisyBscElement,
  MatrixRandomEnsemblesElement,
  MatrixRankQuantizationElement,
  MatrixSpectralFlatnessElement,
  MatrixSymbolErrorElement,
  MleBetaBernoulliElement,
  MleGaussianShrinkageElement,
  MlePriorWashoutElement,
  MleZeroCountElement,
  PcaDistancesElement,
  PcaEigenwarpElement,
  PcaProjectionElement,
  PcaSpectrumElement,
};

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
