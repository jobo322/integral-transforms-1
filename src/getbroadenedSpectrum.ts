import { BorderType, fftConvolution } from 'ml-convolution';
import { getShape1D, Shape1D } from 'ml-peak-shape-generator';

import { yNorm } from './utilities/yNorm';

export function getbroadenedSpectrum(array: number[], options: Options = {}) {
  const {
    shape = { kind: 'gaussian', sd: 1.2 },
    kernelWidth = 7,
    normalized = false,
    height = 1,
  } = options;
  const kernelBasis = getShape1D({ ...shape, fwhm: kernelWidth });
  const kernel = kernelBasis.getData({
    length: kernelWidth,
    height,
  });
  const result = fftConvolution(array, kernel, 'CONSTANT' as BorderType);
  return normalized ? yNorm(result) : result;
}

interface Options {
  shape?: Shape1D;
  kernelWidth?: number;
  normalized?: boolean;
  height?: number;
}
