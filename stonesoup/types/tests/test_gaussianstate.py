# -*- coding: utf-8 -*-

import numpy as np

from stonesoup.types.base import GaussianState


def test_gaussianstate():
    """ GaussianState Type test """

    empty_state = GaussianState()
    assert(empty_state.mean is None
           and empty_state.covar is None
           and empty_state.ndim is None)

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    state = GaussianState(mean, covar)

    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])

    new_mean = np.array([[-6.45], [0.7]]) * 1e2
    new_covar = np.array([[4.1123, 0.0013],
                          [0.0013, 0.0365]])

    state.mean = new_mean
    state.covar = new_covar

    assert(np.array_equal(new_mean, state.mean))
    assert(np.array_equal(new_covar, state.covar))
    assert(state.ndim == new_mean.shape[0])
