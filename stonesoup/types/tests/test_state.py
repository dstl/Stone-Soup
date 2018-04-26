# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from stonesoup.types import GaussianState


def test_gaussianstate():
    """ GaussianState Type test """

    with pytest.raises(TypeError):
        GaussianState()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3

    state = GaussianState(datetime.datetime.now(), mean, covar)

    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])
