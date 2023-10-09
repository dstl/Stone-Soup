import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.prediction import TaggedWeightedGaussianStatePrediction


@pytest.fixture()
def measurement_model():
    return LinearGaussian(ndim_state=2, mapping=[0],
                          noise_covar=np.array([[0.04]]))


@pytest.fixture()
def prediction():
    return TaggedWeightedGaussianStatePrediction(
        np.array([[-6.45], [0.7]]),
        np.array([[4.1123, 0.0013],
                  [0.0013, 0.0365]]),
        weight=1,
        tag=1,
        timestamp=datetime.datetime(2022, 9, 16))


@pytest.fixture()
def measurement():
    return Detection(np.array([[-6.23]]),
                     timestamp=datetime.datetime(2022, 9, 16))
