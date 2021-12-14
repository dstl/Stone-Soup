# -*- coding: utf-8 -*-
import numpy as np
import pytest
from pytest import approx
from scipy.stats import multivariate_normal
import datetime

from ..clutter import ClutterModel
from ...measurement.nonlinear import CartesianToElevationBearingRange
from ....types.detection import Clutter
from ....types.array import StateVector
from ....types.state import State

@pytest.mark.parametrize(
    "model_class",
    [ClutterModel]
)
@pytest.mark.parametrize(
    "ModelClass, clutter_rate, dist,\
     dist_params",
    [
        (   # 2D meas
            ClutterModel,
            5,
            np.random.default_rng().uniform,
            ((-50, 50), (-50, 50)),
        ),
        (   # 3D meas
            ClutterModel,
            5,
            np.random.default_rng().uniform,
            ((-50, 50), (-50, 50), (-50, 50)),
        )
    ],
    ids=["Clutter2D", "Clutter3D"]
)
def test_model(ModelClass, clutter_rate, dist, dist_params):
    model_test = ModelClass(clutter_rate=clutter_rate,
                            distribution=dist,
                            dist_params=dist_params)

    meas_model = CartesianToElevationBearingRange(
            ndim_state=6,
            mapping=(0, 2, 4),
            noise_covar=np.eye(3))
    model_test.measurement_model = meas_model

    truth = State(StateVector([1, 1, 1]), timestamp=datetime.now)
    clutter = model_test.function(truth)

    assert np.all(isinstance(c, Clutter) for c in clutter)
    

def test_ndim(ModelClass, clutter_rate, dist, dist_params):
    model_test = ModelClass(clutter_rate=clutter_rate,
                            distribution=dist,
                            dist_params=dist_params)
    assert model_test.ndim == len(dist_params)

    meas_model = CartesianToElevationBearingRange(
            ndim_state=6,
            mapping=(0, 2, 4),
            noise_covar=np.eye(3))
    model_test.measurement_model = meas_model
    assert model_test.ndim == meas_model.ndim_meas
