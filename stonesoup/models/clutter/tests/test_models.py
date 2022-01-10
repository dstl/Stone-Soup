# -*- coding: utf-8 -*-
import numpy as np
import pytest
import datetime

from ..clutter import ClutterModel
from ...measurement.nonlinear import CartesianToBearingRange, CartesianToElevationBearingRange
from ....types.detection import Clutter
from ....types.array import StateVector
from ....types.state import State


@pytest.mark.parametrize("clutter_rate", [5, 10])
@pytest.mark.parametrize("dist_params, meas_model", [
        (((-50, 50), (-50, 50)),
            CartesianToBearingRange(ndim_state=6,
                                    mapping=[0, 2],
                                    noise_covar=np.eye(2))),
        (((-50, 50), (-50, 50), (-50, 50)),
            CartesianToElevationBearingRange(ndim_state=6,
                                             mapping=[0, 2, 4],
                                             noise_covar=np.eye(4)))
    ],
    ids=["Clutter2D", "Clutter3D"]
)
def test_model(clutter_rate, dist_params, meas_model):
    model_test = ClutterModel(clutter_rate=clutter_rate,
                              distribution=np.random.default_rng().uniform,
                              dist_params=dist_params)

    model_test.measurement_model = meas_model

    # Test on 0 groundtruths
    clutter = model_test.function(set())
    assert not clutter

    # Test on 1 groundtruth
    truth1 = State(StateVector([1, 1, 1, 1, 1, 1]), timestamp=datetime.datetime.now())
    clutter = model_test.function({truth1})
    assert np.all(isinstance(c, Clutter) for c in clutter)
    assert np.all(c.ndim == meas_model.ndim_meas for c in clutter)

    # Test on +1 groundtruth
    truth2 = State(StateVector([1, 1, 1, 1, 1, 1]), timestamp=datetime.datetime.now())
    clutter = model_test.function({truth1, truth2})
    assert np.all(isinstance(c, Clutter) for c in clutter)
    assert np.all(c.ndim == meas_model.ndim_meas for c in clutter)


@pytest.mark.parametrize("clutter_rate", [5])
@pytest.mark.parametrize("dist_params, meas_model", [
        (((-50, 50), (-50, 50)),
            CartesianToBearingRange(ndim_state=6,
                                    mapping=[0, 2],
                                    noise_covar=np.eye(2))),
        (((-50, 50), (-50, 50), (-50, 50)),
            CartesianToElevationBearingRange(ndim_state=6,
                                             mapping=[0, 2, 4],
                                             noise_covar=np.eye(4)))
    ],
    ids=["Clutter2D", "Clutter3D"]
)
def test_ndim(clutter_rate, dist_params, meas_model):
    model_test = ClutterModel(clutter_rate=clutter_rate,
                              distribution=np.random.default_rng().uniform,
                              dist_params=dist_params)
    assert model_test.ndim == len(dist_params)

    model_test.measurement_model = meas_model
    assert model_test.ndim == meas_model.ndim_meas
