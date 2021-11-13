# -*- coding: utf-8 -*-
import numpy as np
from os import path
from datetime import datetime
from typing import Sequence

import pytest

from stonesoup.detector import beamformers
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


@pytest.fixture(params=[beamformers.CaponBeamformer, beamformers.RJMCMCBeamformer])
def detector(request):
    data_file = path.join(path.dirname(__file__), "fixed_target_example.csv")
    # define static sensor positions for two time windows
    X = [0, 0, 0, 10, 10, 10, 20, 20, 20]
    Y = [0, 10, 20, 0, 10, 20, 0, 10, 20]
    Z = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sensor_pos_sequence = []
    for i in range(0, 2):
        sensor_pos_xyz = []
        for j in range(0, 9):
            sensor_pos_xyz.append(X[j])
            sensor_pos_xyz.append(Y[j])
            sensor_pos_xyz.append(Z[j])
        sensor_pos_sequence.append(StateVector(sensor_pos_xyz))
    sensor_pos: Sequence[StateVector] = sensor_pos_sequence
    extra_args = {'seed': 1234} if request.param is beamformers.RJMCMCBeamformer else {}
    return request.param(data_file, sensor_loc=sensor_pos, fs=2000, omega=50, wave_speed=1481,
                         window_size=800, **extra_args)


def test_fixed_target_beamformer(detector):
    truth = [0.8, 0.2]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                              ConstantVelocity(0.01)])

    covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=covar)

    predictor = KalmanPredictor(transition_model)

    updater = KalmanUpdater(measurement_model)

    prior = GaussianState([[0.5], [0], [0.5], [0]], np.diag([1, 0, 1, 0]),
                          timestamp=datetime.now())

    track = Track()

    for timestep, detections in detector:
        for detection in detections:
            # check beamformer accuracy
            assert(np.abs(detection.state_vector[0] - truth[0]) < 0.4)
            assert(np.abs(detection.state_vector[1] - truth[1]) < 0.1)
            prediction = predictor.predict(prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)  # Group prediction + measurement
            post = updater.update(hypothesis)
            track.append(post)
            prior = track[-1]
