# -*- coding: utf-8 -*-
import numpy as np
from os import path
from datetime import datetime

from stonesoup.detector import beamformers
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


def fixed_target_beamformer_test():

    data_file = path.join(path.dirname(__file__), "fixed_target_example.csv")
    truth = [0.8, 0.2]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                              ConstantVelocity(0.01)])

    covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=covar)

    predictor = KalmanPredictor(transition_model)

    updater = KalmanUpdater(measurement_model)

    prior = GaussianState([[0.5], [0], [0.5], [0]], np.diag([1, 0, 1, 0]),
                          timestamp=datetime.now())

    detector1 = beamformers.capon(data_file, sensor_loc="0 0 0; 0 10 0; 0 20 0; 10 0 0; 10 10 0; \
        10 20 0; 20 0 0; 20 10 0; 20 20 0", fs=2000, omega=50, wave_speed=1481)
    detector2 = beamformers.rjmcmc(data_file, sensor_loc="0 0 0; 0 10 0; 0 20 0; 10 0 0; 10 10 0; \
        10 20 0; 20 0 0; 20 10 0; 20 20 0", fs=2000, omega=50, wave_speed=1481)

    track1 = Track()
    track2 = Track()

    for timestep, detections in detector2:
        for detection in detections:
            # check beamformer accuracy
            assert(np.abs(detection.state_vector[0] - truth[0]) < 0.4)
            assert(np.abs(detection.state_vector[1] - truth[1]) < 0.1)
            prediction = predictor.predict(prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)  # Group prediction + measurement
            post = updater.update(hypothesis)
            track2.append(post)
            prior = track2[-1]

    for timestep, detections in detector1:
        for detection in detections:
            # check beamformer accuracy
            assert(np.abs(detection.state_vector[0] - truth[0]) < 0.4)
            assert(np.abs(detection.state_vector[1] - truth[1]) < 0.1)
            prediction = predictor.predict(prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)  # Group prediction + measurement
            post = updater.update(hypothesis)
            track1.append(post)
            prior = track1[-1]
