"""Test for updater.kalman module"""

import pytest
import numpy as np

from datetime import datetime, timedelta

from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import PDA
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction)
from stonesoup.types.state import GaussianState, SqrtGaussianState
from stonesoup.updater.probability import PDAUpdater


'''@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement, predictor, updater",
    [
        (   # Schmidt Kalman
            PDAUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]])),
            ExtendedKalmanPredictor,
            ExtendedKalmanUpdater
        ),
    ],
    ids=["pdaupdater"]
)'''
#def test_pda(UpdaterClass, measurement_model, prediction, measurement, predictor, updater):
def test_pda():
    start_time = datetime.now()
    track = Track([GaussianState(np.array([[-6.45], [0.7]]),
                                 np.array([[4.1123, 0.0013], [0.0013, 0.0365]]), start_time)])
    detection1 = Detection(np.array([[-5]]), timestamp= start_time+ timedelta(seconds=1))
    detection2 = Detection(np.array([[-8]]), timestamp= start_time+ timedelta(seconds=1))
    detections = {detection1, detection2}

    transition_model = ConstantVelocity(0.005)
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model)

    hypothesiser = PDAHypothesiser(predictor=predictor, updater=updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9)

    #   multihypothesis = \
    #       hypothesiser.hypothesise(track, detections, timestamp)

    data_associator = PDA(hypothesiser=hypothesiser)

    hypotheses = data_associator.associate({track}, detections, start_time + timedelta(seconds=1))
    hypotheses = hypotheses[track]

    pdaupdater = PDAUpdater(measurement_model)

    posterior_state = pdaupdater.update(hypotheses, gm_method=False)
    print(posterior_state)


