# -*- coding: utf-8 -*-
import pytest

from ...types.detection import MissedDetection
from ...types.hypothesis import SingleDistanceHypothesis
from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction)
from ...hypothesiser.probability import PDAHypothesiser


@pytest.fixture()
def hypothesiser():
    class TestGaussianHypothesiser:
        def hypothesise(self, track, detections, timestamp):
            hypotheses = list()
            for detection in detections:
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     track.covar * 2,
                                                     detection.timestamp)
                measurement_prediction =\
                    GaussianMeasurementPrediction(prediction.state_vector,
                                                  prediction.covar,
                                                  prediction.timestamp)
                distance = abs(track.state_vector - detection.state_vector)

                hypotheses.append(SingleDistanceHypothesis(
                    prediction, detection, distance, measurement_prediction))

            prediction = GaussianStatePrediction(track.state_vector + 1,
                                                 track.covar * 2, timestamp)
            hypotheses.append(
                SingleDistanceHypothesis(
                    prediction, MissedDetection(timestamp=timestamp), 10))
            return hypotheses
    return TestGaussianHypothesiser()


@pytest.fixture()
def probability_predictor():
    class TestGaussianPredictor:
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return GaussianStatePrediction(prior.state_vector + 1,
                                           prior.covar * 2, timestamp)
    return TestGaussianPredictor()


@pytest.fixture()
def probability_updater():
    class TestGaussianUpdater:
        def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(state_prediction.state_vector,
                                                 state_prediction.covar,
                                                 state_prediction.timestamp)
    return TestGaussianUpdater()


@pytest.fixture()
def probability_hypothesiser(probability_predictor, probability_updater):

    return PDAHypothesiser(probability_predictor, probability_updater,
                           clutter_spatial_density=1.2e-2,
                           prob_detect=0.9, prob_gate=0.99)
