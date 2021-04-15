# -*- coding: utf-8 -*-
import pytest

from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction, StatePrediction, MeasurementPrediction,
    StateMeasurementPrediction)


@pytest.fixture()
def predictor():
    class TestGaussianPredictor:
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return GaussianStatePrediction(prior.state_vector + 1,
                                           prior.covar * 2, timestamp)
    return TestGaussianPredictor()


@pytest.fixture()
def updater():
    class TestGaussianUpdater:
        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(state_prediction.state_vector,
                                                 state_prediction.covar,
                                                 state_prediction.timestamp)
    return TestGaussianUpdater()


@pytest.fixture()
def class_predictor():
    class TestClassPredictor:
        def predict(self, prior, timestamp=None, **kwargs):
            return StatePrediction(prior.state_vector)
    return TestClassPredictor()


@pytest.fixture()
def class_updater():
    class TestClassUpdater:
        def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):
            return StateMeasurementPrediction(state_prediction.state_vector,
                                              state_prediction.timestamp)
    return TestClassUpdater()
