# -*- coding: utf-8 -*-
import pytest

from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction)


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
        def get_measurement_prediction(self, state_prediction,
                                       measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(state_prediction.state_vector,
                                                 state_prediction.covar,
                                                 state_prediction.timestamp)
    return TestGaussianUpdater()
