# -*- coding: utf-8 -*-
import pytest

from ...predictor import Predictor
from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction)
from ...updater import Updater


@pytest.fixture()
def predictor():
    class TestGaussianPredictor(Predictor):
        @property
        def transition_model(self):
            pass

        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return GaussianStatePrediction(prior.state_vector + 1,
                                           prior.covar * 2, timestamp)

    return TestGaussianPredictor()


@pytest.fixture()
def updater():
    class TestGaussianUpdater(Updater):
        @property
        def measurement_model(self):
            pass

        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(state_prediction.state_vector,
                                                 state_prediction.covar,
                                                 state_prediction.timestamp)

        def update(self, hypothesis, **kwargs):
            pass

    return TestGaussianUpdater()
