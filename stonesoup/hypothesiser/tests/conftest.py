# -*- coding: utf-8 -*-
import pytest

from stonesoup.predictor import Predictor
from stonesoup.predictor.categorical import HMMPredictor
from stonesoup.updater import Updater
from stonesoup.updater.categorical import HMMUpdater
from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction, CategoricalStatePrediction,
    CategoricalMeasurementPrediction)


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


@pytest.fixture()
def class_predictor():
    class TestClassPredictor(HMMPredictor):
        @property
        def transition_model(self):
            pass

        def predict(self, prior, timestamp=None, **kwargs):
            return CategoricalStatePrediction(prior.state_vector, timestamp=timestamp)

    return TestClassPredictor()


@pytest.fixture()
def class_updater():
    class TestClassUpdater(HMMUpdater):
        @property
        def measurement_model(self):
            pass

        def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):
            return CategoricalMeasurementPrediction(state_prediction.state_vector,
                                                    timestamp=state_prediction.timestamp)

    return TestClassUpdater()
