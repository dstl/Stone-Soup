# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ...predictor import Predictor
from ...predictor.categorical import HMMPredictor
from ...types.prediction import (
    GaussianMeasurementPrediction, GaussianStatePrediction, CategoricalStatePrediction,
    CategoricalMeasurementPrediction)
from ...updater import Updater
from ...updater.categorical import HMMUpdater


@pytest.fixture()
def predictor():
    class TestGaussianPredictor(Predictor):
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return GaussianStatePrediction(prior.state_vector + 1,
                                           prior.covar * 2, timestamp)

        @property
        def transition_model(self):
            return None

    return TestGaussianPredictor()


@pytest.fixture()
def updater():
    class TestGaussianUpdater(Updater):
        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(state_prediction.state_vector,
                                                 state_prediction.covar,
                                                 state_prediction.timestamp)

        def update(self, hypothesis, **kwargs):
            pass

        @property
        def measurement_model(self):
            return None

    return TestGaussianUpdater()


@pytest.fixture()
def dummy_category_predictor():
    class DummyCategoricalPredictor(HMMPredictor):

        @property
        def transition_model(self):
            pass

        def predict(self, prior, timestamp=None, **kwargs):
            """Return the same state vector."""
            return CategoricalStatePrediction(prior.state_vector, timestamp=timestamp)

    return DummyCategoricalPredictor()


@pytest.fixture()
def dummy_category_updater():
    class DummyCategoricalUpdater(HMMUpdater):

        @property
        def measurement_model(self):
            pass

        def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):
            """Return the first two state vector elements, normalised."""
            vector = state_prediction.state_vector[:2]
            vector = vector / np.sum(vector)
            return CategoricalMeasurementPrediction(vector,
                                                    timestamp=state_prediction.timestamp)

    return DummyCategoricalUpdater()
