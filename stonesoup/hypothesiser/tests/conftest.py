import numpy as np
import pytest

from ...predictor import Predictor
from ...predictor.categorical import HMMPredictor
from ...types.prediction import (
    MeasurementPrediction, Prediction, CategoricalStatePrediction,
    CategoricalMeasurementPrediction)
from ...types.update import Update
from ...updater import Updater
from ...updater.categorical import HMMUpdater


@pytest.fixture()
def predictor():
    class TestGaussianPredictor(Predictor):
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return Prediction.from_state(
                prior,
                state_vector=prior.state_vector + 1,
                covar=prior.covar * 2,
                timestamp=timestamp)

        @property
        def transition_model(self):
            return None

    return TestGaussianPredictor()


@pytest.fixture()
def updater():
    class TestGaussianUpdater(Updater):
        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return MeasurementPrediction.from_state(state_prediction)

        def update(self, hypothesis, **kwargs):
            return Update.from_state(hypothesis.prediction, hypothesis=hypothesis)

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

        def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
            """Return the first two state vector elements, normalised."""
            vector = predicted_state.state_vector[:2]
            vector = vector / np.sum(vector)
            return CategoricalMeasurementPrediction(vector,
                                                    timestamp=predicted_state.timestamp)

    return DummyCategoricalUpdater()
