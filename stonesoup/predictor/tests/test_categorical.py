from datetime import datetime, timedelta
import copy
import numpy as np

from ...models.transition.categorical import MarkovianTransitionModel
from ...predictor.categorical import HMMPredictor
from ...types.prediction import CategoricalStatePrediction
from ...types.state import CategoricalState


def test_hmm_predictor():
    F = np.array([[50, 5, 30],
                  [25, 90, 30],
                  [25, 5, 30]])
    model = MarkovianTransitionModel(F)

    predictor = HMMPredictor(model)

    now = datetime.now()
    interval = timedelta(seconds=1)
    future = now + interval

    prior = CategoricalState([80, 10, 10], timestamp=now)

    # Test interval
    assert predictor._predict_over_interval(prior, future) == interval
    assert predictor._predict_over_interval(prior, None) is None
    prior.timestamp = None
    assert predictor._predict_over_interval(prior, future) is None
    prior.timestamp = now

    # Test predict (with timestamp)
    prediction = predictor.predict(prior=prior, timestamp=future)
    assert isinstance(prediction, CategoricalStatePrediction)
    expected_vector = model.function(prior, time_interval=interval, noise=False)
    assert np.allclose(prediction.state_vector, expected_vector)

    # Test predict (without timestamp)
    prediction = predictor.predict(prior=prior, timestamp=None)
    assert isinstance(prediction, CategoricalStatePrediction)
    assert np.allclose(prediction.state_vector, prior.state_vector)

    # Test predict (without prior timestamp)
    prior = copy.deepcopy(prior)  # predictor caches .predict returns
    prior.timestamp = None
    prediction = predictor.predict(prior=prior, timestamp=future)
    assert isinstance(prediction, CategoricalStatePrediction)
    assert np.allclose(prediction.state_vector, prior.state_vector)
