# coding: utf-8
import datetime

import numpy as np
import pytest

from ...models.transition.categorical import CategoricalTransitionModel
from ...models.transition.tests.test_categorical import create_categorical, \
    create_categorical_matrix
from ...predictor.categorical import HMMPredictor
from ...types.prediction import StatePrediction, CategoricalStatePrediction
from ...types.state import State, CategoricalState


@pytest.mark.parametrize('ndim_state', (2, 3, 4))
def test_hmm_predictor(ndim_state):
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    F = create_categorical_matrix(ndim_state, ndim_state).T  # normalised columns
    Q = np.eye(ndim_state)
    transition_model = CategoricalTransitionModel(F, Q)

    # Test state prediction
    predictor = HMMPredictor(transition_model)

    # Test wrong prior type
    with pytest.raises(ValueError, match="Prior must be a categorical state type"):
        predictor.predict(prior=State([1, 2, 3]), timestamp=new_timestamp)

    priors = [CategoricalState(create_categorical(ndim_state),
                               timestamp=timestamp)
              for _ in range(5)]
    for prior in priors:
        for next_time in [new_timestamp, None]:
            # Evaluate prediction
            Fx = F @ prior.state_vector
            eval_prediction = StatePrediction(Fx / sum(Fx), timestamp=next_time)

            # Test state prediction
            prediction = predictor.predict(prior=prior, timestamp=next_time)

            # Assert presence of transition model
            assert hasattr(prediction, 'transition_model')

            # Test prediction state vector
            assert np.allclose(prediction.state_vector,
                               eval_prediction.state_vector,
                               0,
                               atol=1.e-14)
            assert prediction.timestamp == next_time

            # Test prediction type
            assert isinstance(prediction, CategoricalStatePrediction)
            assert prediction.category_names == prior.category_names
