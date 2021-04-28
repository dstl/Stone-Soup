# coding: utf-8
import datetime

import numpy as np
import pytest

from ...models.transition.categorical import CategoricalTransitionModel
from ...models.transition.tests.test_categorical import create_random_multinomial
from ...predictor.categorical import HMMPredictor
from ...types.prediction import StatePrediction, CategoricalStatePrediction


def get_transition_models(num_models, ndim_state):
    models = set()
    for _ in range(num_models):
        rows = list()
        for _ in range(ndim_state):
            rows.append(create_random_multinomial(ndim_state).state_vector.flatten())
        FT = np.array(rows)
        Q = np.eye(ndim_state)
        models.add(CategoricalTransitionModel(FT.T, Q))
    return models


@pytest.mark.parametrize('ndim_state', (1, 2, 3, 4))
def test_hmm_predictor(ndim_state):
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    transition_models = get_transition_models(10, ndim_state)

    # test state prediction
    for transition_model in transition_models:
        F = transition_model.transition_matrix

        predictor = HMMPredictor(transition_model)
        for _ in range(3):
            # define prior state
            prior = create_random_multinomial(ndim_state)
            prior.timestamp = timestamp

            # evaluate prediction
            Fx = F @ prior.state_vector
            eval_prediction = StatePrediction(Fx / sum(Fx), timestamp=new_timestamp)

            # test state prediction
            prediction = predictor.predict(prior=prior, timestamp=new_timestamp)

            # Assert presence of transition model
            assert hasattr(prediction, 'transition_model')

            # test prediction state vector
            assert np.allclose(prediction.state_vector,
                               eval_prediction.state_vector,
                               0,
                               atol=1.e-14)
            assert prediction.timestamp == new_timestamp

            # test prediction type
            assert isinstance(prediction, CategoricalStatePrediction)
            assert prediction.num_categories == prior.num_categories
            assert prediction.category_names == prior.category_names
