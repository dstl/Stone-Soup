# coding: utf-8
import datetime

import numpy as np
import pytest

from stonesoup.models.transition.classification import \
    BasicTimeInvariantClassificationTransitionModel
from stonesoup.predictor.classification import ClassificationPredictor
from stonesoup.types.array import StateVector
from ...types.prediction import StatePrediction
from ...types.state import State


def create_random_multinomial(length):
    total = 0
    sv = list()
    for i in range(length - 1):
        x = np.random.uniform(0, 1 - total)
        sv.append(x)
        total += x
    sv.append(1 - total)
    sv = StateVector(sv)
    return State(sv)


def get_transition_models(num_models, ndim_state):
    models = set()
    for _ in range(num_models):
        rows = list()
        for _ in range(ndim_state):
            rows.append(create_random_multinomial(ndim_state).state_vector.flatten())
        FT = np.array(rows)
        models.add(BasicTimeInvariantClassificationTransitionModel(FT.T))
    return models


@pytest.mark.parametrize('ndim_state', (1, 2, 3, 4))
def test_classification_predictor(ndim_state):
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    transition_models = get_transition_models(10, ndim_state)

    # test state prediction
    for transition_model in transition_models:
        F = transition_model.transition_matrix

        predictor = ClassificationPredictor(transition_model)
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

            assert np.allclose(prediction.state_vector,
                               eval_prediction.state_vector,
                               0,
                               atol=1.e-14)
            assert prediction.timestamp == new_timestamp
