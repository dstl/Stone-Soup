# coding: utf-8

import datetime
import pytest
import numpy as np

from stonesoup.types.state import State
from stonesoup.types.array import StateVector
from stonesoup.models.transition.linear import ConstantVelocity, ConstantAcceleration, \
    CombinedLinearGaussianTransitionModel
from stonesoup.predictor.alphabeta import AlphaBetaPredictor


def test_alphabeta():

    # Test wrong model correctly rejected
    transition_model = ConstantAcceleration(noise_diff_coeff=0.01)
    with pytest.raises(TypeError):
        predictor = AlphaBetaPredictor(transition_model)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define correct predictor this time
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    predictor = AlphaBetaPredictor(transition_model)

    # Define prior state
    prior = State(StateVector([-6.45, 0.7]), timestamp=timestamp)
    # Manually get the answer
    outputstate = StateVector([-6.45 + timediff*0.7, 0.7])

    prediction = predictor.predict(prior, timestamp=new_timestamp)

    assert np.all(prediction.state_vector == outputstate)
