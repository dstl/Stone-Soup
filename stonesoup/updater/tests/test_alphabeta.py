# -*- coding: utf-8 -*-
"""Test for updater.alphabeta module"""
import pytest
from datetime import timedelta
import numpy as np

from stonesoup.types.array import StateVector
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.state import State
from stonesoup.types.prediction import StatePrediction, StateMeasurementPrediction
from stonesoup.updater.alphabeta import AlphaBetaUpdater
from stonesoup.types.hypothesis import SingleHypothesis


@pytest.mark.parametrize(
    "measurement_model, prediction, measurement, alpha, beta",
    [
        (   # Standard Alpha-Beta
            LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=0),
            StatePrediction(StateVector([-6.45, 0.7, -6.45, 0.7])),
            Detection(StateVector([-6.23, -6.23])),
            0.9,
            0.3
        )
    ],
    ids=["standard"]
)
def test_alphabeta(measurement_model, prediction, measurement, alpha, beta):

    # Time delta
    timediff = timedelta(seconds=2)

    # Calculate evaluation variables - converts
    # to measurement from prediction space
    eval_measurement_prediction = StateMeasurementPrediction(
       measurement_model.matrix()@prediction.state_vector)

    eval_posterior_position = prediction.state_vector[[0, 2]] + \
        alpha * (measurement.state_vector - eval_measurement_prediction.state_vector)
    eval_posterior_velocity = prediction.state_vector[[1, 3]] + \
        beta/timediff.total_seconds() * (measurement.state_vector -
                                         eval_measurement_prediction.state_vector)

    eval_state_vect = np.concatenate((eval_posterior_position, eval_posterior_velocity))
    eval_posterior = State(eval_state_vect[[0, 2, 1, 3]])

    # Initialise an Alpha-Beta updater
    updater = AlphaBetaUpdater(measurement_model=measurement_model,
                               alpha=alpha,
                               beta=beta)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)

    assert(np.allclose(measurement_prediction.state_vector,
                       eval_measurement_prediction.state_vector, 0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement), time_interval=timediff)

    assert(np.allclose(posterior.state_vector, eval_posterior.state_vector, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Check that the vmap parameter can be set
    updater.vmap = np.array([1, 3])
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement), time_interval=timediff)
    assert(np.allclose(posterior.state_vector, eval_posterior.state_vector, 0, atol=1.e-14))
