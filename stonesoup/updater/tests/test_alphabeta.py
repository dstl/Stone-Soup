# -*- coding: utf-8 -*-
"""Test for updater.alphabeta module"""
import pytest
import numpy as np
from stonesoup.models.measurement.linear import LinearMeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.state import State
from stonesoup.types.prediction import (
    StatePrediction, StateMeasurementPrediction)
from stonesoup.updater.alphabeta import AlphaBetaUpdater
from stonesoup.types.hypothesis import SingleHypothesis


@pytest.mark.parametrize(
    "measurement_model, prediction, measurement, alpha, beta",
    [
        (   # Standard Alpha-Beta
            LinearMeasurementModel(ndim_state=4, mapping=[0, 2]),
            StatePrediction(np.array([[-6.45], [0.7], [-6.45], [0.7]]
                                     )),
            Detection(np.array([[-6.23],
                                [-6.23]])),
            0.9,
            0.3
        )
    ],
    ids=["standard"]
)
def test_alphabeta(measurement_model, prediction, measurement, alpha, beta):

    # Calculate evaluation variables - converts
    # to measurement from prediction space
    eval_measurement_prediction = StateMeasurementPrediction(
       measurement_model.matrix()@prediction.state_vector)

    eval_posterior_mean = alpha * (
            measurement.state_vector - eval_measurement_prediction.state_vector)
    eval_posterior_velocity = beta * (
            measurement.state_vector - eval_measurement_prediction.state_vector)

    eval_posterior = State(state_vector=np.zeros(
        prediction.state_vector.shape))
    eval_posterior.state_vector[0] = prediction.state_vector[0] + (
        eval_posterior_mean[0])
    eval_posterior.state_vector[1] = prediction.state_vector[1] + (
        eval_posterior_velocity[0])
    eval_posterior.state_vector[2] = prediction.state_vector[2] + (
        eval_posterior_mean[1])
    eval_posterior.state_vector[3] = prediction.state_vector[3] + (
        eval_posterior_velocity[1])

    # Initialise an Alpha-Beta updater
    updater = AlphaBetaUpdater(measurement_model=measurement_model,
                               alpha=alpha,
                               beta=beta)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)

    assert(np.allclose(measurement_prediction.state_vector,
                       eval_measurement_prediction.state_vector,
                       0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    print(posterior.state_vector)
    assert(np.allclose(posterior.state_vector, eval_posterior.state_vector,
                       0,
                       atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert(np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.allclose(posterior.state_vector, eval_posterior.state_vector, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14)
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)
