# -*- coding: utf-8 -*-
"""Test for updater.information module"""
import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    InformationStatePrediction, InformationState)
from stonesoup.updater.information import InfoFilterUpdater
from numpy.linalg import inv


@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement",
    [
        (   # Standard Information filter
            InfoFilterUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            InformationStatePrediction(np.array([[-6.45], [0.7]]),
                                       np.array([[1.0000, 0.0000],
                                                 [0.0000, 1.0000]])),
            Detection(np.array([[-6.23]]))
        ),
    ],
    ids=["standard"]
)
def test_information(UpdaterClass, measurement_model, prediction, measurement):

    # Calculate evaluation variables - essentially takes the covariance and mean values and
    # projects then into state space.
    # eval_measurement_prediction = InformationMeasurementPrediction(
    #     measurement_model.matrix()@prediction.mean,
    #     measurement_model.matrix()@prediction.info_matrix
    #     @measurement_model.matrix().T
    #     + measurement_model.covar(),
    #     cross_covar=prediction.info_matrix@measurement_model.matrix().T)

    # kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
    #    eval_measurement_prediction.covar)

    print(measurement_model.matrix())
    print(measurement_model.noise_covar)

    # y = prediction.state_vector
    # H = measurement_model.matrix()
    # R = measurement_model.noise_covar
    # x = measurement.state_vector

    # Code directly below works out what the updater would implement
    eval_posterior = InformationState(
        (
                prediction.state_vector +
                (measurement_model.matrix().T @ inv(measurement_model.noise_covar) @
                 measurement.state_vector)),
        (prediction.info_matrix +
         measurement_model.matrix().T @ inv(measurement_model.noise_covar) @
         measurement_model.matrix()))

    # Initialise a information (?) (kalman) updater
    updater = UpdaterClass(measurement_model=measurement_model)

    # Get and assert measurement prediction - need to rewrite thios bit
    # measurement_prediction = updater.predict_measurement(prediction)
    # assert(np.allclose(measurement_prediction.mean,
    #                    eval_measurement_prediction.mean,
    #                    0, atol=1.e-14))
    # assert(np.allclose(measurement_prediction.covar,
    #                    eval_measurement_prediction.covar,
    #                    0, atol=1.e-14))
    # assert(np.allclose(measurement_prediction.cross_covar,
    #                    eval_measurement_prediction.cross_covar,
    #                    0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))

    # Check to see if the information matrix is positive definite (i.e. are all the eigenvalues
    # positive?)
    assert(np.all(np.linalg.eigvals(posterior.info_matrix) >= 0))

    assert(np.allclose(posterior.state_vector, eval_posterior.state_vector, 0, atol=1.e-14))
    assert(np.allclose(posterior.info_matrix, eval_posterior.info_matrix, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    # assert(np.allclose(
    #     posterior.hypothesis.measurement_prediction.state_vector,
    #     measurement_prediction.state_vector, 0, atol=1.e-14))
    # assert (np.allclose(posterior.hypothesis.measurement_prediction.info_matrix,
    #     #                     measurement_prediction.info_matrix, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # # Perform and assert state update
    # posterior = updater.update(SingleHypothesis(
    #     prediction=prediction,
    #     measurement=measurement,
    #     measurement_prediction=measurement_prediction))
    # assert(np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    # assert(np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    # assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    # assert (np.allclose(
    #     posterior.hypothesis.measurement_prediction.state_vector,
    #     measurement_prediction.state_vector, 0, atol=1.e-14))
    # assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
    #                     measurement_prediction.covar, 0, atol=1.e-14))
    # assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    # assert(posterior.timestamp == prediction.timestamp)
