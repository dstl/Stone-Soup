# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import numpy as np

from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.numeric import Probability
from stonesoup.updater.kalman import KalmanUpdater, PDAKalmanUpdater,\
    ExtendedKalmanUpdater
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types import GaussianState, GaussianStatePrediction,\
    GaussianMeasurementPrediction, SingleMeasurementHypothesis
from stonesoup.types.multimeasurementhypothesis import \
    ProbabilityMultipleMeasurementHypothesis
from stonesoup.types.array import CovarianceMatrix


def test_kalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))

    # Define predicted state
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[4.1123, 0.0013],
                                                   [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar(),
        cross_covar=prediction.covar@lg.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = KalmanUpdater(measurement_model=lg)

    # Get and assert measurement prediction
    measurement_prediction = updater.get_measurement_prediction(prediction)
    assert(np.array_equal(measurement_prediction.mean,
                          eval_measurement_prediction.mean))
    assert(np.array_equal(measurement_prediction.covar,
                          eval_measurement_prediction.covar))
    assert(np.array_equal(measurement_prediction.cross_covar,
                          eval_measurement_prediction.cross_covar))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleMeasurementHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleMeasurementHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)


def test_pdakalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=4, mapping=[0, 2],
                        noise_covar=np.diag([0.04, 0.02]))

    # Define predicted state
    prediction = \
        GaussianStatePrediction(np.array([[-6.45], [0.7], [14.5], [-0.4]]),
                                np.array([[4.1123, 0.0013, 0.0450, 0.0022],
                                          [0.0013, 0.0365, 0.0001, 0.0023],
                                          [0.0823, 0.0014, 3.2341, 0.0109],
                                          [0.0053, 0.0027, 0.0326, 0.4561]]))

    # define Detections and their prob of association with the target/track
    measurements = [MissedDetection(),
                    Detection(np.array([[-6.23], [14.23]])),
                    Detection(np.array([[-6.54], [14.53]])),
                    Detection(np.array([[-6.18], [14.98]]))]
    probabilities = [Probability(0.04, log_value=False),
                     Probability(0.31, log_value=False),
                     Probability(0.35, log_value=False),
                     Probability(0.29, log_value=False)]

    # Calculate evaluation variables - the 'ground truth' against which
    # the ouput of the 'PDAKalmanUpdater' will be measured
    eval_measurement_prediction = GaussianMeasurementPrediction(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar(),
        cross_covar=prediction.covar@lg.matrix().T)

    # Initialise a kalman updater
    updater = PDAKalmanUpdater(measurement_model=lg)

    # Get and assert measurement prediction
    measurement_prediction = updater.get_measurement_prediction(prediction)
    assert(np.array_equal(measurement_prediction.mean,
                          eval_measurement_prediction.mean))
    assert(np.array_equal(measurement_prediction.covar,
                          eval_measurement_prediction.covar))
    assert(np.array_equal(measurement_prediction.cross_covar,
                          eval_measurement_prediction.cross_covar))

    # Perform and assert state update
    multihypothesis = ProbabilityMultipleMeasurementHypothesis(
        prediction, measurement_prediction)
    multihypothesis.add_weighted_detections(measurements, probabilities)
    posterior = updater.update(multihypothesis)

    eval_posterior_mean = [[-6.3361], [0.7000], [14.5656], [-0.3992]]
    eval_posterior_covar = CovarianceMatrix(
                           [[0.2278,   7.32e-05,   0.0121, -2.36e-03],
                            [7.23e-05,   0.0365, 9.98e-06,  2.30e-03],
                            [0.0141,   1.30e-03,   0.2327,   -0.0194],
                            [3.90e-04, 2.70e-03, 2.36e-03,    0.4558]])

    assert(np.allclose(posterior.mean, eval_posterior_mean, rtol=5e-2))
    assert(np.allclose(posterior.covar, eval_posterior_covar, rtol=5e-2))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.covar,
        measurement_prediction.covar))
    for index, weighted_measurement in \
            enumerate(posterior.hypothesis.weighted_measurements):
        assert weighted_measurement["measurement"] is measurements[index]
        assert weighted_measurement["weight"] is probabilities[index]

    # Perform and assert state update (without measurement prediction)
    multihypothesis = ProbabilityMultipleMeasurementHypothesis(
        prediction, None)
    multihypothesis.add_weighted_detections(measurements, probabilities)
    posterior = updater.update(multihypothesis)

    assert(np.allclose(posterior.mean, eval_posterior_mean, rtol=5e-2))
    assert(np.allclose(posterior.covar, eval_posterior_covar, rtol=5e-2))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.covar,
        measurement_prediction.covar))
    for index, weighted_measurement in \
            enumerate(posterior.hypothesis.weighted_measurements):
        assert weighted_measurement["measurement"] is measurements[index]
        assert weighted_measurement["weight"] is probabilities[index]


def test_extendedkalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))

    # Define predicted state
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[4.1123, 0.0013],
                                                   [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar(),
        cross_covar=prediction.covar@lg.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = ExtendedKalmanUpdater(measurement_model=lg)

    # Get and asser measurement prediction
    measurement_prediction = updater.get_measurement_prediction(prediction)
    assert(np.array_equal(measurement_prediction.mean,
                          eval_measurement_prediction.mean))
    assert(np.array_equal(measurement_prediction.covar,
                          eval_measurement_prediction.covar))
    assert(np.array_equal(measurement_prediction.cross_covar,
                          eval_measurement_prediction.cross_covar))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleMeasurementHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleMeasurementHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.array_equal(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector))
    assert (np.array_equal(posterior.hypothesis.measurement_prediction.covar,
                           measurement_prediction.covar))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)
