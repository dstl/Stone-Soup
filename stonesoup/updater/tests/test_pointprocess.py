"""Test for updater.gaussianmixture module"""
import pytest
import numpy as np
from scipy.stats import multivariate_normal


from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.prediction import GaussianMeasurementPrediction
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import (
    KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater)
from stonesoup.updater.pointprocess import PHDUpdater, LCCUpdater


@pytest.mark.parametrize(
    "UpdaterClass",
    [
        # Standard Kalman
        KalmanUpdater,
        # Extended Kalman
        ExtendedKalmanUpdater,
        # Unscented Kalman
        UnscentedKalmanUpdater,
    ],
    ids=["standard", "extended", "unscented"]
    )
def test_phd_init(UpdaterClass, measurement_model, prediction, measurement):
    # Initialise a kalman updater
    underlying_updater = UpdaterClass(measurement_model=measurement_model)
    phd_updater = PHDUpdater(updater=underlying_updater)
    assert isinstance(phd_updater.updater, UpdaterClass)


@pytest.mark.parametrize(
    "UpdaterClass",
    [
        # Standard Kalman
        KalmanUpdater,
        # Extended Kalman
        ExtendedKalmanUpdater,
        # Unscented Kalman
        UnscentedKalmanUpdater,
    ],
    ids=["standard", "extended", "unscented"]
    )
def test_phd_single_component_update(UpdaterClass, measurement_model,
                                     prediction, measurement):
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix()@prediction.mean,
        measurement_model.matrix()@prediction.covar@measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar@measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    underlying_updater = UpdaterClass(measurement_model=measurement_model)
    measurement_prediction = underlying_updater.predict_measurement(prediction)

    phd_updater = PHDUpdater(updater=underlying_updater, prob_detection=0.9)
    hypotheses = [MultipleHypothesis([SingleHypothesis(
                                    prediction=prediction,
                                    measurement=measurement)]),
                  MultipleHypothesis([SingleHypothesis(
                                    prediction=prediction,
                                    measurement=None)])]

    updated_mixture = phd_updater.update(hypotheses)
    # One for updated component, one for missed detection
    assert len(updated_mixture) == 2
    # Check updated component
    updated_component = updated_mixture[0]
    assert(np.allclose(updated_component.mean, eval_posterior.mean, 0,
                       atol=1.e-14))
    assert(np.allclose(updated_component.covar, eval_posterior.covar, 0,
                       atol=1.e-14))
    assert(updated_component.timestamp == measurement.timestamp)
    prob_detection = 0.9
    prob_survival = 1
    q = multivariate_normal.pdf(
                    measurement.state_vector.flatten(),
                    mean=measurement_prediction.mean.flatten(),
                    cov=measurement_prediction.covar
                )
    clutter_density = 1e-26
    new_weight = (prob_detection*prediction.weight*q*prob_survival) / \
        ((prob_detection*prediction.weight*q*prob_survival)+clutter_density)
    assert(updated_component.weight == new_weight)
    # Check miss detected component
    miss_detected_component = updated_mixture[1]
    assert(np.allclose(miss_detected_component.mean, prediction.mean, 0,
                       atol=1.e-14))
    assert(np.allclose(miss_detected_component.covar, prediction.covar, 0,
                       atol=1.e-14))
    assert(miss_detected_component.timestamp == prediction.timestamp)
    l1 = 1
    assert(miss_detected_component.weight == prediction.weight *
           (1-prob_detection)*l1)


@pytest.mark.parametrize(
    "UpdaterClass",
    [
        # Standard Kalman
        KalmanUpdater,
        # Extended Kalman
        ExtendedKalmanUpdater,
        # Unscented Kalman
        UnscentedKalmanUpdater,
    ],
    ids=["standard", "extended", "unscented"]
    )
def test_lcc_single_component_update(UpdaterClass, measurement_model,
                                     prediction, measurement):
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix()@prediction.mean,
        measurement_model.matrix()@prediction.covar@measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar@measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    underlying_updater = UpdaterClass(measurement_model=measurement_model)
    measurement_prediction = underlying_updater.predict_measurement(prediction)

    phd_updater = LCCUpdater(updater=underlying_updater, prob_detection=0.9)
    hypotheses = [MultipleHypothesis([SingleHypothesis(
                                    prediction=prediction,
                                    measurement=measurement)]),
                  MultipleHypothesis([SingleHypothesis(
                                    prediction=prediction,
                                    measurement=None)])]

    updated_mixture = phd_updater.update(hypotheses)
    # One for updated component, one for missed detection
    assert len(updated_mixture) == 2
    # Check updated component
    updated_component = updated_mixture[0]
    assert(np.allclose(updated_component.mean, eval_posterior.mean, 0,
                       atol=1.e-14))
    assert(np.allclose(updated_component.covar, eval_posterior.covar, 0,
                       atol=1.e-14))
    assert(updated_component.timestamp == measurement.timestamp)
    prob_detection = 0.9
    prob_survival = 1
    q = multivariate_normal.pdf(
                    measurement.state_vector.flatten(),
                    mean=measurement_prediction.mean.flatten(),
                    cov=measurement_prediction.covar
                )
    clutter_density = 1e-26
    new_weight = (prob_detection*prediction.weight*q*prob_survival) / \
        ((prob_detection*prediction.weight*q*prob_survival)+clutter_density)
    assert(updated_component.weight == new_weight)
    # Check miss detected component
    miss_detected_component = updated_mixture[1]
    assert(np.allclose(miss_detected_component.mean, prediction.mean, 0,
                       atol=1.e-14))
    assert(np.allclose(miss_detected_component.covar, prediction.covar, 0,
                       atol=1.e-14))
    assert(miss_detected_component.timestamp == prediction.timestamp)
    l1 = 1
    assert(miss_detected_component.weight ==
           prediction.weight*(1-prob_detection)*l1)
