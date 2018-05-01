# coding: utf-8

import numpy as np

from stonesoup.transitionmodel.linear import ConstantVelocity1D
from stonesoup.measurementmodel.base import LinearGaussian1D
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a transition model
    cv = ConstantVelocity1D(noise_diff_coeff=0.1, time_variant=1)

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_var=0.04)

    # Define prior state
    mean_prior = np.array([[-6.45], [0.7]])
    covar_prior = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    state_prior = GaussianState(mean_prior, covar_prior)

    # Calculate evaluation variables
    eval_state_pred = GaussianState(
        cv.eval()@state_prior.mean,
        cv.eval()@state_prior.covar@cv.eval().T+cv.covar())
    eval_meas_pred = GaussianState(
        lg.eval()@eval_state_pred.mean,
        lg.eval()@eval_state_pred.covar@lg.eval().T+lg.covar())
    eval_cross_covar = eval_state_pred.covar@lg.eval().T

    # Initialise a kalman predictor
    kp = KalmanPredictor(trans_model=cv, meas_model=lg)

    # Perform and assert state prediction
    state_pred = kp.predict_state(state_prior=state_prior)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))

    # Perform and assert measurement prediction
    meas_pred, cross_covar = kp.predict_meas(state_pred=state_pred)
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))

    # Re-initialise a kalman predictor
    kp = KalmanPredictor(trans_model=cv, meas_model=lg)

    # Perform and assert full prediction
    state_pred, meas_pred, cross_covar = kp.predict(state_prior=state_prior)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))


def test_extendedkalman():

    # Initialise a transition model
    cv = ConstantVelocity1D(noise_diff_coeff=0.1, time_variant=1)

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_var=0.04)

    # Define prior state
    mean_prior = np.array([[-6.45], [0.7]])
    covar_prior = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    state_prior = GaussianState(mean_prior, covar_prior)

    # Calculate evaluation variables
    eval_state_pred = GaussianState(
        cv.eval()@state_prior.mean,
        cv.eval()@state_prior.covar@cv.eval().T+cv.covar())
    eval_meas_pred = GaussianState(
        lg.eval()@eval_state_pred.mean,
        lg.eval()@eval_state_pred.covar@lg.eval().T+lg.covar())
    eval_cross_covar = eval_state_pred.covar@lg.eval().T

    # Initialise a kalman predictor
    kp = ExtendedKalmanPredictor(trans_model=cv, meas_model=lg)

    # Perform and assert state prediction
    state_pred = kp.predict_state(state_prior=state_prior)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))

    # Perform and assert measurement prediction
    meas_pred, cross_covar = kp.predict_meas(state_pred=state_pred)
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))

    # Re-initialise a kalman predictor
    kp = KalmanPredictor(trans_model=cv, meas_model=lg)

    # Perform and assert full prediction
    state_pred, meas_pred, cross_covar = kp.predict(state_prior=state_prior)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))
