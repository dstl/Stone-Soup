# coding: utf-8
import datetime

import numpy as np

from stonesoup.models.transition.linear import ConstantVelocity1D
from stonesoup.models.measurement.linear import LinearGaussian1D
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a transition model
    cv = ConstantVelocity1D(noise_diff_coeff=0.1)

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_covar=0.04)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp
    # Define prior state
    mean_prior = np.array([[-6.45], [0.7]])
    covar_prior = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    state_prior = GaussianState(mean_prior, covar_prior, timestamp=timestamp)

    # Calculate evaluation variables
    eval_state_pred = GaussianState(
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)@state_prior.mean,
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)
        @state_prior.covar
        @cv.matrix(timestamp=new_timestamp,
                   time_interval=time_interval).T
        + cv.covar(timestamp=new_timestamp,
                   time_interval=time_interval))
    eval_meas_pred = GaussianState(
        lg.matrix()@eval_state_pred.mean,
        lg.matrix()@eval_state_pred.covar@lg.matrix().T+lg.covar())
    eval_cross_covar = eval_state_pred.covar@lg.matrix().T

    # Initialise a kalman predictor
    kp = KalmanPredictor(transition_model=cv, measurement_model=lg)

    # Perform and assert state prediction
    state_pred = kp.predict_state(state=state_prior,
                                  timestamp=new_timestamp)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(state_pred.timestamp == new_timestamp)

    # Perform and assert measurement prediction
    meas_pred, cross_covar = kp.predict_measurement(state=state_pred)
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))
    assert(meas_pred.timestamp == new_timestamp)

    # Re-initialise a kalman predictor
    kp = KalmanPredictor(transition_model=cv, measurement_model=lg)

    # Perform and assert full prediction
    state_pred, meas_pred, cross_covar = kp.predict(state=state_prior,
                                                    timestamp=new_timestamp)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))
    assert(state_pred.timestamp == new_timestamp)
    assert(meas_pred.timestamp == new_timestamp)

    # TODO: Test with Control Model


def test_extendedkalman():

    # Initialise a transition model
    cv = ConstantVelocity1D(noise_diff_coeff=0.1)

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_covar=0.04)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    mean_prior = np.array([[-6.45], [0.7]])
    covar_prior = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    state_prior = GaussianState(mean_prior, covar_prior, timestamp=timestamp)

    # Calculate evaluation variables
    eval_state_pred = GaussianState(
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)@state_prior.mean,
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)
        @state_prior.covar
        @cv.matrix(timestamp=new_timestamp,
                   time_interval=time_interval).T
        + cv.covar(timestamp=new_timestamp,
                   time_interval=time_interval))
    eval_meas_pred = GaussianState(
        lg.matrix()@eval_state_pred.mean,
        lg.matrix()@eval_state_pred.covar@lg.matrix().T+lg.covar())
    eval_cross_covar = eval_state_pred.covar@lg.matrix().T

    # Initialise a kalman predictor
    kp = ExtendedKalmanPredictor(transition_model=cv, measurement_model=lg)

    # Perform and assert state prediction
    state_pred = kp.predict_state(state=state_prior,
                                  timestamp=new_timestamp)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(state_pred.timestamp == new_timestamp)

    # Perform and assert measurement prediction
    meas_pred, cross_covar = kp.predict_measurement(state=state_pred)
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))
    assert(meas_pred.timestamp == new_timestamp)

    # Re-initialise a kalman predictor
    kp = ExtendedKalmanPredictor(transition_model=cv, measurement_model=lg)

    # Perform and assert full prediction
    state_pred, meas_pred, cross_covar = kp.predict(state=state_prior,
                                                    timestamp=new_timestamp)
    assert(np.array_equal(state_pred.mean, eval_state_pred.mean))
    assert(np.array_equal(state_pred.covar, eval_state_pred.covar))
    assert(np.array_equal(meas_pred.mean, eval_meas_pred.mean))
    assert(np.array_equal(meas_pred.covar, eval_meas_pred.covar))
    assert(np.array_equal(cross_covar, eval_cross_covar))
    assert(state_pred.timestamp == new_timestamp)
    assert(meas_pred.timestamp == new_timestamp)

    # TODO: Test with Control Model
