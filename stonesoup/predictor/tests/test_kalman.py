# coding: utf-8
import datetime

import numpy as np

from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=0.1)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior = GaussianState(np.array([[-6.45], [0.7]]),
                          np.array([[4.1123, 0.0013],
                                    [0.0013, 0.0365]]),
                          timestamp=timestamp)

    # Calculate evaluation variables
    eval_prediction = GaussianState(
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)@prior.mean,
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)
        @prior.covar
        @cv.matrix(timestamp=new_timestamp,
                   time_interval=time_interval).T
        + cv.covar(timestamp=new_timestamp,
                   time_interval=time_interval))

    # Initialise a kalman predictor
    kp = KalmanPredictor(transition_model=cv)

    # Perform and assert state prediction
    prediction = kp.predict(prior=prior,
                            timestamp=new_timestamp)
    assert(np.array_equal(prediction.mean, eval_prediction.mean))
    assert(np.array_equal(prediction.covar, eval_prediction.covar))
    assert(prediction.timestamp == new_timestamp)

    # TODO: Test with Control Model


def test_extendedkalman():

    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=0.1)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior = GaussianState(np.array([[-6.45], [0.7]]),
                          np.array([[4.1123, 0.0013],
                                    [0.0013, 0.0365]]),
                          timestamp=timestamp)

    # Calculate evaluation variables
    eval_prediction = GaussianState(
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)@prior.mean,
        cv.matrix(timestamp=new_timestamp,
                  time_interval=time_interval)
        @prior.covar
        @cv.matrix(timestamp=new_timestamp,
                   time_interval=time_interval).T
        + cv.covar(timestamp=new_timestamp,
                   time_interval=time_interval))

    # Initialise a kalman predictor
    kp = ExtendedKalmanPredictor(transition_model=cv)

    # Perform and assert state prediction
    prediction = kp.predict(prior=prior,
                            timestamp=new_timestamp)
    assert(np.array_equal(prediction.mean, eval_prediction.mean))
    assert(np.array_equal(prediction.covar, eval_prediction.covar))
    assert(prediction.timestamp == new_timestamp)

    # TODO: Test with Control Model
