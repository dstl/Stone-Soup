# coding: utf-8

import numpy as np

from stonesoup.transitionmodel.base import ConstantVelocity1D
from stonesoup.measurementmodel.base import LinearGaussian1D
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.base import GaussianState

import logging

# create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(classname)s:'
    '%(filename)s:%(lineno)d: %(message)s', "%H:%M:%S")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

logger = logging.LoggerAdapter(
    logger, {'classname': '__name__'})


def test_kf():

    # Initialise a transition model
    cv = ConstantVelocity1D(noise_diff_coeff=0.1, time_variant=1)

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_var=0.04)

    # Define test variables
    mean_prior = np.array([[-6.45], [0.7]])
    covar_prior = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    state_prior = GaussianState(mean_prior, covar_prior)
    state_pred = GaussianState(
        cv.eval()@state_prior.mean,
        cv.eval()@state_prior.covar@cv.eval().T+cv.covar())
    meas_pred = GaussianState(
        lg.eval()@state_pred.mean,
        lg.eval()@state_pred.covar@lg.eval().T+lg.covar())

    # Initialise a kalman predictor
    kp = KalmanPredictor(
        trans_model=cv, state_prior=state_prior, meas_model=lg)

    # Perform and assert state prediction
    kp.predict_state()
    assert(np.array_equal(kp.state_pred.mean, state_pred.mean))
    assert(np.array_equal(kp.state_pred.covar, state_pred.covar))

    # Perform and assert measurement prediction
    kp.predict_meas()
    assert(np.array_equal(kp.meas_pred.mean, meas_pred.mean))
    assert(np.array_equal(kp.meas_pred.covar, meas_pred.covar))

    # Re-initialise a kalman predictor
    kp = KalmanPredictor(
        trans_model=cv, state_prior=state_prior, meas_model=lg)

    # Perform and assert full prediction
    kp.predict()
    assert(np.array_equal(kp.state_pred.mean, state_pred.mean))
    assert(np.array_equal(kp.state_pred.covar, state_pred.covar))
    assert(np.array_equal(kp.meas_pred.mean, meas_pred.mean))
    assert(np.array_equal(kp.meas_pred.covar, meas_pred.covar))
