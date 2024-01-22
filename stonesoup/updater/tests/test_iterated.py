"""Test for updater.iterated module"""
import numpy as np
import datetime

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.smoother.kalman import ExtendedKalmanSmoother
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.state import GaussianState
from stonesoup.updater.iterated import DynamicallyIteratedUpdater
from stonesoup.updater.kalman import ExtendedKalmanUpdater


def test_diekf():

    time1 = datetime.datetime.now()
    time2 = time1 + datetime.timedelta(seconds=1)

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])

    sensor_x = 50  # Placing the sensor off-centre
    sensor_y = 0

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
        # bearing and 1 metre in range
        translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
        # sensor in cartesian.
    )

    prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=time1)

    prediction = GaussianStatePrediction(state_vector=StateVector([[0.], [1.], [0.], [1.]]),
                                         covar=CovarianceMatrix([[1.5, 0., 0., 0.],
                                                                 [0., 0.5, 0., 0.],
                                                                 [0., 0., 1.5, 0.],
                                                                 [0., 0., 0., 0.5]]),
                                         transition_model=transition_model,
                                         timestamp=time2,
                                         prior=prior)

    measurement = Detection(state_vector=StateVector([[Bearing(-3.1217136817127424)],
                                                      [47.7876225398533]]),
                            timestamp=time2,
                            measurement_model=measurement_model)

    hypothesis = SingleHypothesis(prediction=prediction, measurement=measurement)

    sub_updater = ExtendedKalmanUpdater(measurement_model)
    sub_predictor = ExtendedKalmanPredictor(transition_model)
    smoother = ExtendedKalmanSmoother(transition_model)

    updater = DynamicallyIteratedUpdater(predictor=sub_predictor,
                                         updater=sub_updater,
                                         smoother=smoother)

    updated_state = updater.update(hypothesis=hypothesis)

    # Check timestamp is same as provided in measurement
    assert updated_state.timestamp == time2

    # Check prediction and measurement are unchanged
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement

    # Check state vector is correct
    assert np.allclose(updated_state.state_vector, StateVector([[1.81240407],
                                                                [1.21149362],
                                                                [0.60307421],
                                                                [0.89666808]]))

    # Check covariance matrix is correct
    assert np.allclose(updated_state.covar, CovarianceMatrix([[6.68656230e-01, 1.74071663e-01,
                                                               1.18372990e-02, 3.08161090e-03],
                                                              [1.74071663e-01, 4.58642623e-01,
                                                               3.08161090e-03, 8.02237548e-04],
                                                              [1.18372990e-02, 3.08161090e-03,
                                                               1.61478471e+00, 4.20377837e-01],
                                                              [3.08161090e-03, 8.02237548e-04,
                                                               4.20377837e-01, 5.22763652e-01]]),
                       atol=1.e-4)
