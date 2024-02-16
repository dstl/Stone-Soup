"""Test for updater.iterated module"""
import numpy as np
import datetime

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.nonlinear import ConstantTurn
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
    time3 = time2 + datetime.timedelta(seconds=1)

    transition_model = ConstantTurn([0.05, 0.05], np.radians(2))

    sensor_x = 50  # Placing the sensor off-centre
    sensor_y = 0

    measurement_model = CartesianToBearingRange(
        ndim_state=5,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
        # bearing and 1 metre in range
        translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
        # sensor in cartesian.
    )

    prior = GaussianState(
        [[0.], [1.], [0.], [1.], [0.]],
        np.diag([1.5, 0.5, 1.5, 0.5, np.radians(0.5)]),
        timestamp=time1)

    prediction = GaussianStatePrediction(
        state_vector=StateVector([[1.], [1.], [1.], [1.], [0.]]),
        covar=CovarianceMatrix([[1.5, 0., 0., 0., 0.],
                                [0., 0.5, 0., 0., 0.],
                                [0., 0., 1.5, 0., 0.],
                                [0., 0., 0., 0.5, 0.],
                                [0., 0., 0., 0., np.radians(0.5)]]),
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
    assert np.allclose(
        updated_state.state_vector,
        StateVector([[1.810], [1.203], [0.605], [0.901], [0.]]),
        atol=1e-3)

    # Check covariance matrix is correct
    assert np.allclose(
        updated_state.covar,
        CovarianceMatrix(
            [[0.666, 0.167, 0.011, 0.002, 0.],
             [0.167, 0.427, 0.002, -0.007, -0.008],
             [0.011, 0.002, 1.604,  0.401, 0.],
             [0.002, -0.007, 0.401, 0.486, 0.008],
             [0., -0.008, 0., 0.008, 0.009]]),
        atol=1.e-3)

    prediction = sub_predictor.predict(updated_state, time3)
    measurement = Detection(state_vector=StateVector([[Bearing(3.133)], [47.777]]),
                            timestamp=time3,
                            measurement_model=measurement_model)
    hypothesis = SingleHypothesis(prediction=prediction, measurement=measurement)
    updated_state = updater.update(hypothesis=hypothesis)

    assert np.allclose(
        updated_state.state_vector,
        StateVector([[2.555], [1.010], [1.226], [0.820], [0.002]]),
        atol=1e-3)
    assert np.allclose(
        updated_state.covar,
        CovarianceMatrix(
            [[5.904e-01,  2.501e-01, 2.800e-02, 3.878e-04, -5.127e-03],
             [2.501e-01,  2.984e-01, -4.381e-03, -2.161e-02, -1.362e-02],
             [2.800e-02, -4.381e-03, 2.121e+00, 6.620e-01, 9.775e-03],
             [3.878e-04, -2.161e-02, 6.620e-01, 4.395e-01, 1.740e-02],
             [-5.127e-03, -1.362e-02, 9.775e-03, 1.740e-02, 1.107e-02]]),
        atol=1e-3)
