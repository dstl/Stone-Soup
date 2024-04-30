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
        StateVector([[1.812], [1.211], [0.603], [0.897], [0.]]),
        atol=1e-3)

    # Check covariance matrix is correct
    assert np.allclose(
        updated_state.covar,
        CovarianceMatrix(
            [[0.669, 0.174, 0.012, 0.003, 0.],
             [0.174, 0.467, 0.003, -0.008, -0.009],
             [0.012, 0.003, 1.615,  0.420, 0.],
             [0.003, -0.008, 0.420, 0.531, 0.009],
             [0., -0.009, 0., 0.009, 0.044]]),
        atol=1.e-3)

    prediction = sub_predictor.predict(updated_state, time3)
    measurement = Detection(state_vector=StateVector([[Bearing(3.133)], [47.777]]),
                            timestamp=time3,
                            measurement_model=measurement_model)
    hypothesis = SingleHypothesis(prediction=prediction, measurement=measurement)
    updated_state = updater.update(hypothesis=hypothesis)

    assert np.allclose(
        updated_state.state_vector,
        StateVector([[2.550], [0.999], [1.215], [0.811], [0.005]]),
        atol=1e-3)
    assert np.allclose(
        updated_state.covar,
        CovarianceMatrix(
            [[0.603, 0.275, 0.027, -0.003, -0.010],
             [0.275, 0.370, -0.013, -0.043, -0.037],
             [0.027, -0.013, 2.182, 0.730, 0.022],
             [-0.003, -0.043, 0.730, 0.548, 0.049],
             [-0.010, -0.037, 0.022, 0.049, 0.078]]),
        atol=1e-3)
