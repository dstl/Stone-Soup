# -*- coding: utf-8 -*-
import numpy as np
import pytest
from pytest import approx
from scipy.stats import multivariate_normal

from ..nonlinear import (
    CartesianToElevationBearingRange, CartesianToBearingRange,
    CartesianToElevationBearing, Cartesian2DToBearing, CartesianToBearingRangeRate,
    CartesianToElevationBearingRangeRate, RangeRangeRateBinning)

from ...base import ReversibleModel
from ....functions import jacobian as compute_jac
from ....functions import pol2cart
from ....functions import rotz, rotx, roty, cart2sphere
from ....types.angle import Bearing, Elevation
from ....types.array import StateVector, StateVectors
from ....types.state import State, CovarianceMatrix


def h1d(state_vector, pos_map, translation_offset, rotation_offset):
    xyz = StateVector([[state_vector[0, 0] - translation_offset[0, 0]],
                      [state_vector[1, 0] - translation_offset[1, 0]],
                      [0]])

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    _, phi, _ = cart2sphere(*xyz_rot)

    return StateVector([Bearing(phi)])


def h2d(state_vector, pos_map, translation_offset, rotation_offset):

    xyz = StateVector([[state_vector[0, 0] - translation_offset[0, 0]],
                      [state_vector[1, 0] - translation_offset[1, 0]],
                      [0]])

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, _ = cart2sphere(*xyz_rot)

    return StateVector([Bearing(phi), rho])


def h3d(state_vector, pos_map,  translation_offset, rotation_offset):
    xyz = state_vector[pos_map, :] - translation_offset

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]
    theta_y = - theta_y

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, theta = cart2sphere(*xyz_rot)

    return StateVector([Elevation(theta), Bearing(phi), rho])


def hbearing(state_vector, pos_map, translation_offset, rotation_offset):
    xyz = state_vector[pos_map, :] - translation_offset

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    _, phi, theta = cart2sphere(*xyz_rot)

    return StateVector([Elevation(theta), Bearing(phi)])


@pytest.mark.parametrize(
    "h, ModelClass, state_vec, R , mapping,\
     translation_offset, rotation_offset",
    [
        (   # 1D meas, 2D state
            h1d,
            Cartesian2DToBearing,
            StateVector([[0], [1]]),
            CovarianceMatrix([[0.015]]),
            np.array([0, 1]),
            StateVector([[1], [-1]]),
            StateVector([[0], [0], [1]])

        ),
        (   # 1D meas, 2D state
            h1d,
            Cartesian2DToBearing,
            StateVector([[0], [1]]),
            CovarianceMatrix([[0.015]]),
            np.array([0, 1]),
            None,
            None

        ),
        (   # 2D meas, 2D state
            h2d,
            CartesianToBearingRange,
            StateVector([[0], [1]]),
            CovarianceMatrix([[0.015, 0],
                              [0, 0.1]]),
            np.array([0, 1]),
            StateVector([[1], [-1]]),
            StateVector([[0], [0], [1]])

        ),
        (   # 2D meas, 2D state
            h2d,
            CartesianToBearingRange,
            StateVector([[0], [1]]),
            CovarianceMatrix([[0.015, 0],
                              [0, 0.1]]),
            np.array([0, 1]),
            None,
            None

        ),
        (   # 3D meas, 3D state
            h3d,
            CartesianToElevationBearingRange,
            StateVector([[1], [2], [2]]),
            CovarianceMatrix([[0.05, 0, 0],
                              [0, 0.015, 0],
                              [0, 0, 0.1]]),
            np.array([0, 1, 2]),
            StateVector([[0], [0], [0]]),
            StateVector([[.2], [3], [-1]])
        ),
        (   # 3D meas, 3D state
            h3d,
            CartesianToElevationBearingRange,
            StateVector([[1], [2], [2]]),
            CovarianceMatrix([[0.05, 0, 0],
                              [0, 0.015, 0],
                              [0, 0, 0.1]]),
            np.array([0, 1, 2]),
            None,
            None
        ),
        (   # 2D meas, 3D state
            hbearing,
            CartesianToElevationBearing,
            StateVector([[1], [2], [3]]),
            np.array([[0.05, 0],
                      [0, 0.015]]),
            np.array([0, 1, 2]),
            StateVector([[0], [0], [0]]),
            StateVector([[-3], [0], [np.pi/3]])
        ),
        (   # 2D meas, 3D state
            hbearing,
            CartesianToElevationBearing,
            StateVector([[1], [2], [3]]),
            np.array([[0.05, 0],
                      [0, 0.015]]),
            np.array([0, 1, 2]),
            None,
            None
        )
    ],
    ids=["Bearing1", "Bearing2",
         "BearingElevation1", "BearingElevation2",
         "RangeBearingElevation1", "RangeBearingElevation1",
         "BearingsOnly1", "BearingsOnly2"]
)
def test_models(h, ModelClass, state_vec, R,
                mapping, translation_offset, rotation_offset):
    """ Test for the CartesianToBearingRange, CartesianToElevationBearingRange,
     and CartesianToElevationBearing Measurement Models """

    ndim_state = state_vec.size
    state = State(state_vec)

    # Check default translation_offset, rotation_offset and velocity is applied
    model_test = ModelClass(ndim_state=ndim_state,
                            mapping=mapping,
                            noise_covar=R)

    assert len(model_test.translation_offset) == ndim_state
    assert len(model_test.rotation_offset) == 3

    # Create and a measurement model object
    model = ModelClass(ndim_state=ndim_state,
                       mapping=mapping,
                       noise_covar=R,
                       translation_offset=translation_offset,
                       rotation_offset=rotation_offset)

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state)
    eval_m = h(state_vec, mapping, model.translation_offset, model.rotation_offset)
    assert np.array_equal(meas_pred_wo_noise, eval_m)

    # Ensure ```lg.transfer_function()``` returns H
    def fun(x):
        return model.function(x)
    H = compute_jac(fun, state)
    assert np.array_equal(H, model.jacobian(state))

    # Check Jacobian has proper dimensions
    assert H.shape == (model.ndim_meas, ndim_state)

    # Ensure inverse function returns original
    if isinstance(model, ReversibleModel):
        J = model.inverse_function(State(meas_pred_wo_noise))
        assert np.allclose(J, state_vec)

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, model.covar())

    # Ensure model creates noise
    rvs = model.rvs()
    assert rvs.shape == (model.ndim_meas, 1)
    assert isinstance(rvs, StateVector)
    rvs = model.rvs(10)
    assert rvs.shape == (model.ndim_meas, 10)
    assert isinstance(rvs, StateVectors)
    assert not isinstance(rvs, StateVector)

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state)
    assert np.array_equal(meas_pred_wo_noise,  h(
        state_vec, mapping, model.translation_offset, model.rotation_offset))

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = model.pdf(State(meas_pred_wo_noise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_wo_noise
         - np.array(h(state_vec, mapping, model.translation_offset, model.rotation_offset))
         ).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = model.function(state, noise=True)
    assert not np.array_equal(
        meas_pred_w_inoise,  h(state_vec,
                               mapping,
                               model.translation_offset,
                               model.rotation_offset))

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_inoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_w_inoise
         - np.array(h(state_vec, mapping, model.translation_offset, model.rotation_offset))
         ).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with external noise)
    noise = model.rvs()
    meas_pred_w_enoise = model.function(state,
                                        noise=noise)
    assert np.array_equal(meas_pred_w_enoise,  h(
        state_vec, mapping, model.translation_offset, model.rotation_offset)+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_enoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_w_enoise
         - h(state_vec, model.mapping, model.translation_offset, model.rotation_offset)
         ).ravel(),
        cov=R)


position_measurement_sets = [((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0),
                              (0, 0, 1, -1)),
                             ((0, 0, -50, 0.25, 0, 0), (0, 0, 130, -0.25, 0, 0),
                              (0, 0, 180, -0.5)),
                             ((0, 0, 0, 1, 0, 0), (10, 0, 10, 0, 0, 0),
                              (0, -np.pi/4, np.sqrt(200), -1/np.sqrt(2))),
                             ((0, 1, 0, 0, 0, 0), (10, 0, 10, 0, 0, 0),
                              (0, np.pi / 4, np.sqrt(200), -1/np.sqrt(2))),
                             ((0, 1, 0, 1, 0, 0), (10, 0, 10, 0, 0, 0),
                              (0, 0, np.sqrt(200), -np.sqrt(2))),
                             ((0, 1, 0, 0, 0, 0), (10, 0, 0, 0, 10, 0),
                              (np.pi/4, 0, np.sqrt(200), -1/np.sqrt(2))),
                             ((1, 1, 0, 0, 1, 0), (10, 0, 0, 0, 10, 0),
                              (np.pi/4, 0, np.sqrt(81*2), -1/np.sqrt(2))),
                             ((-1, 1, 0, 0, -1, 0), (10, 0, 0, 0, 10, 0),
                              (np.pi/4, 0, np.sqrt(121*2), -1/np.sqrt(2))),
                             ((0, 1, 0, 0, 0, 0.5), (10, 0, 0, 0, 5, 0),
                              (0, 0, np.sqrt(125), -np.sqrt(1.25))),
                             ((0, 1, 0, 0, 0, 0), (10, 0, 0, 0, 5, 0),
                              (np.arctan(0.5), 0, np.sqrt(125), - np.cos(np.arctan(5/10)))),
                             ((0, 1, 0, 0, 0, 1.2), (10, 0, 0, 0, 12, 0),
                              (0, 0, np.sqrt(244), -np.sqrt(2.44))),
                             ((0, 1, 0, 0, 0, 0), (10, 0, 0, 0, 12, 0),
                              (np.arctan(1.2), 0, np.sqrt(244), -np.cos(np.arctan(12/10)))),
                             ((0, 1, 0, 0, 0, 1), (10, 0, 0, 0, 10, 0),
                              (0, 0, np.sqrt(200), -np.sqrt(2))),
                             ((0, 1, 0, 0, 0, 1), (10, 0, 0, 0, 0, 0),
                              (-np.pi/4, 0, 10, -1)),
                             ((0, 1, 0, 0, 0, 0), (10, 0, 0, 0, -10, 0),
                              (-np.pi / 4, 0, np.sqrt(200), -1 / np.sqrt(2))),
                             ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 10, 0),
                              (0, 0, 10, -1)),
                             ]


@pytest.mark.parametrize('sensor_state, target_state, expected_measurement',
                         position_measurement_sets)
@pytest.mark.parametrize('model_class, measure_mapping, use_velocity',
                         [(CartesianToElevationBearing, [0, 1], False),
                          (CartesianToElevationBearingRange, [0, 1, 2], False),
                          (CartesianToElevationBearingRangeRate, [0, 1, 2, 3], True)])
def test_model_predictions(sensor_state, target_state, expected_measurement, model_class,
                           measure_mapping, use_velocity):
    sensor_state = StateVector(sensor_state)
    target_state = State(StateVector(target_state), timestamp=None)
    expected_measurement = StateVector([Elevation(expected_measurement[0]),
                                        Bearing(expected_measurement[1]),
                                        expected_measurement[2],  # range
                                        expected_measurement[3]])  # range rate
    pos_mapping = [0, 2, 4]
    vel_mapping = [1, 3, 5]
    sensor_velocity = sensor_state[vel_mapping]
    _, bearing, elevation = cart2sphere(*sensor_velocity)
    orientation = StateVector([0, elevation, bearing])
    model = model_class(ndim_state=6,
                        translation_offset=sensor_state[pos_mapping],
                        rotation_offset=orientation,
                        mapping=pos_mapping,
                        noise_covar=np.eye(len(expected_measurement)))
    if use_velocity:
        model.velocity = sensor_velocity
    actual_measurement = model.function(target_state, noise=False)
    assert np.allclose(actual_measurement, expected_measurement[measure_mapping])


def test_angle_pdf():
    model = CartesianToBearingRange(ndim_state=2,
                                    mapping=(0, 1),
                                    noise_covar=np.diag([np.radians(10), 2]))

    # Around 0 degrees
    measurement = State(StateVector([[Bearing(np.radians(1.))], [10.]]))
    x, y = pol2cart(10, np.radians(-1))
    state = State(StateVector([[x], [y]]))
    reference_probability = model.pdf(measurement, state)

    # Check same result around 90 degrees
    measurement.state_vector[0, 0] += np.radians(90)
    x, y = pol2cart(10, np.radians(89))
    state = State(StateVector([[x], [y]]))
    assert approx(reference_probability) == model.pdf(measurement, state)

    # Check same result around 180 degrees
    measurement.state_vector[0, 0] += np.radians(90)
    x, y = pol2cart(10, np.radians(179))
    state = State(StateVector([[x], [y]]))
    assert approx(reference_probability) == model.pdf(measurement, state)


def h2d_rr(state_vector, pos_map, vel_map, translation_offset, rotation_offset, velocity):

    xyz = StateVector([[state_vector[pos_map[0], 0] - translation_offset[0, 0]],
                      [state_vector[pos_map[1], 0] - translation_offset[1, 0]],
                      [0]])

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, _ = cart2sphere(*xyz_rot)

    # Calculate range rate extension
    # Determine the net velocity component in the engagement
    xyz_vel = np.array([[state_vector[vel_map[0], 0] - velocity[0, 0]],
                        [state_vector[vel_map[1], 0] - velocity[1, 0]],
                        [0]])

    # Use polar to calculate range rate
    rr = np.dot(xyz[:, 0], xyz_vel[:, 0]) / np.linalg.norm(xyz)

    return StateVector([Bearing(phi), rho, rr])


def h3d_rr(state_vector, pos_map, vel_map, translation_offset, rotation_offset, velocity):

    xyz = state_vector[pos_map, :] - translation_offset

    # Get rotation matrix
    theta_x, theta_y, theta_z = - rotation_offset[:, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, theta = cart2sphere(*xyz_rot)

    # Calculate range rate extension
    # Determine the net velocity component in the engagement
    xyz_vel = np.array([[state_vector[vel_map[0], 0] - velocity[0, 0]],
                        [state_vector[vel_map[1], 0] - velocity[1, 0]],
                        [state_vector[vel_map[2], 0] - velocity[2, 0]]])

    # Use polar to calculate range rate
    rr = np.dot(xyz[:, 0], xyz_vel[:, 0]) / np.linalg.norm(xyz)

    return StateVector([Elevation(theta), Bearing(phi), rho, rr])


@pytest.mark.parametrize(
    "h, modelclass, state_vec, ndim_state, pos_mapping, vel_mapping,\
    noise_covar, position, orientation",
    [
        (   # 3D meas, 6D state
            h2d_rr,  # h
            CartesianToBearingRangeRate,  # ModelClass
            StateVector([[200.], [10.], [0.], [0.], [0.], [0.]]),  # state_vec
            6,  # ndim_state
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            CovarianceMatrix([[0.05, 0, 0],
                              [0, 0.015, 0],
                              [0, 0, 10]]),  # noise_covar
            StateVector([[1], [-1], [0]]),  # position (translation offset)
            StateVector([[0], [0], [1]])  # orientation (rotation offset)
        ),
        (   # 3D meas, 6D state
            h2d_rr,  # h
            CartesianToBearingRangeRate,  # ModelClass
            StateVector([[200.], [10.], [0.], [0.], [0.], [0.]]),  # state_vec
            6,  # ndim_state
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            CovarianceMatrix([[0.05, 0, 0],
                              [0, 0.015, 0],
                              [0, 0, 10]]),  # noise_covar
            None,  # position (translation offset)
            None  # orientation (rotation offset)
        ),
        (   # 4D meas, 6D state
            h3d_rr,  # h
            CartesianToElevationBearingRangeRate,  # ModelClass
            StateVector([[200.], [10.], [0.], [0.], [0.], [0.]]),  # state_vec
            6,  # ndim_state
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            CovarianceMatrix([[0.05, 0, 0, 0],
                              [0, 0.05, 0, 0],
                              [0, 0, 0.015, 0],
                              [0, 0, 0, 10]]),  # noise_covar
            StateVector([[100], [0], [0]]),  # position (translation offset)
            StateVector([[0], [0], [0]])  # orientation (rotation offset)
        ),
        (   # 4D meas, 6D state
            h3d_rr,  # h
            CartesianToElevationBearingRangeRate,  # ModelClass
            StateVector([[200.], [10.], [0.], [0.], [0.], [0.]]),  # state_vec
            6,  # ndim_state
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            CovarianceMatrix([[0.05, 0, 0, 0],
                              [0, 0.05, 0, 0],
                              [0, 0, 0.015, 0],
                              [0, 0, 0, 10]]),  # noise_covar
            None,  # position (translation offset)
            None  # orientation (rotation offset)
        )
    ],
    ids=["rrRB_1", "rrRB_2", "rrRBE_1", "rrRBE_2"]
)
def test_rangeratemodels(h, modelclass, state_vec, ndim_state, pos_mapping, vel_mapping,
                         noise_covar, position, orientation):
    """ Test for the CartesianToBearingRangeRate and
    CartesianToElevationBearingRangeRate Measurement Models """

    state = State(state_vec)

    # Check default translation_offset, rotation_offset and velocity is applied
    model_test = modelclass(ndim_state=ndim_state,
                            mapping=pos_mapping,
                            velocity_mapping=vel_mapping,
                            noise_covar=noise_covar)

    assert len(model_test.translation_offset) == 3
    assert len(model_test.rotation_offset) == 3
    assert len(model_test.velocity) == 3

    # Create and a measurement model object
    model = modelclass(ndim_state=ndim_state,
                       mapping=pos_mapping,
                       velocity_mapping=vel_mapping,
                       noise_covar=noise_covar,
                       translation_offset=position,
                       rotation_offset=orientation)

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state)
    eval_m = h(state_vec,
               model.mapping,
               model.velocity_mapping,
               model.translation_offset,
               model.rotation_offset,
               model.velocity)
    assert np.array_equal(meas_pred_wo_noise, eval_m)

    # Ensure ```lg.transfer_function()``` returns H
    def fun(x):
        return model.function(x)

    H = compute_jac(fun, state)
    assert np.array_equal(H, model.jacobian(state))

    # Check Jacobian has proper dimensions
    assert H.shape == (model.ndim_meas, ndim_state)

    # Ensure inverse function returns original
    if isinstance(model, ReversibleModel):
        J = model.inverse_function(State(meas_pred_wo_noise))
        assert np.allclose(J, state_vec)

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(noise_covar, model.covar())

    # Ensure model creates noise
    rvs = model.rvs()
    assert rvs.shape == (model.ndim_meas, 1)
    assert isinstance(rvs, StateVector)
    rvs = model.rvs(10)
    assert rvs.shape == (model.ndim_meas, 10)
    assert isinstance(rvs, StateVectors)
    # StateVector is subclass of Matrix, so need to check explicitly.
    assert not isinstance(rvs, StateVector)

    # Project a state throught the model
    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state)
    assert np.array_equal(meas_pred_wo_noise, h(state_vec,
                                                model.mapping,
                                                model.velocity_mapping,
                                                model.translation_offset,
                                                model.rotation_offset,
                                                model.velocity))

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = model.pdf(State(meas_pred_wo_noise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_wo_noise
         - h(state_vec, model.mapping, model.velocity_mapping, model.translation_offset,
             model.rotation_offset, model.velocity)
         ).ravel(),
        cov=noise_covar)

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = model.function(state, noise=True)
    assert not np.array_equal(
        meas_pred_w_inoise, h(state_vec,
                              model.mapping,
                              model.velocity_mapping,
                              model.translation_offset,
                              model.rotation_offset,
                              model.velocity))

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_inoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_w_inoise
         - h(state_vec, model.mapping, model.velocity_mapping, model.translation_offset,
             model.rotation_offset, model.velocity)
         ).ravel(),
        cov=noise_covar)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = model.rvs()
    meas_pred_w_enoise = model.function(state,
                                        noise=noise)
    assert np.array_equal(meas_pred_w_enoise, h(state_vec,
                                                model.mapping,
                                                model.velocity_mapping,
                                                model.translation_offset,
                                                model.rotation_offset,
                                                model.velocity) + noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_enoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        (meas_pred_w_enoise
         - h(state_vec, model.mapping, model.velocity_mapping, model.translation_offset,
             model.rotation_offset, model.velocity)
         ).ravel(),
        cov=noise_covar)


def test_inverse_function():
    measure_model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=np.array([0, 2, 4]),
        velocity_mapping=np.array([1, 3, 5]),
        noise_covar=np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]]))

    measured_state = State(StateVector([np.pi / 18, np.pi / 18, 10e3, 100.0]))

    inv_measure_state = measure_model.inverse_function(measured_state)

    assert approx(inv_measure_state[0], 0.02) == 9698.46
    assert approx(inv_measure_state[1], 0.02) == 96.98
    assert approx(inv_measure_state[2], 0.02) == 1710.1
    assert approx(inv_measure_state[3], 0.02) == 17.10
    assert approx(inv_measure_state[4], 0.02) == 1736.48
    assert approx(inv_measure_state[5], 0.02) == 17.36


def test_binning():
    real_state = State(state_vector=StateVector([10e3, 100., 10e3, 100., 10e3, 100.]))

    measurement_model = RangeRangeRateBinning(range_res=10,
                                              range_rate_res=5,
                                              ndim_state=6,
                                              mapping=[0, 2, 4],
                                              velocity_mapping=[1, 3, 5],
                                              noise_covar=np.array([100,
                                                                   10,
                                                                   np.pi/18,
                                                                   np.pi/18]))

    measured = measurement_model.function(real_state, noise=True)
    assert ((measured[2, 0]-measurement_model.range_res/2) /
            measurement_model.range_res).is_integer()
    assert ((measured[3, 0]-measurement_model.range_rate_res/2) /
            measurement_model.range_rate_res).is_integer()


def test_binning_pdf():
    real_state = State(state_vector=StateVector([10e3, 100., 10e3, 100., 10e3, 100.]))

    measurement_model = RangeRangeRateBinning(range_res=10,
                                              range_rate_res=5,
                                              ndim_state=6,
                                              mapping=[0, 2, 4],
                                              velocity_mapping=[1, 3, 5],
                                              noise_covar=np.array([100,
                                                                   10,
                                                                   np.pi/18,
                                                                   np.pi/18]))

    measured = measurement_model.function(real_state, noise=True)
    pdf = measurement_model.pdf(State(measured), real_state)
    assert pdf != 0
    not_measured = measured.copy()
    not_measured[2, 0] = not_measured[2, 0] + 0.5*measurement_model.range_res
    pdf = measurement_model.pdf(State(not_measured), real_state)
    assert pdf == 0
    not_measured = measured.copy()
    not_measured[3, 0] = not_measured[3, 0] + 0.5*measurement_model.range_rate_res
    pdf = measurement_model.pdf(State(not_measured), real_state)
    assert pdf == 0


def test_binning_integral():
    measurement_model = RangeRangeRateBinning(range_res=None,
                                              range_rate_res=None,
                                              ndim_state=None,
                                              mapping=[],
                                              velocity_mapping=[],
                                              noise_covar=None)

    mean = 33.33333
    a = 40
    b = 30
    cov = 10
    expected_integral = 0.8365720412132509
    assert approx(measurement_model._gaussian_integral(a, b, mean, cov), 0.02) == expected_integral

    bin_sizes = 10
    state_vector1 = 35
    expected_pdf = 0.08365720412132509
    assert (approx(measurement_model._binned_pdf(state_vector1, mean, bin_sizes, cov)) ==
            expected_pdf)

@pytest.mark.parametrize('sensor_state, target_state, expected_measurement',
                         position_measurement_sets)
def test_noiseless_binning_predictions(sensor_state, target_state, expected_measurement):
    sensor_state = StateVector(sensor_state)
    target_state = State(StateVector(target_state), timestamp=None)
    expected_measurement = StateVector([Elevation(expected_measurement[0]),
                                        Bearing(expected_measurement[1]),
                                        expected_measurement[2],  # range
                                        expected_measurement[3]])  # range rate
    pos_mapping = [0, 2, 4]
    vel_mapping = [1, 3, 5]
    sensor_velocity = sensor_state[vel_mapping]
    _, bearing, elevation = cart2sphere(*sensor_velocity)
    orientation = StateVector([0, elevation, bearing])
    model = RangeRangeRateBinning(
        range_res=3,
        range_rate_res=1,
        ndim_state=6,
        mapping=pos_mapping,
        velocity_mapping=vel_mapping,
        noise_covar=np.array([0., 0., 0., 0.]),
        translation_offset=sensor_state[pos_mapping],
        rotation_offset=orientation,
        velocity=sensor_velocity)
    actual_measurement = model.function(target_state, noise=False)
    measure_mapping = [0, 1, 2, 3]
    assert np.allclose(actual_measurement, expected_measurement[measure_mapping])


def test_compare_rrrb_to_ctebrr():
    model = RangeRangeRateBinning(
        range_res=0.00001,
        range_rate_res=0.00001,
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.array([1., 1., 1., 1.]))

    state = State([50.000005, 50.000005,
                   0., 0.,
                   0., 0.])
    out = model.function(state)
    act_pdf = model.pdf(State(out), state)

    compare_model = CartesianToElevationBearingRangeRate(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=np.diag([1, 1, 1, 1]))

    out = compare_model.function(state, noise=False)
    exp_pdf = compare_model.pdf(State(out), state)
    assert np.isclose(float(act_pdf), float(exp_pdf))
