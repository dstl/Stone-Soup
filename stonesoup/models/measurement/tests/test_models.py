# -*- coding: utf-8 -*-
import pytest
from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ....functions import rotz, rotx, roty, cart2sphere
from ..nonlinear import (
    CartesianToElevationBearingRange, CartesianToBearingRange,
    CartesianToElevationBearing, CartesianToBearingRangeRate,
    CartesianToElevationBearingRangeRate)
from ...base import ReversibleModel
from ....types.state import State, CovarianceMatrix
from ....functions import jacobian as compute_jac
from ....types.angle import Bearing, Elevation
from ....types.array import StateVector, StateVectors
from ....functions import pol2cart


def h2d(state_vector, pos_map, translation_offset, rotation_offset):

    xyz = [[state_vector[0, 0] - translation_offset[0, 0]],
           [state_vector[1, 0] - translation_offset[1, 0]],
           [0]]

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
    ids=["BearingElevation1", "BearingElevation2",
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

    xyz = np.array([[state_vector[pos_map[0], 0] - translation_offset[0, 0]],
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
