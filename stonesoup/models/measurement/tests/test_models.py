# -*- coding: utf-8 -*-
import pytest
from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ..nonlinear import (
    CartesianToElevationBearingRange, CartesianToBearingRange,
    CartesianToElevationBearing)
from ...base import ReversibleModel
from ....types.state import State
from ....functions import jacobian as compute_jac
from ....types.angle import Bearing, Elevation


def h2d(state_vector, translation_offset, rotation_offset):

    xyz = [[state_vector[0, 0] - translation_offset[0, 0]],
           [state_vector[1, 0] - translation_offset[1, 0]],
           [0]]

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    rot_z = np.array([[cos_z, -sin_z, 0],
                      [sin_z, cos_z, 0],
                      [0, 0, 1]])

    theta_y = - rotation_offset[1, 0]
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    rot_y = np.array([[cos_y, 0, sin_y],
                      [0, 1, 0],
                      [-sin_y, 0, cos_y]])

    theta_x = - rotation_offset[0, 0]
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    rot_x = np.array([[1, 0, 0],
                      [0, cos_x, -sin_x],
                      [0, sin_x, cos_x]])

    rotation_matrix = rot_z@rot_y@rot_x

    xyz_rot = rotation_matrix @ xyz
    x = xyz_rot[0, 0]
    y = xyz_rot[1, 0]
    # z = 0  # xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([[Bearing(phi)], [rho]])


def h3d(state_vector,  translation_offset, rotation_offset):

    xyz = [[state_vector[0, 0] - translation_offset[0, 0]],
           [state_vector[1, 0] - translation_offset[1, 0]],
           [state_vector[2, 0] - translation_offset[2, 0]]]

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    rot_z = np.array([[cos_z, -sin_z, 0],
                      [sin_z, cos_z, 0],
                      [0, 0, 1]])

    theta_y = - rotation_offset[1, 0]
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    rot_y = np.array([[cos_y, 0, sin_y],
                      [0, 1, 0],
                      [-sin_y, 0, cos_y]])

    theta_x = - rotation_offset[0, 0]
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    rot_x = np.array([[1, 0, 0],
                      [0, cos_x, -sin_x],
                      [0, sin_x, cos_x]])

    rotation_matrix = rot_z@rot_y@rot_x

    xyz_rot = rotation_matrix @ xyz
    x = xyz_rot[0, 0]
    y = xyz_rot[1, 0]
    z = xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)

    return np.array([[Elevation(theta)], [Bearing(phi)], [rho]])


def hbearing(state_vector, translation_offset, rotation_offset):
    xyz = [[state_vector[0, 0] - translation_offset[0, 0]],
           [state_vector[1, 0] - translation_offset[1, 0]],
           [state_vector[2, 0] - translation_offset[2, 0]]]

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    rot_z = np.array([[cos_z, -sin_z, 0],
                      [sin_z, cos_z, 0],
                      [0, 0, 1]])

    theta_y = - rotation_offset[1, 0]
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    rot_y = np.array([[cos_y, 0, sin_y],
                      [0, 1, 0],
                      [-sin_y, 0, cos_y]])

    theta_x = - rotation_offset[0, 0]
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    rot_x = np.array([[1, 0, 0],
                      [0, cos_x, -sin_x],
                      [0, sin_x, cos_x]])

    rotation_matrix = rot_z@rot_y@rot_x

    xyz_rot = rotation_matrix @ xyz
    x = xyz_rot[0, 0]
    y = xyz_rot[1, 0]
    z = xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)

    return np.array([[Elevation(theta)], [Bearing(phi)]])


@pytest.mark.parametrize(
    "h, ModelClass, state_vec, R , mapping,\
     translation_offset, rotation_offset",
    [
        (   # 2D meas, 2D state
            h2d,
            CartesianToBearingRange,
            np.array([[0], [1]]),
            np.array([[0.015, 0],
                      [0, 0.1]]),
            np.array([0, 1]),
            np.array([[1], [-1]]),
            np.array([[0], [0], [1]])

        ),
        (   # 3D meas, 3D state
            h3d,
            CartesianToElevationBearingRange,
            np.array([[1], [2], [2]]),
            np.array([[0.05, 0, 0],
                      [0, 0.015, 0],
                      [0, 0, 0.1]]),
            np.array([0, 1, 2]),
            np.array([[0], [0], [0]]),
            np.array([[.2], [3], [-1]])
        ),
        (   # 2D meas, 3D state
            hbearing,
            CartesianToElevationBearing,
            np.array([[1], [2], [3]]),
            np.array([[0.05, 0],
                      [0, 0.015]]),
            np.array([0, 1, 2]),
            np.array([[0], [0], [0]]),
            np.array([[-3], [0], [np.pi/3]])
        )
    ],
    ids=["standard", "RBE", "BearingsOnly"]
)
def test_models(h, ModelClass, state_vec, R,
                mapping, translation_offset, rotation_offset):
    """ CartesianToBearingRange Measurement Model test """

    ndim_state = state_vec.size
    state = State(state_vec)

    # Create and a measurement model object
    model = ModelClass(ndim_state=ndim_state,
                       mapping=mapping,
                       noise_covar=R,
                       translation_offset=translation_offset,
                       rotation_offset=rotation_offset)

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state, noise=0)
    eval_m = h(state_vec, model.translation_offset, model.rotation_offset)
    assert np.array_equal(meas_pred_wo_noise, eval_m)

    # Ensure ```lg.transfer_function()``` returns H
    def fun(x):
        return model.function(x, noise=0)
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

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = model.function(state, noise=0)
    assert np.array_equal(meas_pred_wo_noise,  h(
        state_vec, model.translation_offset, model.rotation_offset))

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = model.pdf(State(meas_pred_wo_noise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(h(state_vec,
                        model.translation_offset,
                        model.rotation_offset)).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = model.function(state)
    assert not np.array_equal(
        meas_pred_w_inoise,  h(state_vec, model.translation_offset,
                               model.rotation_offset))

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_inoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(h(state_vec,
                        model.translation_offset,
                        model.rotation_offset)).ravel(),
        cov=R)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = model.rvs()
    meas_pred_w_enoise = model.function(state,
                                        noise=noise)
    assert np.array_equal(meas_pred_w_enoise,  h(
        state_vec, model.translation_offset, model.rotation_offset)+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(State(meas_pred_w_enoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(h(state_vec,
                        model.translation_offset,
                        model.rotation_offset)).ravel(),
        cov=R)
