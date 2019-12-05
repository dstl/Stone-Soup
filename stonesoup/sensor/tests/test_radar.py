# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ...functions import cart2pol
from ...types.angle import Bearing
from ...types.array import StateVector, CovarianceMatrix
from ...types.state import State
from ..radar import RadarRangeBearing, RadarRotatingRangeBearing


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


def test_simple_radar():

    # Input arguments
    # TODO: pytest parametarization
    noise_covar = CovarianceMatrix([[0.015, 0],
                                   [0, 0.1]])
    radar_position = StateVector([1, 1])
    radar_orientation = StateVector([0, 0, 0])
    target_state = State(radar_position +
                         np.array([[1], [1]]),
                         timestamp=datetime.datetime.now())
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRangeBearing(
        position=radar_position,
        orientation=radar_orientation,
        ndim_state=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())
    assert(np.equal(radar.measurement_model.translation_offset,
                    radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=0)
    rho, phi = cart2pol(target_state.state_vector[0, 0]
                        - radar_position[0, 0],
                        target_state.state_vector[1, 0]
                        - radar_position[1, 0])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector,
                    StateVector([phi, rho])).all())


def test_rotating_radar():

    # Input arguments
    # TODO: pytest parametarization
    timestamp = datetime.datetime.now()
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # The radar is positioned at (1,1)
    radar_position = StateVector(
        np.array(([[1], [1]])))
    # The radar is facing left/east
    radar_orientation = StateVector([[0], [0], [np.pi]])
    # The radar antenna is facing opposite the radar orientation
    dwell_center = State(StateVector([[-np.pi]]),
                         timestamp=timestamp)
    rpm = 20            # 20 Rotations Per Minute
    max_range = 100     # Max range of 100m
    fov_angle = np.pi/3       # FOV angle of pi/3

    target_state = State(radar_position +
                         np.array([[5], [5]]),
                         timestamp=timestamp)
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRotatingRangeBearing(
        position=radar_position,
        orientation=radar_orientation,
        ndim_state=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar,
        dwell_center=dwell_center,
        rpm=rpm,
        max_range=max_range,
        fov_angle=fov_angle)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())
    assert(np.equal(radar.measurement_model.translation_offset,
                    radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=0)

    # Assert measurement is None since target is not in FOV
    assert(measurement is None)

    # Rotate radar such that the target is in FOV
    timestamp = timestamp + datetime.timedelta(seconds=0.5)
    target_state = State(radar_position +
                         np.array([[5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=0)
    eval_m = h2d(target_state.state_vector,
                 radar.position,
                 radar.orientation+[[0],
                                    [0],
                                    [radar.dwell_center.state_vector[0, 0]]])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector, eval_m).all())
