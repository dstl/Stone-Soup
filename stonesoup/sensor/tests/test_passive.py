# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ...functions import cart2angles, rotx, roty, rotz
from ...types.array import StateVector, CovarianceMatrix
from ...types.state import State
from ..passive import PassiveElevationBearing


def test_passive_sensor():
    # Input arguments
    # TODO: pytest parametarization
    noise_covar = CovarianceMatrix([[np.deg2rad(0.015), 0],
                                   [0, np.deg2rad(0.1)]])
    detector_position = StateVector([1, 1, 0])
    detector_orientation = StateVector([0, 0, 0])
    target_state = State(detector_position +
                         np.array([[1], [1], [0]]),
                         timestamp=datetime.datetime.now())
    measurement_mapping = np.array([0, 1, 2])

    # Create a radar object
    detector = PassiveElevationBearing(
        position=detector_position,
        orientation=detector_orientation,
        ndim_state=3,
        mapping=measurement_mapping,
        noise_covar=noise_covar)

    # Assert that the object has been correctly initialised
    assert (np.equal(detector.position, detector_position).all())

    # Generate a noiseless measurement for the given target
    measurement = detector.measure(target_state, noise=False)

    # Account
    xyz = target_state.state_vector - detector_position

    # Calculate rotation matrix
    theta_x = -detector_orientation[0, 0]
    theta_y = -detector_orientation[1, 0]
    theta_z = -detector_orientation[2, 0]
    rot_mat = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

    # Rotate coordinates
    xyz_rot = rot_mat @ xyz

    # Convert to Angles
    phi, theta = cart2angles(*xyz_rot[:, 0])

    # Assert correction of generated measurement
    assert (measurement.timestamp == target_state.timestamp)
    assert (np.equal(measurement.state_vector,
                     StateVector([theta, phi])).all())
