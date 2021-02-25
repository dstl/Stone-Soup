# -*- coding: utf-8 -*-
import datetime

import numpy as np

from stonesoup.types.detection import TrueDetection
from ..passive import PassiveElevationBearing
from ...functions import cart2angles, rotx, roty, rotz
from ...types.array import StateVector, CovarianceMatrix
from ...types.groundtruth import GroundTruthState, GroundTruthPath


def test_passive_sensor():
    # Input arguments
    # TODO: pytest parametarization
    noise_covar = CovarianceMatrix([[np.deg2rad(0.015), 0],
                                    [0, np.deg2rad(0.1)]])
    detector_position = StateVector([1, 1, 0])
    detector_orientation = StateVector([0, 0, 0])

    target_state = GroundTruthState(detector_position + np.array([[1], [1], [0]]),
                                    timestamp=datetime.datetime.now())
    target_truth = GroundTruthPath([target_state])

    truth = {target_truth}

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
    measurement = detector.measure(truth, noise=False)
    measurement = next(iter(measurement))  # Get measurement from set

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
    phi, theta = cart2angles(*xyz_rot)

    # Assert correction of generated measurement
    assert (measurement.timestamp == target_state.timestamp)
    assert (np.equal(measurement.state_vector,
                     StateVector([theta, phi])).all())

    # Assert is TrueDetection type
    assert isinstance(measurement, TrueDetection)
    assert measurement.groundtruth_path is target_truth
    assert isinstance(measurement.groundtruth_path, GroundTruthPath)

    target2_state = GroundTruthState(detector_position + np.array([[-1], [-1], [0]]),
                                     timestamp=datetime.datetime.now())
    target2_truth = GroundTruthPath([target2_state])

    truth.add(target2_truth)

    # Generate a noiseless measurement for each of the given target states
    measurements = detector.measure(truth, noise=False)

    # Two measurements for 2 truth states
    assert len(measurements) == 2

    # Measurements store ground truth paths
    for measurement in measurements:
        assert measurement.groundtruth_path in truth
        assert isinstance(measurement.groundtruth_path, GroundTruthPath)
