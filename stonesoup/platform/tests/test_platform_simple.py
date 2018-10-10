# coding: utf-8
# import pytest
import datetime
import numpy as np

from stonesoup.types.state import State
from stonesoup.platform.simple import SensorPlatform
from stonesoup.models.transition.linear import ConstantVelocity,\
    CombinedLinearGaussianTransitionModel
from stonesoup.functions import cart2pol
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.sensor.radar import SimpleRadar

# Input arguments
# TODO: pytest parametarization


def test_sensor_platform():
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))
    radar_position = StateVector(
        np.array(([[1], [1]])))
    target_state = State(radar_position +
                         np.array([[1], [1]]),
                         timestamp=datetime.datetime.now())

    # Create a radar object
    radar = SimpleRadar(radar_position, noise_covar)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    new_timestamp2 = new_timestamp + datetime.timedelta(seconds=timediff)

    # Define transition model and position
    model_1d = ConstantVelocity(0.5)
    model_2d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d])

    # Define a new platform
    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1]]),
                           timestamp)
    mounting_offsets = np.array([[0], [0]])
    mounting_mappings = np.array([[0], [2]])
    platform = SensorPlatform(platform_state,
                              model_2d,
                              [radar],
                              mounting_offsets,
                              mounting_mappings)

    # Move the platform
    platform.move(new_timestamp)

    # Assert that the radar position has also been updated
    platform_position = StateVector([[platform.state.state_vector[0][0]],
                                     [platform.state.state_vector[2][0]]])
    assert(np.equal(platform_position, radar.position).all())

    # Add mounting offset, move platform and assert correctness
    new_mounting_offset = np.array([[0.25], [5]])
    platform.mounting_offsets = new_mounting_offset
    platform.move(new_timestamp2)
    platform_position = StateVector([[platform.state.state_vector[0][0]],
                                     [platform.state.state_vector[2][0]]])
    assert(np.equal(platform_position+new_mounting_offset,
                    radar.position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.gen_measurement(target_state, noise=0)
    rho, phi = cart2pol(target_state.state_vector[0][0]
                        - radar.position[0][0],
                        target_state.state_vector[1][0]
                        - radar.position[1][0])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector,
                    StateVector(np.array([[phi], [rho]]))).all())
