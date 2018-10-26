# coding: utf-8
# import pytest
import datetime
import numpy as np

from stonesoup.types.state import State
from stonesoup.platform.simple import SensorPlatform
from stonesoup.models.transition.linear import ConstantVelocity,\
    CombinedLinearGaussianTransitionModel
# from stonesoup.functions import cart2pol
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.sensor.radar import SimpleRadar

# Input arguments
# TODO: pytest parametarization


def test_sensor_platform():
    # Create a pair of radars
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))
    radar_position = StateVector(
        np.array(([[1], [1]])))

    radar1 = SimpleRadar(radar_position, noise_covar)
    radar2 = SimpleRadar(radar_position, noise_covar)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    # new_timestamp2 = new_timestamp + datetime.timedelta(seconds=timediff)

    # Define transition model and position
    model_1d = ConstantVelocity(0.0)
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])

    # Define a new platform
    platform_state = State(np.array([[0],
                                     [1],
                                     [0],
                                     [0],
                                     [0],
                                     [0]]),
                           timestamp)
    # define a mounting offset which is different from the sensor position
    mounting_offsets = np.array([[0, 0, 0],
                                 [1, 1, 1]])
    mounting_mappings = np.array([[0, 2, 4],
                                  [0, 2, 4]])
    platform = SensorPlatform(platform_state,
                              model_3d,
                              [radar1, radar2],
                              mounting_offsets,
                              mounting_mappings)

    # Check that the sensor position has been modified to reflect the offset
    for i in range(len(platform.sensors)):
        radar_position = platform.sensors[i].position
        platform_position = StateVector([platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 0]] +
                                         platform.mounting_offsets[i, 0],
                                         platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 1]] +
                                         platform.mounting_offsets[i, 1],
                                         platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 2]] +
                                         platform.mounting_offsets[i, 2]])
        # Asset radar_position equals platform_position
        assert(np.equal(platform_position, radar_position).all())

    # Move the platform
    platform.move(new_timestamp)

    # Check both platform and sensor have been correctly moved
    for i in range(len(platform.sensors)):
        radar_position = platform.sensors[i].position
        platform_position = StateVector([platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 0]] +
                                         platform.mounting_offsets[i, 0],
                                         platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 1]] +
                                         platform.mounting_offsets[i, 1],
                                         platform.state.state_vector[
                                             platform.mounting_mappings[
                                                 i, 2]] +
                                         platform.mounting_offsets[i, 2]])
        # Asset radar_position equals platform_position
        assert(np.equal(platform_position, radar_position).all())

    # TODO: add tests to cope with sensors rotating around platform vector
