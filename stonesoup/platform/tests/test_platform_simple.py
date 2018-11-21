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
    # Generate 5 radar model
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # Note 1 - the radar position is irrelevant once mounted
    radar1_position = StateVector(np.array(([[100],
                                             [100]])))
    radar2_position = StateVector(np.array(([[100], [100]])))
    radar3_position = StateVector(np.array(([[100], [100]])))
    radar4_position = StateVector(np.array(([[100], [100]])))
    radar5_position = StateVector(np.array(([[100], [100]])))

    measurement_mapping = np.array([0, 2])

    # Create 5 simple radar
    # Create a radar object
    radar1 = SimpleRadar(
        position=radar1_position,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar2 = SimpleRadar(
        position=radar2_position,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar3 = SimpleRadar(
        position=radar3_position,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar4 = SimpleRadar(
        position=radar4_position,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar5 = SimpleRadar(
        position=radar5_position,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    # new_timestamp2 = new_timestamp + datetime.timedelta(seconds=timediff)

    # Define transition model and position for platform
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    model_2d = CombinedLinearGaussianTransitionModel([model_1d, model_1d])

    # Define a 2d platform with a simple velocity [1,0] starting at the origin
    platform_state = State(np.array([[0],
                                     [1],
                                     [0],
                                     [0]]),
                           timestamp)

    # Define a mounting offset for a sensor relative to the platform -
    #  0,0 offset
    mounting_offsets = np.array([[0, 0],
                                 [1, 0],
                                 [0, 1],
                                 [-1, 0],
                                 [0, -1]])
    # This defines the mapping to the platforms state vector (i.e. x and y)
    mounting_mappings = np.array([[0, 2]])

    # create a platform with the simple radar mounted
    platform1 = SensorPlatform(platform_state,
                               model_2d,
                               [radar1, radar2, radar3, radar4, radar5],
                               mounting_offsets,
                               mounting_mappings)

    # Check that mounting_mappings has been modified to match number of sensor
    assert(platform1.mounting_mappings.shape[0] == len(platform1.sensors))

    # Check that the sensor position has been modified to reflect the offset
    for i in range(len(platform1.sensors)):
        radar_position = platform1.sensors[i].position
        expected_radar_position = np.zeros(
            [platform1.mounting_offsets.shape[1], 1])
        for j in range(platform1.mounting_offsets.shape[1]):
            expected_radar_position[j, 0] = (platform1.mounting_offsets[i, j] +
                                             platform1.state.state_vector[
                                                 platform1.mounting_mappings[
                                                     i, j]])
        assert (np.equal(expected_radar_position, radar_position).all())

    # Define a 2d platform with a simple velocity [0, 1] starting at the origin
    platform_state2 = State(np.array([[10],
                                      [0],
                                      [0],
                                      [1]]),
                            timestamp)
    # create a platform with the simple radar mounted
    platform2 = SensorPlatform(platform_state2,
                               model_2d,
                               [radar1, radar2, radar3, radar4, radar5],
                               mounting_offsets,
                               mounting_mappings)

    # Define the expected radar position
    rotated_radar_positions = np.array([[0, 0],
                                        [0, 1],
                                        [-1, 0],
                                        [0, -1],
                                        [1, 0]])  # 0, +1, +1, +1, -3

    # mounting_offsets = np.array([[0, 0], -> [0. 0], index
    #                             [1, 0],  -> [0, 1], index +1
    #                             [0, 1],  -> [-1, 0], index +1
    #                             [-1, 0], -> [0, -1], index +1
    #                             [0, -1]])-> [1, 0], index -3

    # This will match each sensor location to the mounting offsets provided
    expected_radar_position = np.zeros(
        [len(platform2.sensors), platform2.mounting_offsets.shape[1]])
    for i in range(len(platform2.sensors)):
        radar_position = platform2.sensors[i].position
        for j in range(mounting_offsets.shape[1]):
            expected_radar_position[i, j] = (rotated_radar_positions[i, j] +
                                             platform2.state.state_vector[
                                                 platform2.mounting_mappings[
                                                     i, j]])
            # check that outputs are close (allowing for rounding errors in
            # floating point maths)
            assert (np.isclose(expected_radar_position[i, j],
                               radar_position[j], atol=1e-8))

    # Move the unrotated platform... does not work?
    platform1.move(new_timestamp)

    # This will match each sensor location to the mounting offsets provided
    expected_radar_position = np.zeros(
        [len(platform1.sensors), platform1.mounting_offsets.shape[1]])
    for i in range(len(platform1.sensors)):
        radar_position = platform1.sensors[i].position
        for j in range(platform1.mounting_offsets.shape[1]):
            expected_radar_position[i, j] = (platform1.mounting_offsets[i, j] +
                                             platform1.state.state_vector[
                                                 platform1.mounting_mappings[
                                                     i, j]])
            # check that outputs are close (allowing for rounding errors in
            # floating point maths
            assert (np.isclose(expected_radar_position[i, j],
                               radar_position[j], atol=1e-8))

    # Move the platform which has rotated sensors
    platform2.move(new_timestamp)

    # This will match each sensor location to the mounting offsets provided
    expected_radar_position = np.zeros(
        [len(platform2.sensors), platform2.mounting_offsets.shape[1]])
    for i in range(len(platform2.sensors)):
        radar_position = platform2.sensors[i].position
        for j in range(platform2.mounting_offsets.shape[1]):
            expected_radar_position[i, j] = (rotated_radar_positions[i, j] +
                                             platform2.state.state_vector[
                                                 platform2.mounting_mappings[
                                                     i, j]])
            # check that outputs are close (allowing for rounding errors in
            # floating point maths
            assert (np.isclose(expected_radar_position[i, j],
                               radar_position[j], atol=1e-8))
