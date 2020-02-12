# coding: utf-8
import datetime

import numpy as np
import pytest

from ...types.state import State
from ...platform.simple import SensorPlatform
from ...models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
from ...sensor.radar.radar import RadarRangeBearing
from ...types.array import StateVector, CovarianceMatrix

# Input arguments
# TODO: pytest parametarization


def get_3d_expected(i):
    if i == 0:
        # static platform or X velocity
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0],
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    elif i == 1:
        # y-axis motion
        return np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
                         [1, 0, 0], [0, 0, 1], [0, 0, -1]])
    elif i == 2:
        # negative x-axis motion
        return np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0],
                         [0, 1, 0], [0, 0, 1], [0, 0, -1]])
    elif i == 3:
        # negative y-axis motion
        return np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0],
                         [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
    elif i == 4:
        # x-y motion
        return np.array([[0, 0, 0], [1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), -1/np.sqrt(2), 0], [0, 0, 1],
                         [0, 0, -1]])
    elif i == 5:
        # neg x- neg y motion
        return np.array([[0, 0, 0], [-1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 1],
                         [0, 0, -1]])
    elif i == 6:
        # pos x- neg y motion
        return np.array([[0, 0, 0], [1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), -1/np.sqrt(2), 0], [0, 0, 1],
                         [0, 0, -1]])
    elif i == 7:
        # neg x- pos y motion
        return np.array([[0, 0, 0], [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 1],
                         [0, 0, -1]])
    elif i == 8:
        # "z vel"
        return np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, -1],
                         [0, -1, 0], [-1, 0, 0], [1, 0, 0]])
    elif i == 9:
        # "-z vel"
        return np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1],
                         [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
    elif i == 10:
        # "y.z vel"
        return np.array([[0, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), -1/np.sqrt(2)],
                         [1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 11:
        # "y.-z vel"
        return np.array([[0, 0, 0], [0,  1/np.sqrt(2), -1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, -1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 12:
        # "-y.z vel"
        return np.array([[0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                         [-1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, -1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 13:
        # "-y.-z vel"
        return np.array([[0, 0, 0], [0, -1/np.sqrt(2), -1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 14:
        # x.z vel
        return np.array([[0, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), -1/np.sqrt(2)],
                         [1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 15:
        # -x.z vel
        return np.array([[0, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                         [-1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, -1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 16:
        # x.-z vel
        return np.array([[0, 0, 0], [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, -1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 17:
        # -x,-z vel
        return np.array([[0, 0, 0], [0, -1/np.sqrt(2), -1/np.sqrt(2)],
                         [1, 0, 0], [0, 1/np.sqrt(2), 1/np.sqrt(2)],
                         [-1, 0, 0], [0, -1/np.sqrt(2), 1/np.sqrt(2)],
                         [0, 1/np.sqrt(2), -1/np.sqrt(2)]])
    elif i == 18:
        # x.y.z vel
        a = np.cos(np.arctan2(1, np.sqrt(2)) * -1)
        b = np.sin(np.arctan2(1, np.sqrt(2)) * -1) / np.sqrt(2)
        return np.array([[0, 0, 0], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                         [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                         [1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [b, b, a], [-b, -b, -a]])
    elif i == 19:
        # -x.-y.-z vel
        a = np.cos(np.arctan2(-1, np.sqrt(2)) * -1)
        b = np.sin(np.arctan2(-1, np.sqrt(2)) * -1) / np.sqrt(2)
        return np.array([[0, 0, 0],
                         [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
                         [1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                         [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                         [-b, -b, a], [b, b, -a]])


@pytest.fixture
def radars_2d():
    # Generate 5 radar models for testing purposes
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # define arbitrary sensor origin
    radar1_position = StateVector(np.array(([[100], [100]])))
    radar1_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar2_position = StateVector(np.array(([[100], [100]])))
    radar2_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar3_position = StateVector(np.array(([[100], [100]])))
    radar3_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar4_position = StateVector(np.array(([[100], [100]])))
    radar4_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar5_position = StateVector(np.array(([[100], [100]])))
    radar5_orientation = StateVector(np.array(([[0], [0], [0]])))

    measurement_mapping = np.array([0, 2])

    # Create 5 simple radar sensor objects
    radar1 = RadarRangeBearing(
        position=radar1_position,
        orientation=radar1_orientation,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar,
    )

    radar2 = RadarRangeBearing(
        position=radar2_position,
        orientation=radar2_orientation,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar3 = RadarRangeBearing(
        position=radar3_position,
        orientation=radar3_orientation,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar4 = RadarRangeBearing(
        position=radar4_position,
        orientation=radar4_orientation,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar5 = RadarRangeBearing(
        position=radar5_position,
        orientation=radar5_orientation,
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    return [radar1, radar2, radar3, radar4, radar5]


@pytest.fixture
def radars_3d():
    # Generate 7 radar models for testing purposes
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # Note 1 - the radar position is irrelevant once mounted
    radar1_position = StateVector(np.array(([[100], [100], [100]])))
    radar1_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar2_position = StateVector(np.array(([[100], [100], [100]])))
    radar2_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar3_position = StateVector(np.array(([[100], [100], [100]])))
    radar3_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar4_position = StateVector(np.array(([[100], [100], [100]])))
    radar4_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar5_position = StateVector(np.array(([[100], [100], [100]])))
    radar5_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar6_position = StateVector(np.array(([[100], [100], [100]])))
    radar6_orientation = StateVector(np.array(([[0], [0], [0]])))
    radar7_position = StateVector(np.array(([[100], [100], [100]])))
    radar7_orientation = StateVector(np.array(([[0], [0], [0]])))

    measurement_mapping = np.array([0, 2, 4])

    # Create 5 simple radar sensor objects
    radar1 = RadarRangeBearing(
        position=radar1_position,
        orientation=radar1_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar2 = RadarRangeBearing(
        position=radar2_position,
        orientation=radar2_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar3 = RadarRangeBearing(
        position=radar3_position,
        orientation=radar3_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar4 = RadarRangeBearing(
        position=radar4_position,
        orientation=radar4_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar5 = RadarRangeBearing(
        position=radar5_position,
        orientation=radar5_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar6 = RadarRangeBearing(
        position=radar6_position,
        orientation=radar6_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar7 = RadarRangeBearing(
        position=radar7_position,
        orientation=radar7_orientation,
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    return [radar1, radar2, radar3, radar4, radar5, radar6, radar7]


@pytest.fixture(scope='session')
def mounting_offsets_2d():
    # Generate sensor mounting offsets for testing purposes
    return np.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [-1, 0],
                     [0, -1]])


@pytest.fixture(scope='session')
def mounting_offsets_3d():
    # Generate sensor mounting offsets for testing purposes
    return np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, 1],
                     [0, 0, -1]])


@pytest.fixture(params=[True, False], ids=["Moving", "Static"])
def move(request):
    return request.param


testdata_2d = [
    np.array([[0], [0], [0], [0]]),
    np.array([[10], [0], [0], [0]]),
    np.array([[0], [1], [0], [0]]),
    np.array([[0], [0], [0], [1]]),
    np.array([[0], [-1], [0], [0]]),
    np.array([[0], [0], [0], [-1]]),
    np.array([[0], [1], [0], [1]]),
    np.array([[0], [-1], [0], [-1]]),
    np.array([[0], [1], [0], [-1]]),
    np.array([[0], [-1], [0], [1]])
]

expected_2d = [
    # static platform or X velocity
    np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]),
    # static platform or X velocity
    np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]),
    # static platform or X velocity
    np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]),
    # y-axis motion
    np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]),
    # negative x-axis motion
    np.array([[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]]),
    # negative y-axis motion
    np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]),
    # x-y motion
    np.array([[0, 0], [1/np.sqrt(2), 1/np.sqrt(2)],
              [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]]),
    # neg x- neg y motion
    np.array([[0, 0], [-1/np.sqrt(2), -1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), 1/np.sqrt(2)],
              [-1/np.sqrt(2), 1/np.sqrt(2)]]),
    # pos x- neg y motion
    np.array([[0, 0], [1/np.sqrt(2), -1/np.sqrt(2)],
              [1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)],
              [-1/np.sqrt(2), -1/np.sqrt(2)]]),
    # neg x- pos y motion
    np.array([[0, 0], [-1/np.sqrt(2), 1/np.sqrt(2)],
              [-1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
              [1/np.sqrt(2), 1/np.sqrt(2)]])
]


@pytest.mark.parametrize(
    'state, expected', zip(testdata_2d, expected_2d),
    ids=["Static", "pos offset", "x vel", "y vel", "-x vel", "-y vel",
         "x,y vel", "-x,-y vel", "x,-y vel", "-x,y vel"])
def test_2d_platform(state, expected, move, radars_2d, mounting_offsets_2d):
    # Define time related variables
    timestamp = datetime.datetime.now()
    # Define transition model and position for platform
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    trans_model = CombinedLinearGaussianTransitionModel(
        [model_1d] * (radars_2d[0].ndim_state // 2))
    platform_state = State(state, timestamp)

    # This defines the mapping to the platforms state vector (i.e. x and y)
    mounting_mappings = np.array([[0, 2]])
    # create a platform with the simple radar mounted
    platform = SensorPlatform(
        state=platform_state,
        transition_model=trans_model,
        sensors=radars_2d,
        mounting_offsets=mounting_offsets_2d,
        mounting_mappings=mounting_mappings
    )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    sensor_positions_test(expected, platform)


testdata_3d = [
    (np.array([[0], [0], [0], [0], [0], [0]]), get_3d_expected(0)),
    (np.array([[10], [0], [0], [0], [0], [0]]), get_3d_expected(0)),
    (np.array([[0], [1], [0], [0], [0], [0]]), get_3d_expected(0)),
    (np.array([[0], [0], [0], [1], [0], [0]]), get_3d_expected(1)),
    (np.array([[0], [-1], [0], [0], [0], [0]]), get_3d_expected(2)),
    (np.array([[0], [0], [0], [-1], [0], [0]]), get_3d_expected(3)),
    (np.array([[0], [1], [0], [1], [0], [0]]), get_3d_expected(4)),
    (np.array([[0], [-1], [0], [-1], [0], [0]]), get_3d_expected(5)),
    (np.array([[0], [1], [0], [-1], [0], [0]]), get_3d_expected(6)),
    (np.array([[0], [-1], [0], [1], [0], [0]]), get_3d_expected(7)),
    (np.array([[0], [0], [0], [0], [0], [1]]), get_3d_expected(8)),
    (np.array([[0], [0], [0], [0], [0], [-1]]), get_3d_expected(9)),
    (np.array([[0], [0], [0], [1], [0], [1]]), get_3d_expected(10)),
    (np.array([[0], [0], [0], [1], [0], [-1]]), get_3d_expected(11)),
    (np.array([[0], [0], [0], [-1], [0], [1]]), get_3d_expected(12)),
    (np.array([[0], [0], [0], [-1], [0], [-1]]), get_3d_expected(13)),
    (np.array([[0], [1], [0], [0], [0], [1]]), get_3d_expected(14)),
    (np.array([[0], [-1], [0], [0], [0], [1]]), get_3d_expected(15)),
    (np.array([[0], [1], [0], [0], [0], [-1]]), get_3d_expected(16)),
    (np.array([[0], [-1], [0], [0], [0], [-1]]), get_3d_expected(17)),
    (np.array([[0], [1], [0], [1], [0], [1]]), get_3d_expected(18)),
    (np.array([[0], [-1], [0], [-1], [0], [-1]]), get_3d_expected(19))
]


@pytest.mark.parametrize('state, expected', testdata_3d, ids=[
    "Static", "pos offset", "x vel", "y vel", "-x vel", "-y vel", "x,y vel",
    "-x,-y vel", "x,-y vel", "-x,y vel", "z vel", "-z vel", "y.z vel",
    "y.-z vel", "-y.z vel", "-y.-z vel", "x.z vel", "-x.z vel", "x.-z vel",
    "-x,-z vel", "x,y,z vel", "-x,-y,-z vel"
])
def test_3d_platform(state, expected, move, radars_3d, mounting_offsets_3d):
    # Define time related variables
    timestamp = datetime.datetime.now()
    # Define transition model and position for platform
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    trans_model = CombinedLinearGaussianTransitionModel(
        [model_1d] * (radars_3d[0].ndim_state // 2))
    platform_state = State(state, timestamp)

    # This defines the mapping to the platforms state vector (i.e. x and y)
    mounting_mappings = np.array([[0, 2, 4]])
    # create a platform with the simple radar mounted
    platform = SensorPlatform(
        state=platform_state,
        transition_model=trans_model,
        sensors=radars_3d,
        mounting_offsets=mounting_offsets_3d,
        mounting_mappings=mounting_mappings
    )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    sensor_positions_test(expected, platform)


def sensor_positions_test(expected, platform):
    """
    This function asserts that the sensor positions on the platform have been
    correctly updated when the platform has been moved or sensor mounted on the
    platform.

    :param expected: nD array of expected sensor position post rotation
    :param platform: platform object
    :return:
    """
    expected_radar_position = np.zeros(
        [len(platform.sensors), platform.mounting_offsets.shape[1]])
    for i in range(len(platform.sensors)):
        radar_position = platform.sensors[i].position
        for j in range(platform.mounting_offsets.shape[1]):
            expected_radar_position[i, j] = (expected[i, j] +
                                             platform.state.state_vector[
                                                 platform.mounting_mappings[
                                                     i, j]])
        assert (np.allclose(expected_radar_position[i, j], radar_position[j]))
