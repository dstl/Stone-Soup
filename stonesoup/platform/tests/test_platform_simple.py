# coding: utf-8
import datetime

import numpy as np
import pytest

from ...types.state import State
from ...platform.simple import MovingSensorPlatform
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
        return np.array([[0, 0, 0], [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, 1, 0], [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                         [0, -1, 0], [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [1/np.sqrt(2), 0, -1/np.sqrt(2)]])
    elif i == 15:
        # -x.z vel
        return np.array([[0, 0, 0], [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, -1, 0], [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                         [0, 1, 0], [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [-1/np.sqrt(2), 0, -1/np.sqrt(2)]])
    elif i == 16:
        # x.-z vel
        return np.array([[0, 0, 0], [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                         [0, 1, 0], [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, -1, 0], [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [-1/np.sqrt(2), 0, -1/np.sqrt(2)]])
    elif i == 17:
        # -x,-z vel
        return np.array([[0, 0, 0], [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                         [0, -1, 0], [1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, 1, 0], [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [1/np.sqrt(2), 0, -1/np.sqrt(2)]])
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

    measurement_mapping = np.array([0, 2])

    # Create 5 simple radar sensor objects
    radar1 = RadarRangeBearing(
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar,
    )

    radar2 = RadarRangeBearing(
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar3 = RadarRangeBearing(
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar4 = RadarRangeBearing(
        ndim_state=4,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar5 = RadarRangeBearing(
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

    measurement_mapping = np.array([0, 2, 4])

    # Create 5 simple radar sensor objects
    radar1 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar2 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar3 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )

    radar4 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar5 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar6 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    radar7 = RadarRangeBearing(
        ndim_state=6,
        mapping=measurement_mapping,
        noise_covar=noise_covar
    )
    return [radar1, radar2, radar3, radar4, radar5, radar6, radar7]


@pytest.fixture(scope='session')
def mounting_offsets_2d():
    # Generate sensor mounting offsets for testing purposes
    offsets = [[0, 0],
               [1, 0],
               [0, 1],
               [-1, 0],
               [0, -1]]
    return [StateVector(offset) for offset in offsets]


@pytest.fixture(scope='session')
def mounting_offsets_3d():
    # Generate sensor mounting offsets for testing purposes
    offsets = [[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [-1, 0, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]]
    return [StateVector(offset) for offset in offsets]


@pytest.fixture(params=[True, False], ids=["Moving", "Static"])
def move(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["Add", "Initialise"])
def add_sensor(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["MM_empty", "MM_added"])
def mounting_mapping_on_add(request):
    return request.param


testdata_2d = [
    StateVector([0, 0, 0, 0]),
    StateVector([10, 0, 0, 0]),
    StateVector([0, 1, 0, 0]),
    StateVector([0, 0, 0, 1]),
    StateVector([0, -1, 0, 0]),
    StateVector([0, 0, 0, -1]),
    StateVector([0, 1, 0, 1]),
    StateVector([0, -1, 0, -1]),
    StateVector([0, 1, 0, -1]),
    StateVector([0, -1, 0, 1])
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
def test_2d_platform(state, expected, move, radars_2d,
                     mounting_offsets_2d, add_sensor, mounting_mapping_on_add):
    # Define time related variables
    timestamp = datetime.datetime.now()
    # Define transition model and position for platform
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    trans_model = CombinedLinearGaussianTransitionModel(
        [model_1d] * (radars_2d[0].ndim_state // 2))
    platform_state = State(state, timestamp)

    # This defines the mapping to the platforms state vector (i.e. x and y)
    mounting_mapping = np.array([0, 2])
    # create a platform with the simple radar mounted
    if add_sensor:
        platform = MovingSensorPlatform(
            state=platform_state,
            transition_model=trans_model,
            sensors=[],
            mounting_offsets=[],
            mounting_mappings=mounting_mapping,
            mapping=mounting_mapping
        )
        for sensor, offset in zip(radars_2d, mounting_offsets_2d):
            if mounting_mapping_on_add:
                platform.add_sensor(sensor, offset,
                                    mounting_mapping=mounting_mapping)
            else:
                platform.add_sensor(sensor, offset)
    else:
        platform = MovingSensorPlatform(
            state=platform_state,
            transition_model=trans_model,
            sensors=radars_2d,
            mounting_offsets=mounting_offsets_2d,
            mounting_mappings=mounting_mapping,
            mapping=mounting_mapping
        )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    sensor_positions_test(expected, platform)


testdata_3d = [
    (StateVector([0, 0, 0, 0, 0, 0]), get_3d_expected(0)),
    (StateVector([10, 0, 0, 0, 0, 0]), get_3d_expected(0)),
    (StateVector([0, 1, 0, 0, 0, 0]), get_3d_expected(0)),
    (StateVector([0, 0, 0, 1, 0, 0]), get_3d_expected(1)),
    (StateVector([0, -1, 0, 0, 0, 0]), get_3d_expected(2)),
    (StateVector([0, 0, 0, -1, 0, 0]), get_3d_expected(3)),
    (StateVector([0, 1, 0, 1, 0, 0]), get_3d_expected(4)),
    (StateVector([0, -1, 0, -1, 0, 0]), get_3d_expected(5)),
    (StateVector([0, 1, 0, -1, 0, 0]), get_3d_expected(6)),
    (StateVector([0, -1, 0, 1, 0, 0]), get_3d_expected(7)),
    (StateVector([0, 0, 0, 0, 0, 1]), get_3d_expected(8)),
    (StateVector([0, 0, 0, 0, 0, -1]), get_3d_expected(9)),
    (StateVector([0, 0, 0, 1, 0, 1]), get_3d_expected(10)),
    (StateVector([0, 0, 0, 1, 0, -1]), get_3d_expected(11)),
    (StateVector([0, 0, 0, -1, 0, 1]), get_3d_expected(12)),
    (StateVector([0, 0, 0, -1, 0, -1]), get_3d_expected(13)),
    (StateVector([0, 1, 0, 0, 0, 1]), get_3d_expected(14)),
    (StateVector([0, -1, 0, 0, 0, 1]), get_3d_expected(15)),
    (StateVector([0, 1, 0, 0, 0, -1]), get_3d_expected(16)),
    (StateVector([0, -1, 0, 0, 0, -1]), get_3d_expected(17)),
    (StateVector([0, 1, 0, 1, 0, 1]), get_3d_expected(18)),
    (StateVector([0, -1, 0, -1, 0, -1]), get_3d_expected(19))
]


@pytest.mark.parametrize('state, expected', testdata_3d, ids=[
    "Static", "pos offset", "x vel", "y vel", "-x vel", "-y vel", "x,y vel",
    "-x,-y vel", "x,-y vel", "-x,y vel", "z vel", "-z vel", "y.z vel",
    "y.-z vel", "-y.z vel", "-y.-z vel", "x.z vel", "-x.z vel", "x.-z vel",
    "-x,-z vel", "x,y,z vel", "-x,-y,-z vel"
])
def test_3d_platform(state, expected, move, radars_3d, mounting_offsets_3d,
                     add_sensor, mounting_mapping_on_add):
    # Define time related variables
    timestamp = datetime.datetime.now()
    # Define transition model and position for platform
    model_1d = ConstantVelocity(0.0)  # zero noise so pure movement
    trans_model = CombinedLinearGaussianTransitionModel(
        [model_1d] * (radars_3d[0].ndim_state // 2))
    platform_state = State(state, timestamp)

    # This defines the mapping to the platforms state vector (i.e. x and y)
    mounting_mapping = np.array([0, 2, 4])
    # create a platform with the simple radar mounted
    if add_sensor:
        platform = MovingSensorPlatform(
            state=platform_state,
            transition_model=trans_model,
            sensors=[],
            mounting_offsets=[],
            mounting_mappings=mounting_mapping,
            mapping=mounting_mapping
        )
        for sensor, offset in zip(radars_3d, mounting_offsets_3d):
            if mounting_mapping_on_add:
                platform.add_sensor(sensor, offset,
                                    mounting_mapping=mounting_mapping)
            else:
                platform.add_sensor(sensor, offset)
    else:
        platform = MovingSensorPlatform(
            state=platform_state,
            transition_model=trans_model,
            sensors=radars_3d,
            mounting_offsets=mounting_offsets_3d,
            mounting_mappings=mounting_mapping,
            mapping=mounting_mapping
        )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    sensor_positions_test(expected, platform)


def sensor_positions_test(expected_offset, platform):
    """
    This function asserts that the sensor positions on the platform have been
    correctly updated when the platform has been moved or sensor mounted on the
    platform.

    :param expected_offset: nD array of expected sensor position post rotation
    :param platform: platform object
    :return:
    """
    radar_position = np.zeros(
        [len(platform.sensors), len(platform.mapping)])
    expected_radar_position = np.zeros_like(radar_position)
    for i, sensor in enumerate(platform.sensors):
        radar_position[i, :] = sensor.position.flatten()

        platform_position = platform.state_vector[platform.mounting_mappings[i]]

        expected_radar_position[i, :] = (expected_offset[i, :] +
                                         platform_position.flatten())

        assert np.allclose(radar_position[i, :], platform.get_sensor_position(sensor).flatten())
        assert np.allclose(platform_position, platform.position)
    assert np.allclose(expected_radar_position, radar_position)
