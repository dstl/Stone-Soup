# coding: utf-8
import copy
import datetime

import numpy as np
import pytest

from stonesoup.platform.simple import FixedSensorPlatform
from stonesoup.platform.tests import test_platform_base
from ...types.state import State
from ...platform.simple import MovingSensorPlatform
from ...models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
from ...sensor.radar.radar import RadarRangeBearing
from ...types.array import StateVector, CovarianceMatrix


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


@pytest.fixture(params=[MovingSensorPlatform, FixedSensorPlatform],
                ids=['MovingSensorPlatform', 'FixedSensorPlatform'])
def platform_type(request):
    return request.param


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


@pytest.fixture(scope='session')
def rotation_offsets_2d():
    # Generate sensor mounting offsets for testing purposes
    offsets = [[0, 0],
               [np.pi / 4, 0],
               [0, np.pi / 4],
               [-np.pi / 4, 0],
               [0, -np.pi / 4]]
    return [StateVector(offset) for offset in offsets]


@pytest.fixture(scope='session')
def rotation_offsets_3d():
    # Generate sensor rotation offsets for testing purposes
    offsets = [[0, 0, 0],
               [np.pi / 4, 0, 0],
               [0, np.pi / 4, 0],
               [-np.pi / 4, 0, 0],
               [0, -np.pi / 4, 0],
               [0, 0, np.pi / 4],
               [0, 0, -np.pi / 4]]
    return [StateVector(offset) for offset in offsets]


def expected_orientations_3d():
    pi = np.pi
    offset_3d_movement = np.arctan(1/np.sqrt(2))

    return [
        np.array([[0., 0., 0.], [pi/4, 0., 0.], [0., pi/4, 0.], [-pi/4, 0., 0.],
                  [0., -pi/4, 0.], [0., 0., pi/4], [0., 0., -pi/4]]),
        np.array([[0., 0., pi/2], [pi/4, 0., pi/2], [0., pi/4, pi/2], [-pi/4, 0., pi/2],
                  [0., -pi/4, pi/2], [0., 0., 3 * pi/4], [0., 0., pi/4]]),
        np.array([[0., pi/2, 0.],
                  [pi/4, pi/2, 0.], [0., 3 * pi/4, 0.], [-pi/4, pi/2, 0.],
                  [0., pi/4, 0.], [0., pi/2, pi/4], [0., pi/2, -pi/4]]),
        np.array([[0., 0., 0.], [pi/4, 0., 0.], [0., pi/4, 0.], [-pi/4, 0., 0.],
                  [0., -pi/4, 0.], [0., 0., pi/4], [0., 0., -pi/4]]),
        np.array([[0., 0., pi/2], [pi/4, 0., pi/2], [0., pi/4, pi/2], [-pi/4, 0., pi/2],
                  [0., -pi/4, pi/2], [0., 0., 3 * pi/4], [0., 0., pi/4]]),
        np.array([[0., pi/2, 0.], [pi/4, pi/2, 0.], [0., 3 * pi/4, 0.], [-pi/4, pi/2, 0.],
                  [0., pi/4, 0.], [0., pi/2, pi/4], [0., pi/2, -pi/4]]),
        np.array([[0., 0., pi/4], [pi/4, 0., pi/4], [0., pi/4, pi/4], [-pi/4, 0., pi/4],
                  [0., -pi/4, pi/4], [0., 0., pi/2], [0., 0., 0.]]),
        np.array([[0., pi/2, pi/4], [pi/4, pi/2, pi/4], [0., 3 * pi/4, pi/4], [-pi/4, pi/2, pi/4],
                  [0., pi/4, pi/4], [0., pi/2, pi/2], [0., pi/2, 0.]]),
        np.array([[0., pi/4, offset_3d_movement], [pi/4, pi/4, offset_3d_movement],
                  [0., pi/2, offset_3d_movement], [-pi/4, pi/4, offset_3d_movement],
                  [0., 0., offset_3d_movement], [0., pi/4, pi/4 + offset_3d_movement],
                  [0., pi/4, -pi/4 + offset_3d_movement]]),
        np.array([[0., pi, 0.], [pi/4, pi, 0.], [0., 5 * pi/4, 0.], [-pi/4, pi, 0.],
                  [0., 3 * pi/4, 0.], [0., pi, pi/4], [0., pi, -pi/4]]),
        np.array([[0., -pi/2, 0.], [pi/4, -pi/2, 0.], [0., -pi/4, 0.], [-pi/4, -pi/2, 0.],
                  [0., -3 * pi/4, 0.], [0., -pi/2, pi/4], [0., -pi/2, -pi/4]]),
        np.array([[0., 0., -pi/2], [pi/4, 0., -pi/2], [0., pi/4, -pi/2], [-pi/4, 0., -pi/2],
                  [0., -pi/4, -pi/2], [0., 0., -pi/4], [0., 0., -3 * pi/4]]),
        np.array([[0., pi, 0.], [pi/4, pi, 0.], [0., 5 * pi/4, 0.], [-pi/4, pi, 0.],
                  [0., 3 * pi/4, 0.], [0., pi, pi/4], [0., pi, -pi/4]]),
        np.array([[0., -pi/2, 0.], [pi/4, -pi/2, 0.], [0., -pi/4, 0.], [-pi/4, -pi/2, 0.],
                  [0., -3 * pi/4, 0.], [0., -pi/2, pi/4], [0., -pi/2, -pi/4]]),
        np.array([[0., 0., -pi/2], [pi/4, 0., -pi/2], [0., pi/4, -pi/2], [-pi/4, 0., -pi/2],
                  [0., -pi/4, -pi/2], [0., 0., -pi/4], [0., 0., -3 * pi/4]]),
        np.array([[0., pi, -pi/4], [pi/4, pi, -pi/4], [0., 5 * pi/4, -pi/4], [-pi/4, pi, -pi/4],
                  [0., 3 * pi/4, -pi/4], [0., pi, 0.], [0., pi, -pi/2]]),
        np.array([[0., -pi/2, -pi/4], [pi/4, -pi/2, -pi/4], [0., -pi/4, -pi/4],
                  [-pi/4, -pi/2, -pi/4], [0., -3 * pi/4, -pi/4], [0., -pi/2, 0.],
                  [0., -pi/2, -pi/2]]),
    ]


def expected_orientations_2d():
    pi = np.pi
    return [
        np.array([[0., 0.], [pi/4, 0.], [0., pi/4], [-pi/4, 0.], [0., -pi/4]]),
        np.array([[0., pi/2], [pi/4, pi/2], [0., 3 * pi/4], [-pi/4, pi/2],
                  [0., pi/4]]),
        np.array([[0., 0.], [pi/4, 0.], [0., pi/4], [-pi/4, 0.],
                  [0., -pi/4]]),
        np.array([[0., pi/2], [pi/4, pi/2], [0., 3 * pi/4], [-pi/4, pi/2],
                  [0., pi/4]]),
        np.array([[0., pi/4], [pi/4, pi/4], [0., pi/2], [-pi/4, pi/4],
                  [0., 0.]]),
        np.array([[0., pi], [pi/4, pi], [0., 5*pi/4], [-pi/4, pi],
                  [0., 3 * pi/4]]),
        np.array([[0., -pi/2], [pi/4, -pi/2], [0., -pi/4], [-pi/4, -pi/2],
                  [0., -3 * pi/4]]),
        np.array([[0., pi], [pi/4, pi], [0., 5 * pi/4], [-pi/4, pi],
                  [0., 3 * pi/4]]),
        np.array([[0., -pi/2], [pi/4, -pi/2], [0., -pi/4], [-pi/4, -pi/2],
                  [0., -3 * pi/4]]),
        np.array([[0., -3 * pi/4], [pi/4, -3 * pi/4], [0., -pi/2], [-pi/4, -3 * pi/4],
                  [0., -pi]])
    ]


@pytest.mark.parametrize('state, expected_platform_orientation, expected_sensor_orientations',
                         zip(*zip(*test_platform_base.orientation_tests_2d),
                             expected_orientations_2d()))
def test_rotation_offsets_2d(state, expected_platform_orientation, expected_sensor_orientations,
                             move, radars_2d, rotation_offsets_2d):
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
    platform = MovingSensorPlatform(
        state=platform_state,
        transition_model=trans_model,
        sensors=radars_2d,
        rotation_offsets=rotation_offsets_2d,
        mounting_mappings=mounting_mapping,
        mapping=mounting_mapping
    )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    assert np.allclose(platform.orientation, expected_platform_orientation)
    assert np.allclose(all_sensor_orientations(platform), expected_sensor_orientations)


@pytest.mark.parametrize('state, expected_platform_orientation, expected_sensor_orientations',
                         zip(*zip(*test_platform_base.orientation_tests_3d),
                             expected_orientations_3d()))
def test_rotation_offsets_3d(state, expected_platform_orientation, expected_sensor_orientations,
                             move, radars_3d, rotation_offsets_3d):
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
    platform = MovingSensorPlatform(
        state=platform_state,
        transition_model=trans_model,
        sensors=radars_3d,
        rotation_offsets=rotation_offsets_3d,
        mounting_mappings=mounting_mapping,
        mapping=mounting_mapping
    )
    if move:
        # Move the platform
        platform.move(timestamp + datetime.timedelta(seconds=2))
    assert np.allclose(platform.orientation, expected_platform_orientation)
    assert np.allclose(all_sensor_orientations(platform), expected_sensor_orientations)


def all_sensor_orientations(platform):
    radar_orientation = np.stack([sensor.orientation for sensor in platform.sensors], axis=1)
    return radar_orientation.T


def test_defaults(radars_3d, platform_type, add_sensor):
    platform_state = State(state_vector=StateVector([0, 1, 2, 1, 4, 1]),
                           timestamp=datetime.datetime.now())
    platform_args = {}
    if platform_type is MovingSensorPlatform:
        platform_args['transition_model'] = None

    if add_sensor:
        platform = platform_type(state=platform_state, sensors=[], mapping=[0, 2, 4],
                                 **platform_args)
        for sensor in radars_3d:
            platform.add_sensor(sensor)
    else:
        platform = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                                 **platform_args)

    for i, sensor in enumerate(radars_3d):
        assert np.array_equal(platform.mounting_mappings[i], np.array([0, 2, 4]))
        assert np.array_equal(platform.mounting_offsets[i], StateVector([0, 0, 0]))
        assert np.array_equal(platform.rotation_offsets[i], StateVector([0, 0, 0]))
        assert np.array_equal(sensor.position, platform.position)
        assert np.array_equal(sensor.orientation, platform.orientation)


def test_add_sensor_mapping_error(radars_3d, platform_type):
    platform_state = State(state_vector=StateVector([0, 1, 2, 1, 4, 1]),
                           timestamp=datetime.datetime.now())
    platform_args = {}
    if platform_type is MovingSensorPlatform:
        platform_args['transition_model'] = None

    mappings = [[0, 1, 2]] + [[0, 2, 4]] * (len(radars_3d) - 1)
    platform = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                             mounting_mappings=mappings, **platform_args)
    with pytest.raises(ValueError):
        platform.add_sensor(radars_3d[0])
    # no error if we specify mapping
    platform.add_sensor(radars_3d[0], mounting_mapping=[0, 1, 2])


@pytest.mark.parametrize('mapping', [[0, 2, 4],
                                     (0, 2, 4),
                                     np.array([0, 2, 4])])
def test_mounting_mapping_list(radars_3d, platform_type, mapping):
    platform_state = State(state_vector=StateVector([0, 1, 2, 1, 4, 1]),
                           timestamp=datetime.datetime.now())
    platform_args = {}
    if platform_type is MovingSensorPlatform:
        platform_args['transition_model'] = None

    mappings = [mapping] * len(radars_3d)
    platform = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                             mounting_mappings=mappings, **platform_args)

    for i, sensor in enumerate(radars_3d):
        assert np.array_equal(platform.mounting_mappings[i], np.array([0, 2, 4]))

    mappings = [mapping] * (len(radars_3d) - 1)
    with pytest.raises(ValueError):
        _ = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                          mounting_mappings=mappings, **platform_args)

    mappings = [copy.copy(mapping)] * len(radars_3d)
    try:
        mappings[0][2] = 6  # this value is out of bounds, and should cause an error
    except TypeError:
        # Tuple mapping is not assignable, but we can skip that one as the other two are good
        # enough tests
        return
    with pytest.raises(IndexError):
        _ = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                          mounting_mappings=mappings, **platform_args)


def test_sensor_offset_error(radars_3d, platform_type):
    platform_state = State(state_vector=StateVector([0, 1, 2, 1, 4, 1]),
                           timestamp=datetime.datetime.now())
    platform_args = {}
    if platform_type is MovingSensorPlatform:
        platform_args['transition_model'] = None

    offset = StateVector([0, 0, 0])

    offsets = [offset] * (len(radars_3d) - 1)
    with pytest.raises(ValueError):
        _ = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                          mounting_offsets=offsets, **platform_args)

    with pytest.raises(ValueError):
        _ = platform_type(state=platform_state, sensors=radars_3d, mapping=[0, 2, 4],
                          rotation_offsets=offsets, **platform_args)


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
        radar_position[i, :] = sensor.position.flat

        platform_position = platform.state_vector[platform.mounting_mappings[i]]

        expected_radar_position[i, :] = (expected_offset[i, :] +
                                         platform_position.flatten())

        assert np.allclose(radar_position[i, :], platform.get_sensor_position(sensor).flatten())
        assert np.allclose(platform_position, platform.position)
    assert np.allclose(expected_radar_position, radar_position)
