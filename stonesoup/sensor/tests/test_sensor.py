import copy
import datetime

import pytest

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange, \
    CartesianToElevationBearingRange
from stonesoup.movable import FixedMovable
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate, RadarBearingRange, \
    RadarElevationBearingRange
from stonesoup.platform import FixedPlatform
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from ..base import PlatformMountable
from ..sensor import Sensor, SensorSuite
from ...types.array import StateVector, CovarianceMatrix
from ...types.state import State

import numpy as np


class DummySensor(Sensor):
    def measure(self, **kwargs):
        pass


class DummyBaseSensor(PlatformMountable):
    def measure(self, **kwargs):
        pass


def test_sensor_position_orientation_setting():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))
    sensor.position = StateVector([0, 1, 0])
    assert np.array_equal(sensor.position, StateVector([0, 1, 0]))
    sensor.orientation = StateVector([0, 1, 0])
    assert np.array_equal(sensor.orientation, StateVector([0, 1, 0]))

    position = StateVector([0, 0, 1])
    sensor = DummySensor()
    platform_state = State(state_vector=position + 1, timestamp=datetime.datetime.now())
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    platform.add_sensor(sensor)
    with pytest.raises(AttributeError):
        sensor.position = StateVector([0, 1, 0])
    with pytest.raises(AttributeError):
        sensor.orientation = StateVector([0, 1, 0])


def test_default_platform():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 0]))

    sensor = DummySensor(orientation=StateVector([0, 0, 1]))
    assert np.array_equal(sensor.orientation, StateVector([0, 0, 1]))
    assert np.array_equal(sensor.position, StateVector([0, 0, 0]))


def test_internal_platform_flag():
    sensor = DummySensor(position=StateVector([0, 0, 1]))
    assert sensor._has_internal_controller

    sensor = DummySensor()
    assert not sensor._has_internal_controller

    sensor = DummyBaseSensor()
    assert not sensor._has_internal_controller


def test_changing_platform_from_default():
    position = StateVector([0, 0, 1])
    sensor = DummySensor(position=StateVector([0, 0, 1]))

    platform_state = State(state_vector=position + 1, timestamp=datetime.datetime.now())
    platform = FixedPlatform(states=platform_state, position_mapping=[0, 1, 2])
    with pytest.raises(AttributeError):
        platform.add_sensor(sensor)


@pytest.mark.parametrize('sensor', [DummyBaseSensor, DummySensor])
def test_sensor_measure(sensor):
    # needed for test coverage... Does no harm
    assert sensor().measure() is None


@pytest.fixture
def radar_platform_target():
    pos_mapping = np.array([0, 2, 4])
    vel_mapping = np.array([1, 3, 5])
    noise_covar = CovarianceMatrix(np.eye(4))

    radar = RadarElevationBearingRangeRate(ndim_state=6,
                                           position_mapping=pos_mapping,
                                           velocity_mapping=vel_mapping,
                                           noise_covar=noise_covar)

    # create a platform with the simple radar mounted
    timestamp_init = datetime.datetime.now()
    platform_1_prior_state_vector = StateVector([[5], [0], [0], [0.25], [0], [0]])
    platform_1_state = State(platform_1_prior_state_vector, timestamp_init)

    radar.mounting_offset = StateVector([[0], [0], [0]])
    radar.rotation_offset = StateVector([[0], [0], [0]])

    platform = FixedPlatform(states=platform_1_state,
                             position_mapping=pos_mapping,
                             sensors=[radar])
    target = State(StateVector([[0], [0], [0], [0], [0], [0]]), timestamp_init)

    return radar, platform, target


def test_platform_deepcopy(radar_platform_target):
    # Set up Radar Model
    radar_1, platform_1, target_state = radar_platform_target

    # Make a measurement with a platform
    _ = platform_1.sensors[0].measure({target_state})

    assert platform_1.sensors[0] is radar_1

    # 1st Error - Try and create another platform and make a measurement
    platform_2 = copy.deepcopy(platform_1)

    assert platform_2 is not platform_1
    assert platform_2.sensors[0] is not platform_1.sensors[0]

    assert platform_2.sensors[0].movement_controller is platform_2.movement_controller
    # check no error in measurement
    _ = platform_2.sensors[0].measure({target_state})


def test_sensor_assignment(radar_platform_target):
    radar_1, platform, target_state = radar_platform_target

    # platform_2 = copy.deepcopy(platform)
    radar_2 = copy.deepcopy(radar_1)
    radar_3 = copy.deepcopy(radar_1)

    platform.add_sensor(radar_2)
    platform.add_sensor(radar_3)

    assert len(platform.sensors) == 3
    assert platform.sensors == (radar_1, radar_2, radar_3)

    with pytest.raises(AttributeError):
        platform.sensors = []

    with pytest.raises(TypeError):
        platform.sensors[0] = radar_3

    platform.pop_sensor(1)
    assert len(platform.sensors) == 2
    assert platform.sensors == (radar_1, radar_3)

    platform.add_sensor(radar_2)
    assert len(platform.sensors) == 3
    assert platform.sensors == (radar_1, radar_3, radar_2)

    platform.remove_sensor(radar_3)
    assert len(platform.sensors) == 2
    assert platform.sensors == (radar_1, radar_2)


def test_informative():
    dummy_truth1 = GroundTruthPath([GroundTruthState([1, 1, 1, 1]),
                                    GroundTruthState([2, 1, 2, 1]),
                                    GroundTruthState([3, 1, 3, 1])])
    dummy_truth2 = GroundTruthPath([GroundTruthState([1, 0, -1, -1]),
                                    GroundTruthState([1, 0, -2, -1]),
                                    GroundTruthState([1, 0, -3, -1])])

    position = State(StateVector([0, 0, 0, 0]))
    orientation = StateVector([np.radians(90), 0, 0])
    test_radar = RadarBearingRange(ndim_state=4,
                                   position_mapping=(0, 2),
                                   noise_covar=np.diag([np.radians(1) ** 2, 5 ** 2]),
                                   movement_controller=FixedMovable(states=[position],
                                                                    orientation=orientation,
                                                                    position_mapping=(0, 2))
                                   )
    test_sensor = SensorSuite(attributes_inform=['position', 'orientation'],
                              sensors=[test_radar])

    detections = test_sensor.measure({dummy_truth1, dummy_truth2})

    assert len(detections) == 2

    for detection in detections:
        assert isinstance(detection.measurement_model, CartesianToBearingRange)
        metadata_position = detection.metadata.get('position')
        assert np.allclose(metadata_position, position.state_vector[(0, 2), :])
        metadata_orientation = detection.metadata.get('orientation')
        assert np.allclose(metadata_orientation, orientation)

    position2 = State(StateVector([50, 0, 50, 0]))
    orientation2 = StateVector([np.radians(-90), np.radians(10), 0])
    test_radar2 = RadarElevationBearingRange(ndim_state=4,
                                             position_mapping=(0, 2, 3),
                                             noise_covar=np.diag(
                                                 [np.radians(10) ** 2,
                                                  np.radians(2) ** 2,
                                                  0.1 ** 2]
                                             ),
                                             movement_controller=FixedMovable(
                                                 states=[position2],
                                                 orientation=orientation2,
                                                 position_mapping=(0, 2, 3)
                                             )
                                             )

    test_sensor = SensorSuite(attributes_inform=['position', 'orientation'],
                              sensors=[test_radar, test_radar2])

    detections = test_sensor.measure({dummy_truth1, dummy_truth2})

    assert len(detections) == 4

    for detection in detections:
        assert isinstance(detection.measurement_model, CartesianToBearingRange) \
               or isinstance(detection.measurement_model, CartesianToElevationBearingRange)
        metadata_position = detection.metadata.get('position')
        metadata_orientation = detection.metadata.get('orientation')
        if isinstance(detection.measurement_model, CartesianToBearingRange):
            assert np.allclose(metadata_position, position.state_vector[(0, 2), :])
            assert np.allclose(metadata_orientation, orientation)
        else:
            assert np.allclose(metadata_position, position2.state_vector[(0, 2, 3), :])
            assert np.allclose(metadata_orientation, orientation2)
