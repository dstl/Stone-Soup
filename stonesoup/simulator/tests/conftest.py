import pytest
import numpy as np

from ...types.detection import Detection
from ...sensor.sensor import Sensor


@pytest.fixture()
def transition_model1():
    class TestTransitionModel:
        def function(self, state, noise, time_interval):
            return state.state_vector + time_interval.total_seconds()
    return TestTransitionModel()


@pytest.fixture()
def transition_model2():
    class TestTransitionModel:
        def function(self, state, noise, time_interval):
            return state.state_vector + 2*time_interval.total_seconds()
    return TestTransitionModel()


@pytest.fixture()
def measurement_model():
    class TestMeasurementModel:
        ndim_state = 4
        ndim_meas = 2

        @staticmethod
        def function(state, noise):
            matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
            return matrix @ state.state_vector
    return TestMeasurementModel()


@pytest.fixture()
def sensor_model1():
    class TestSensor(Sensor):
        @property
        def measurement_model(self):
            raise NotImplementedError

        def measure(self, ground_truths):
            detections = set()
            for truth in ground_truths:
                detections.add(Detection(truth.state_vector[(0, 2), :], timestamp=truth.timestamp))
            return detections
    return TestSensor()


@pytest.fixture()
def sensor_model2():
    class TestSensor(Sensor):
        @property
        def measurement_model(self):
            raise NotImplementedError

        def measure(self, ground_truths):
            detections = set()
            for truth in ground_truths:
                detections.add(Detection(truth.state_vector, timestamp=truth.timestamp))
            return detections
    return TestSensor()
