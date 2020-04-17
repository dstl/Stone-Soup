# -*- coding: utf-8 -*-

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

        def measure(self, ground_truth):
            return Detection(ground_truth.state_vector[(0, 2), :],
                             timestamp=ground_truth.timestamp)
    return TestSensor()


@pytest.fixture()
def sensor_model2():
    class TestSensor(Sensor):

        def measure(self, ground_truth):
            return Detection(ground_truth.state_vector,
                             timestamp=ground_truth.timestamp)
    return TestSensor()
