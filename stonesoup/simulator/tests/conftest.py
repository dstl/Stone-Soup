# -*- coding: utf-8 -*-

import datetime

import pytest
import numpy as np

from ...types.detection import Detection
from ...types.state import State


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
    class TestSensor():

        @staticmethod
        def measure(ground_truth):
            return Detection(ground_truth.state_vector[(0, 2), :],
                             timestamp=ground_truth.timestamp)
    return TestSensor()


@pytest.fixture()
def sensor_model2():
    class TestSensor:

        @staticmethod
        def measure(ground_truth):
            return Detection(ground_truth.state_vector,
                             timestamp=ground_truth.timestamp)
    return TestSensor()


@pytest.fixture()
def platform_class():
    class TestPlatform:

        def __init__(self, sensors, x_velocity):
            self.state = State([[0], [x_velocity], [0], [0]],
                               datetime.datetime(2020, 4, 1))
            self.sensors = sensors

        def move(self, timestamp):

            time_delta = timestamp - self.state.timestamp

            self.state.state_vector[0, 0] += \
                time_delta.total_seconds()*self.state.state_vector[1, 0]
            self.state.timestamp = timestamp

        @property
        def state_vector(self):
            return self.state.state_vector

        @property
        def timestamp(self):
            return self.state.timestamp
    return TestPlatform
