# -*- coding: utf-8 -*-
import pytest
import numpy as np


@pytest.fixture()
def transition_model1():
    class TestTransitionModel:
        def function(self, state, time_interval):
            return state.state_vector + time_interval.total_seconds()
    return TestTransitionModel()


@pytest.fixture()
def transition_model2():
    class TestTransitionModel:
        def function(self, state, time_interval):
            return state.state_vector + 2*time_interval.total_seconds()
    return TestTransitionModel()


@pytest.fixture()
def measurement_model():
    class TestMeasurementModel:
        ndim_state = 4
        ndim_meas = 2

        @staticmethod
        def function(state):
            matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
            return matrix @ state.state_vector
    return TestMeasurementModel()
