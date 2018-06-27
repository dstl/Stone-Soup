# -*- coding: utf-8 -*-
import pytest


@pytest.fixture()
def transition_model():
    class TestTransitionModel:
        def function(self, state_vector, time_interval):
            return state_vector + time_interval.total_seconds()
    return TestTransitionModel()
