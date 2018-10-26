# coding: utf-8
# import pytest
import datetime
import numpy as np

from stonesoup.types.state import State
from stonesoup.platform import Platform
from stonesoup.models.transition.linear import ConstantVelocity,\
    CombinedLinearGaussianTransitionModel


def test_base():

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define transition model and position
    model_1d = ConstantVelocity(0.5)
    model_2d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d])

    # Define a new platform
    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1]]),
                           timestamp)
    platform = Platform(platform_state, model_2d)

    # Move the platform
    platform.move(new_timestamp)

    # TODO: More assertions
    assert(platform.state.timestamp == new_timestamp)
