# coding: utf-8
import datetime

import numpy as np

from ...types.state import State
from ...models.transition.linear import (
    ConstantVelocity, CombinedLinearGaussianTransitionModel)
from ..base import MovingPlatform, FixedPlatform


def test_base():
    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)

    # Define a static 2d platform and check it does not move
    platform_state2d = State(np.array([[2],
                                       [2]]),
                             timestamp)
    platform = FixedPlatform(state=platform_state2d, mapping=np.array([0, 1]))
    platform.move(new_timestamp)
    new_statevector = np.array([[2],
                                [2]])
    # Ensure 2d platform has not moved
    assert (np.array_equal(platform.state.state_vector, new_statevector))
    # Test to ensure platform time has updated
    assert (platform.state.timestamp == new_timestamp)

    # Define a static 3d platform and check it does not move
    platform_state3d = State(np.array([[2],
                                       [2],
                                       [2]]),
                             timestamp)
    platform = FixedPlatform(state=platform_state3d, mapping=[0, 1, 2])
    platform.move(new_timestamp)
    new_statevector = np.array([[2],
                                [2],
                                [2]])
    # Ensure 2d platform has not moved
    assert (np.array_equal(platform.state.state_vector, new_statevector))

    # Define zero noise 2d constant velocity transition model
    model_1d = ConstantVelocity(0.0)
    model_2d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d])

    # Define a 2d platform with constant velocity motion and test motion
    platform_state2d = State(np.array([[2],
                                       [1],
                                       [2],
                                       [1]]),
                             timestamp)
    platform = MovingPlatform(state=platform_state2d, transition_model=model_2d, mapping=[0, 2])
    platform.move(new_timestamp)

    # Define expected platform location after movement
    new_statevector = np.array([[4],
                                [1],
                                [4],
                                [1]])
    assert (np.array_equal(platform.state.state_vector, new_statevector))

    # Define zero noise 3d constant velocity transition model
    model_3d = CombinedLinearGaussianTransitionModel(
        [model_1d, model_1d, model_1d])

    # Define a 3d platform with constant velocity motion and test motion
    platform_state = State(np.array([[2],
                                     [1],
                                     [2],
                                     [1],
                                     [0],
                                     [1]]),
                           timestamp)
    platform = MovingPlatform(state=platform_state, transition_model=model_3d, mapping=[0, 2, 4])
    platform.move(new_timestamp)

    # Define expected platform location in 3d after movement
    new_statevector = np.array([[4],
                                [1],
                                [4],
                                [1],
                                [2],
                                [1]])
    assert (np.array_equal(platform.state.state_vector, new_statevector))

    # TODO: More assertions
