from datetime import datetime, timedelta

import pytest
import numpy as np

from stonesoup.models.transition.linear import ConstantVelocity, ConstantTurn, \
    CombinedLinearGaussianTransitionModel
from stonesoup.movable import MovingMovable, FixedMovable, MultiTransitionMovable
from stonesoup.types.array import StateVector
from stonesoup.types.state import State


def test_fixed_movable_velocity_mapping_error():
    # test no error for MovingMovable
    _ = MovingMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                      position_mapping=[0, 2, 4],
                      velocity_mapping=[1, 2, 5],
                      transition_model=None,
                      )
    with pytest.raises(ValueError, match='Velocity mapping should not be set for a FixedMovable'):
        _ = FixedMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                         position_mapping=[0, 2, 4],
                         velocity_mapping=[1, 2, 5],
                         )


def test_empty_state_error():
    # first, check no error
    _ = MovingMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                      position_mapping=[0, 2, 4],
                      velocity_mapping=[1, 2, 5],
                      transition_model=None,
                      )
    with pytest.raises(ValueError, match='States must not be empty'):
        _ = MovingMovable(position_mapping=[0, 2, 4],
                          velocity_mapping=[1, 2, 5],
                          transition_model=None,
                          )
    with pytest.raises(ValueError, match='States must not be empty'):
        _ = MovingMovable(states=[],
                          position_mapping=[0, 2, 4],
                          velocity_mapping=[1, 2, 5],
                          transition_model=None,
                          )


def test_multi_transition_movable_errors():
    # First check no error
    models = [ConstantVelocity(0), ConstantTurn(0, np.pi/2)]
    now = datetime.now()
    times = [timedelta(seconds=10), timedelta(seconds=10)]
    _ = MultiTransitionMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                               position_mapping=[0, 2, 4],
                               velocity_mapping=[1, 2, 5],
                               transition_models=models,
                               transition_times=times,
                               )

    with pytest.raises(AttributeError,
                       match='transition_models and transition_times must be same length'):
        _ = MultiTransitionMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                                   position_mapping=[0, 2, 4],
                                   velocity_mapping=[1, 2, 5],
                                   transition_models=[models[0]],
                                   transition_times=times,
                                   )

    with pytest.raises(AttributeError,
                       match='transition_models and transition_times must be same length'):
        _ = MultiTransitionMovable(states=State(StateVector([0, 0, 0, 0, 0, 0])),
                                   position_mapping=[0, 2, 4],
                                   velocity_mapping=[1, 2, 5],
                                   transition_models=models,
                                   transition_times=[now],
                                   )


def test_multi_transition_movable_move():
    input_state_vector = StateVector([0, 1, 2.2, 78.6])
    pre_state = State(input_state_vector, timestamp=None)
    models = [CombinedLinearGaussianTransitionModel((ConstantVelocity(0), ConstantVelocity(0))),
              ConstantTurn([0, 0], turn_rate=np.pi / 2)]
    times = [timedelta(seconds=10), timedelta(seconds=10)]

    movable = MultiTransitionMovable(states=pre_state,
                                     position_mapping=[0, 2],
                                     velocity_mapping=[1, 3],
                                     transition_models=models,
                                     transition_times=times,
                                     )

    assert movable.state.state_vector is input_state_vector
    assert movable.state.timestamp is None

    now = datetime.now()
    movable.move(now)
    assert movable.state.state_vector is input_state_vector
    assert movable.state.timestamp is now

    movable.move(None)
    assert movable.state.state_vector is input_state_vector
    assert movable.state.timestamp is now

    movable.move(now + timedelta(seconds=10))
    assert movable.state.state_vector is not input_state_vector
    assert movable.state.timestamp is not now
