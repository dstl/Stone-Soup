from datetime import datetime, timedelta

import numpy as np
import pytest

from stonesoup.models.transition.linear import ConstantVelocity, ConstantTurn, \
    CombinedLinearGaussianTransitionModel
from stonesoup.movable import MovingMovable, FixedMovable, MultiTransitionMovable
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State, GaussianState


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
    models = [ConstantVelocity(0), ConstantTurn(0, np.pi / 2)]
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


def test_preserved_state():
    state_vector = np.zeros(4)
    start = datetime.now()
    states = [
        State(state_vector=state_vector, timestamp=start),
        GaussianState(state_vector=state_vector, timestamp=start, covar=np.eye(4)),
        GroundTruthState(state_vector=state_vector, timestamp=start,
                         metadata={"colour": "blue"})
    ]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1),
                                                              ConstantVelocity(0.1)])

    for state in states:
        movables = [
            FixedMovable(states=[state], position_mapping=(0, 2)),
            MovingMovable(states=[state], position_mapping=(0, 2),
                          transition_model=transition_model),
            MultiTransitionMovable(states=[state], position_mapping=(0, 2),
                                   transition_models=[transition_model],
                                   transition_times=[timedelta(seconds=10)])
        ]

        for movable in movables:
            movable.move(state.timestamp + timedelta(seconds=5))
            assert len(movable) == 2
            new_state = movable.state
            assert new_state != state
            assert isinstance(new_state, type(state))
            assert new_state.timestamp == state.timestamp + timedelta(seconds=5)
            if isinstance(state, GaussianState):
                new_state.covar  # covar exists in new state
            elif isinstance(state, GroundTruthState):
                assert new_state.metadata == state.metadata  # metadata carried-over to new state
