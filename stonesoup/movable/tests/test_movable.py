import pytest

from stonesoup.movable import MovingMovable, FixedMovable
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
