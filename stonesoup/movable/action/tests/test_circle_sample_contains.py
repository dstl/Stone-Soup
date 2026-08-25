from datetime import datetime, timedelta

import numpy as np

from stonesoup.movable.action.move_position_action import (
    CircleSamplePositionActionGenerator,
    MovePositionAction,
)
from stonesoup.types.state import StateVector


class _Owner:
    def __init__(self, position):
        self.position = position


def test_circle_sample_contains_uses_travel_radius():
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)
    owner = _Owner(StateVector([1.0, 2.0]))
    generator = CircleSamplePositionActionGenerator(
        owner=owner,
        attribute="position",
        start_time=start_time,
        end_time=end_time,
        maximum_travel=2.0,
    )

    inside = StateVector([2.5, 2.0])
    outside = StateVector([3.1, 2.0])
    action = MovePositionAction(
        generator=generator,
        end_time=end_time,
        target_value=inside,
    )

    assert inside in generator
    assert action in generator
    assert outside not in generator
    assert np.isclose(np.linalg.norm(inside - owner.position), 1.5)
