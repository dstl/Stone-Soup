from datetime import datetime, timedelta

import numpy as np
import pytest

from ..dwell_action import DwellActionsGenerator, ChangeDwellAction
from ...actionable import Actionable, ActionableProperty
from ...base import Property
from ....types.angle import Bearing, Angle
from ....types.array import StateVector


class DummyActionable(Actionable):
    dwell_centre: StateVector = ActionableProperty(doc="Actionable dwell centre.",
                                                   generator_cls=DwellActionsGenerator)
    timestamp: datetime = Property(doc="Current time that actionable exists at.")
    rpm: float = Property(doc="Dwell centre revolutions per minute")

    def validate_timestamp(self):
        if self.timestamp:
            return True
        else:
            return False


@pytest.mark.parametrize('initial_bearing', (0, 45, 90, 135, 180, -45, -90, -135))
def test_dwell_action(initial_bearing):

    initial_bearing = Bearing(np.radians(initial_bearing))

    start = datetime.now()
    end = start + timedelta(seconds=15)

    actionable = DummyActionable(StateVector([initial_bearing, 0, 0]),
                                 start,
                                 1)  # 1 revolution per minute

    generator = DwellActionsGenerator(actionable,
                                      "dwell_centre",
                                      start,
                                      end)  # 15s maximum action duration

    # Test call and resolution
    generator()
    assert generator.resolution == np.radians(1)  # default resolution is 1 degree
    assert generator.epsilon == np.radians(1e-6)
    generator(np.radians(15))
    assert generator.resolution == np.radians(15)  # calling with arg sets resolution to arg
    assert generator.epsilon == np.radians(1e-6)  # tolerance does not change
    generator(np.radians(30), np.radians(1e-7))
    assert generator.resolution == np.radians(30)
    assert generator.epsilon == np.radians(1e-7)

    # Test value
    assert np.array_equal(generator.current_value, actionable.dwell_centre)
    assert generator.initial_value == actionable.dwell_centre[0, 0]

    # Test duration
    assert generator.duration == end - start

    # Test rps
    assert generator.rps == actionable.rpm / 60

    # Test angles
    assert pytest.approx(generator.angle_delta, np.radians(90))
    assert pytest.approx(generator.min, actionable.dwell_centre[0, 0] - np.radians(90))
    assert pytest.approx(generator.max, actionable.dwell_centre[0, 0] + np.radians(90))

    # Test contains and end-time/direction
    for angle in np.linspace(0, 90, 10):

        angle1 = Angle(actionable.dwell_centre[0, 0]) + np.radians(angle)
        angle2 = Angle(actionable.dwell_centre[0, 0]) - np.radians(angle)

        # any bearing in [dwell - 90, dwell + 90] should be achievable
        assert angle1 in generator
        assert float(angle1) in generator
        assert angle2 in generator
        assert float(angle2) in generator

        rotation_time = np.radians(angle)/np.radians(90) * (end - start)
        rot_end = start + rotation_time

        action1 = ChangeDwellAction(generator=generator,
                                    end_time=end,
                                    target_value=angle1,
                                    rotation_end_time=rot_end,
                                    increasing_angle=True)
        assert action1 in generator
        action2 = ChangeDwellAction(generator=generator,
                                    end_time=end,
                                    target_value=angle2,
                                    rotation_end_time=rot_end,
                                    increasing_angle=False)
        assert action2 in generator

        rot_end1, increasing1 = generator._end_time_direction(angle1)
        rot_end2, increasing2 = generator._end_time_direction(angle2)

        if angle == 0:
            assert increasing1 is None
            assert increasing2 is None
            assert rot_end1 == rot_end2 == start
        else:
            assert increasing1
            assert not increasing2
        assert pytest.approx(rot_end1, rot_end)
        assert pytest.approx(rot_end2, rot_end)

    # Test iterable
    actions = [action for action in generator]
    assert len(actions) == 7
    target_bearings = np.array([-90, -60, -30, 0, 30, 60, 90])
    target_bearings = np.radians(target_bearings) + actionable.dwell_centre[0, 0]
    target_bearings = target_bearings.tolist()
    for action, target_bearing in zip(actions, target_bearings):
        # actions in value order and all values accounted for
        assert pytest.approx(action.target_value, target_bearing)

    # Test action from value
    for angle in np.linspace(0, 180, 10):
        angle1 = Angle(actionable.dwell_centre[0, 0]) + np.radians(angle)
        angle2 = Angle(actionable.dwell_centre[0, 0]) - np.radians(angle)

        action1 = generator.action_from_value(angle1)
        action2 = generator.action_from_value(angle2)

        if angle > 90:
            assert action1 is None
            assert action2 is None
        else:
            assert isinstance(action1, ChangeDwellAction)
            assert pytest.approx(action1.target_value, angle1)
            assert isinstance(action2, ChangeDwellAction)
            assert pytest.approx(action2.target_value, angle2)

    # Test action from invalid
    with pytest.raises(ValueError, match="Can only generate action from an Angle/float/int type"):
        generator.action_from_value('hello')
