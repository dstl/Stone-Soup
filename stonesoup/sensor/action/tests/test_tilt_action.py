import numpy as np
import pytest

from datetime import datetime, timedelta
from stonesoup.sensormanager.action import Actionable, ActionableProperty
from stonesoup.base import Property
from stonesoup.types.array import StateVector

from ..tilt_action import TiltActionsGenerator, ChangeTiltAction


class DummyTiltActionable(Actionable):
    tilt_centre: StateVector = ActionableProperty(doc="Actionable Tilt.",
                                                  generator_cls=TiltActionsGenerator,
                                                  generator_kwargs_mapping={'rpm': 'rpm'})
    timestamp: datetime = Property(doc="Current time that actionable exists at.")
    rpm: float = Property(doc="Rotations per minute.")
    min_tilt = np.radians(-90)
    max_tilt = np.radians(90)

    def validate_timestamp(self):
        if self.timestamp:
            return True
        else:
            return False


@pytest.mark.parametrize('initial_tilt_deg', np.arange(-90, 91, 10))
def test_tilt_action(initial_tilt_deg):
    initial_tilt = np.radians(initial_tilt_deg)

    start = datetime.now()
    end = start + timedelta(seconds=15)

    actionable = DummyTiltActionable(StateVector([initial_tilt]),
                                     start,
                                     2/3)  # 4 degrees per second

    generator = TiltActionsGenerator(actionable,
                                     attribute="tilt_centre",
                                     start_time=start,
                                     end_time=end,
                                     rpm=actionable.rpm)

    generator()
    assert generator.resolution == np.radians(1)
    assert generator.epsilon == np.radians(1e-6)
    generator(np.radians(15))
    assert generator.resolution == np.radians(15)
    assert generator.epsilon == np.radians(1e-6)
    generator(np.radians(30), np.radians(1e-7))
    assert generator.resolution == np.radians(30)
    assert generator.epsilon == np.radians(1e-7)

    assert generator.current_value == actionable.tilt_centre[0, 0]
    assert generator.initial_value == actionable.tilt_centre[0, 0]

    assert generator.duration == end-start
    assert generator.rpm == actionable.rpm
    assert generator.angle_delta == np.radians(60)
    assert generator.max == min(actionable.tilt_centre[0, 0] + np.radians(60), np.radians(90))
    assert generator.min == max(actionable.tilt_centre[0, 0] - np.radians(60), -np.radians(90))

    for angle in np.linspace(0, np.radians(60), 10):
        tilt1 = min(actionable.tilt_centre[0, 0] + angle, actionable.max_tilt)
        tilt2 = max(actionable.tilt_centre[0, 0] - angle, actionable.min_tilt)

        assert tilt1 in generator
        assert float(tilt1) in generator
        assert tilt2 in generator
        assert float(tilt2) in generator

        tilt_time = angle/np.radians(60) * (end-start)
        tilt_end = start + tilt_time

        action1 = ChangeTiltAction(generator=generator,
                                   end_time=end,
                                   target_value=tilt1,
                                   rotation_end_time=tilt_end,
                                   increasing_angle=True)
        assert action1 in generator

        action2 = ChangeTiltAction(generator=generator,
                                   end_time=end,
                                   target_value=tilt2,
                                   rotation_end_time=tilt_end,
                                   increasing_angle=False)
        assert action2 in generator

        tilt_end1, increasing1 = generator._end_time_direction(tilt1)
        tilt_end2, increasing2 = generator._end_time_direction(tilt2)

        if angle == 0:
            assert increasing1 is None
            assert increasing2 is None
            assert tilt_end1 == tilt_end2 == start
        elif actionable.tilt_centre[0, 0] == actionable.min_tilt:
            assert increasing1
            assert increasing2 is None
        elif actionable.tilt_centre[0, 0] == actionable.max_tilt:
            assert increasing1 is None
            assert not increasing2
        else:
            assert increasing1
            assert not increasing2

        if tilt1 == actionable.max_tilt:
            new_tilt_time = ((actionable.max_tilt - actionable.tilt_centre[0, 0]) /
                             np.radians(60) * (end-start))
            new_tilt_end = start + new_tilt_time
            assert tilt_end1 == new_tilt_end
        else:
            assert tilt_end1 == tilt_end

        if tilt2 == actionable.min_tilt:
            new_tilt_time = ((actionable.tilt_centre[0, 0] - actionable.min_tilt) /
                             np.radians(60) * (end-start))
            new_tilt_end = start + new_tilt_time
            assert tilt_end2 == new_tilt_end
        else:
            assert tilt_end2 == tilt_end

    actions = [action for action in generator]
    if (actionable.min_tilt + np.radians(60) <= actionable.tilt_centre[0, 0] <=
            actionable.max_tilt - np.radians(60)):
        assert len(actions) == 5
        target_tilts = np.array([-60, -30, 0, 30, 60])
    elif (actionable.min_tilt + np.radians(30) <= actionable.tilt_centre[0, 0] <=
            actionable.max_tilt - np.radians(30)):
        assert len(actions) == 4
        if actionable.tilt_centre[0, 0] < (actionable.max_tilt + actionable.min_tilt) / 2:
            target_tilts = np.array([-30, 0, 30, 60])
        else:
            target_tilts = np.array([-60, -30, 0, 30])
    else:
        assert len(actions) == 3
        if actionable.tilt_centre[0, 0] < (actionable.max_tilt + actionable.min_tilt) / 2:
            target_tilts = np.array([0, 30, 60])
        else:
            target_tilts = np.array([-60, -30, 0])

    target_tilts = np.radians(target_tilts) + actionable.tilt_centre[0, 0]
    target_tilts = target_tilts.tolist()

    for action, target_tilt in zip(actions, target_tilts):
        assert action.target_value == pytest.approx(target_tilt)

    generator(np.radians(1))
    for angle in np.arange(0, 181, 1):
        tilt1 = min(actionable.tilt_centre[0, 0] + np.radians(angle), actionable.max_tilt)
        tilt2 = max(actionable.tilt_centre[0, 0] - np.radians(angle), actionable.min_tilt)
        action1 = generator.action_from_value(tilt1)
        action2 = generator.action_from_value(tilt2)
        if (angle > 60 and (actionable.min_tilt + np.radians(60) + generator.epsilon <
                            actionable.tilt_centre[0, 0] <
                            actionable.max_tilt - np.radians(60) - generator.epsilon)):
            assert action1 is None
            assert action2 is None
        elif angle > 60 and (actionable.tilt_centre[0, 0] <=
                             actionable.min_tilt + np.radians(60) + generator.epsilon):
            assert action1 is None
            assert isinstance(action2, ChangeTiltAction)
            assert pytest.approx(action2.target_value) == tilt2
        elif angle > 60 and (actionable.tilt_centre[0, 0] >=
                             actionable.max_tilt - np.radians(60) - generator.epsilon):
            assert isinstance(action1, ChangeTiltAction)
            assert pytest.approx(action1.target_value) == tilt1
            assert action2 is None
        else:
            assert isinstance(action1, ChangeTiltAction)
            assert pytest.approx(action1.target_value) == tilt1
            assert isinstance(action2, ChangeTiltAction)
            assert pytest.approx(action2.target_value) == tilt2

            if np.isclose(angle, 0):
                assert action1 == action2
                assert hash(action1) == hash(action2)
            else:
                assert action1 != action2
                assert hash(action1) != hash(action2)

    with pytest.raises(ValueError, match="Can only generate action from an Angle/float/int type"):
        generator.action_from_value("hello")
