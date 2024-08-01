from datetime import datetime, timedelta

import numpy as np
import pytest

from ..action.dwell_action import DwellActionsGenerator, ChangeDwellAction
from ...sensormanager.action import Actionable, ActionableProperty
from ..base import Property
from ...types.angle import Bearing
from ...types.array import StateVector


class DummyActionable(Actionable):
    dwell_centre: StateVector = ActionableProperty(
        doc="Actionable dwell centre.",
        generator_cls=DwellActionsGenerator,
        generator_kwargs_mapping={'rpm': 'rpm'})
    timestamp: datetime = Property(doc="Current time that actionable exists at.")
    rpm: float = Property(doc="Dwell centre revolutions per minute")

    def validate_timestamp(self):
        if self.timestamp:
            return True
        else:
            return False


def test_actionable():
    start = datetime.now()
    actionable = DummyActionable(StateVector([Bearing(np.radians(90)), 0, 0]),
                                 start,
                                 0.25)  # 1 revolution every 4 minutes (90 degrees in a minute)

    # Test initial schedule
    assert isinstance(actionable.scheduled_actions, dict)  # dictionary schedule is present
    assert not actionable.scheduled_actions  # no scheduled actions

    # Test actionable property/properties
    actionable_properties = actionable._actionable_properties
    assert isinstance(actionable_properties, dict)
    assert len(actionable_properties) == 1
    assert 'dwell_centre' in actionable_properties.keys()
    assert isinstance(actionable_properties['dwell_centre'], Property)

    end_time = start + timedelta(minutes=2)

    # Test default action(s)
    default_action = actionable._default_action('dwell_centre',
                                                actionable_properties['dwell_centre'],
                                                end_time)
    generator = actionable_properties['dwell_centre'].generator_cls(owner=actionable,
                                                                    attribute='dwell_centre',
                                                                    start_time=start,
                                                                    end_time=end_time)
    exp_default_action = generator.default_action

    # generators will be different (and not hashable), so not tested for equality
    assert default_action.end_time == exp_default_action.end_time == end_time
    assert default_action.increasing_angle is True
    assert exp_default_action.increasing_angle is True
    assert default_action.rotation_end_time == exp_default_action.rotation_end_time == end_time

    # Test actions
    generators = actionable.actions(end_time, start)
    assert len(generators) == 1
    generator = generators.pop()
    assert isinstance(generator, DwellActionsGenerator)
    assert generator.start_time == start
    assert generator.end_time == end_time
    assert generator.owner is actionable
    assert generator.attribute == 'dwell_centre'

    # Test add actions
    dwell_action = ChangeDwellAction(generator=generator,
                                     end_time=end_time,
                                     target_value=actionable.dwell_centre[0, 0] + np.radians(90),
                                     rotation_end_time=start + timedelta(minutes=1),
                                     increasing_angle=True)
    # action to rotate by 90 degrees anti-clockwise for 1 minute, then stay still for 1 minute
    actionable.add_actions([dwell_action])

    assert actionable.scheduled_actions
    assert actionable.scheduled_actions['dwell_centre'] is dwell_action

    actionable.act(start + timedelta(seconds=30))  # act for half the rotation time
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(135))

    actionable.act(start + timedelta(seconds=30))  # same timestamp (shouldn't change anything)
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(135))

    actionable.act(start + timedelta(seconds=60))  # act for rest of rotation time
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(180))

    actionable.act(start + timedelta(seconds=90))  # act for another 30s (shouldn't do anything)
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(180))

    actionable.act(start + timedelta(seconds=120))  # act for another 30s (shouldn't do anything)
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(180))

    assert not actionable.scheduled_actions  # action finished -> schedule empty

    actionable.act(start + timedelta(seconds=180))  # carry-out default action
    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(270))

    next_end_time = start + timedelta(seconds=240)
    generator = actionable.actions(next_end_time).pop()

    # Test actioning for longer than scheduled action
    dwell_action = ChangeDwellAction(generator=generator,
                                     end_time=next_end_time,
                                     target_value=actionable.dwell_centre[0, 0] + np.radians(45),
                                     rotation_end_time=start + timedelta(seconds=210),
                                     increasing_angle=True)
    # action to rotate by 45 degrees anti-clockwise for 1 minute
    actionable.add_actions([dwell_action])

    # do action for a minute, then rotate for 30 seconds:
    # 30s 45 degree rotation
    # 30s do nothing
    # 30s 45 degree rotation (from default action)
    actionable.act(next_end_time + timedelta(seconds=30))

    assert actionable.dwell_centre[0, 0] == Bearing(np.radians(360))

    # Test setting timestamp
    actionable.timestamp = None
    assert actionable.actions(start + timedelta(minutes=5))
    assert actionable.timestamp == start + timedelta(minutes=5)  # timestamp set

    # Test impossible actions
    bad_generator = actionable.actions(start + timedelta(minutes=1)).pop()
    bad_action = ChangeDwellAction(generator=bad_generator,
                                   end_time=next_end_time,
                                   target_value=actionable.dwell_centre[0, 0] + np.radians(45),
                                   rotation_end_time=start + timedelta(seconds=30),
                                   increasing_angle=True)
    with pytest.raises(ValueError,
                       match="Cannot schedule an action that ends before the current time."):
        actionable.add_actions([bad_action])

    next_next_end_time = start + timedelta(seconds=330)
    generator = actionable.actions(next_next_end_time).pop()
    action1 = ChangeDwellAction(generator=generator,
                                end_time=next_next_end_time,
                                target_value=actionable.dwell_centre[0, 0] + np.radians(450),
                                rotation_end_time=start + timedelta(seconds=300),
                                increasing_angle=True)
    action2 = ChangeDwellAction(generator=generator,
                                end_time=next_next_end_time,
                                target_value=actionable.dwell_centre[0, 0] + np.radians(450),
                                rotation_end_time=start + timedelta(seconds=310),
                                increasing_angle=True)
    with pytest.raises(ValueError,
                       match="Cannot schedule more actions than there are actionable properties."):
        actionable.add_actions([action1, action2])

    # Test invalid timestamps
    actionable = DummyActionable(StateVector([Bearing(np.radians(90)), 0, 0]),
                                 None,  # no timestamp
                                 0.25)

    actionable.timestamp = None
    assert actionable.add_actions({}) is None  # returns early
    assert actionable.act(start + timedelta(seconds=30)) is None  # returns early
