# -*- coding: utf-8 -*-
import datetime
import inspect
from abc import ABC
from abc import abstractmethod
from typing import Set, Sequence

from .action import Action, ActionGenerator
from ..base import Base, Property


class ActionableProperty(Property):
    """Property that is modified via an :class:`~.Action` with defined, non-equal start and end
    times."""

    def __init__(self, generator_cls,
                 cls=None, *, default=inspect.Parameter.empty,
                 doc=None, readonly=False):
        super().__init__(cls=cls, default=default, doc=doc, readonly=readonly)
        self.generator_cls = generator_cls


class Actionable(Base, ABC):
    """Base Actionable type.

    Contains the core methods of an actionable sensor/platform type.

    Notes
    -----
    An Actionable is required to have a `timestamp` attribute, in order to validate actions and
    act. This is an abstract base class, and not intended for direct use. Attaining a timestamp is
    left to the inheriting type.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scheduled_actions = dict()  # dictionary of property - action pairs

    @abstractmethod
    def validate_timestamp(self):
        """Method of attaining actionable's timestamp.
        Returns
        -------
        bool
            True if timestamp has been successfully set, False otherwise.
        """
        raise NotImplementedError()

    @property
    def _actionable_properties(self):
        """Dictionary of all name - property pairs where the property is an
        :class:`~.ActionableProperty` (i.e. it is modified via action)."""
        return {_name: _property for _name, _property in type(self).properties.items()
                if isinstance(_property, ActionableProperty)}

    def _default_action(self, name, property_, timestamp):
        """Returns the default action of the action generator associated with the property
        (assumes the property is an :class:`~.ActionableProperty`."""
        generator = property_.generator_cls(owner=self,
                                            attribute=name,
                                            start_time=self.timestamp,
                                            end_time=timestamp)
        return generator.default_action

    def actions(self, timestamp: datetime.datetime, start_timestamp: datetime.datetime = None
                ) -> Set[ActionGenerator]:
        """Method to return a set of action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.ActionGenerator`
            Set of action generators, that describe the bounds of each action space.
        """

        if not self.validate_timestamp():
            self.timestamp = timestamp

        if start_timestamp is None:
            start_timestamp = self.timestamp

        generators = set()
        for name, property_ in self._actionable_properties.items():
            generators.add(property_.generator_cls(owner=self,
                                                   attribute=name,
                                                   start_time=start_timestamp,
                                                   end_time=timestamp))
        return generators

    def add_actions(self, actions: Sequence[Action]) -> bool:
        """Add actions to the sensor

        Parameters
        ----------
        actions: sequence of :class:`~.Action`
            Sequence of actions that will be executed in order

        Returns
        -------
        bool
            Return True if actions accepted. False if rejected.

        Raises
        ------
        NotImplementedError
            If sensor cannot be tasked.

        Notes
        -----
        Base class returns True
        """

        if not self.validate_timestamp():
            return

        if any(action.end_time < self.timestamp for action in actions):
            raise ValueError("Cannot schedule an action that ends before the current time.")

        if len(actions) > len(self._actionable_properties):
            raise ValueError("Cannot schedule more actions than there are actionable properties.")

        for name in self._actionable_properties:
            for action in actions:
                if action.generator.attribute == name:
                    self.scheduled_actions[name] = action
                    break
        return True

    def act(self, timestamp: datetime.datetime):
        """Carry out actions up to a timestamp.

        Parameters
        ----------
        timestamp: datetime.datetime
            Carry out actions up to this timestamp.
        """

        if not self.validate_timestamp():
            self.timestamp = timestamp
            return

        for name, property_ in self._actionable_properties.items():
            value = getattr(self, name)
            try:
                action = self.scheduled_actions[name]
            except KeyError:
                action = self._default_action(name, property_, timestamp)
                setattr(self, name, action.act(self.timestamp, timestamp, value))
            else:
                end_time = action.end_time
                if end_time < timestamp:
                    # complete action, remove from schedule
                    # switch to default, and carry-out default until timestamp
                    interim_value = action.act(self.timestamp, end_time, value)

                    # remove scheduled action
                    self.scheduled_actions.pop(name)

                    action = self._default_action(name, property_, timestamp)
                    setattr(self, name, action.act(end_time, timestamp, interim_value))
                elif end_time == timestamp:
                    # complete action and remove from schedule
                    setattr(self, name, action.act(self.timestamp, timestamp, value))
                    self.scheduled_actions.pop(name)
                else:
                    # carry-out action to timestamp
                    setattr(self, name, action.act(self.timestamp, timestamp, value))

        self.timestamp = timestamp
