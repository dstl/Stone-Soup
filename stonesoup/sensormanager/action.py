import datetime
import inspect
from abc import ABC, abstractmethod
from typing import Set, Sequence, Iterator, Any

from stonesoup.base import Base, Property


class Action(Base):
    """The base class for an action that can be taken by a sensor or platform with an
    :class:`~.ActionableProperty`."""

    generator: Any = Property(default=None,
                              readonly=True,
                              doc="Action generator that created the action.")
    end_time: datetime.datetime = Property(readonly=True,
                                           doc="Time at which modification of the "
                                               "attribute ends.")
    target_value: Any = Property(doc="Target value.")

    def act(self, current_time, timestamp, init_value, **kwargs):
        """Return the attribute modified.

        Parameters
        ----------
        current_time: datetime.datetime
            Current time
        timestamp: datetime.datetime
            Modification of attribute ends at this time stamp
        init_value: Any
            Current value of the modifiable attribute

        Returns
        -------
        Any
            The new value of the attribute
        """
        raise NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return all(getattr(self, name) == getattr(other, name) for name in type(self).properties)

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in type(self).properties))


class ActionGenerator(Base):
    """The base class for an action generator."""

    owner: object = Property(doc="Actionable object that has the attribute to be modified.")
    attribute: str = Property(doc="The name of the attribute to be modified.")
    start_time: datetime.datetime = Property(doc="Start time of action.")
    end_time: datetime.datetime = Property(doc="End time of action.")
    resolution: float = Property(default=None, doc="Resolution of action space")

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[Action]:
        raise NotImplementedError()

    @property
    def current_value(self):
        """Return the current value of the owner's attribute."""
        return getattr(self.owner, self.attribute)

    @property
    def default_action(self):
        """The default action to modify the property if there is no given action."""
        raise NotImplementedError()


class RealNumberActionGenerator(ActionGenerator):
    """Action generator where action is a choice of a real number."""

    @property
    @abstractmethod
    def initial_value(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def min(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max(self):
        raise NotImplementedError


class ActionableProperty(Property):
    """Property that is modified via an :class:`~.Action` with defined, non-equal start and end
    times."""

    def __init__(self, generator_cls, generator_kwargs_mapping=None,
                 cls=None, *, default=inspect.Parameter.empty,
                 doc=None, readonly=False):
        super().__init__(cls=cls, default=default, doc=doc, readonly=readonly)
        self.generator_cls = generator_cls
        self.generator_kwargs_mapping = generator_kwargs_mapping
        if generator_kwargs_mapping is None:
            self.generator_kwargs_mapping = dict()


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

        self._generator_kwargs = dict()
        self.scheduled_actions = dict()  # dictionary of property - action pairs

    @property
    def _actionable_properties(self):
        """Dictionary of all name - property pairs where the property is an
        :class:`~.ActionableProperty` (i.e. it is modified via action)."""
        return {_name: _property for _name, _property in type(self).properties.items()
                if isinstance(_property, ActionableProperty)}

    def _default_action(self, name, property_, timestamp):
        """Returns the default action of the action generator associated with the property
        (assumes the property is an :class:`~.ActionableProperty`."""

        generator = property_.generator_cls(
            owner=self,
            attribute=name,
            start_time=self.timestamp,
            end_time=timestamp,
            **{key: getattr(self, value)
               for key, value in property_.generator_kwargs_mapping.items()})

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
            generators.add(property_.generator_cls(
                owner=self,
                attribute=name,
                start_time=start_timestamp,
                end_time=timestamp,
                **{key: getattr(self, value)
                   for key, value in property_.generator_kwargs_mapping.items()}))
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
            Returns neither if timestamp is invalid.

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

    def act(self, timestamp: datetime.datetime, **kwargs):
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
                setattr(self, name, action.act(self.timestamp, timestamp, value, **kwargs))
            else:
                end_time = action.end_time
                if end_time < timestamp:
                    # complete action, remove from schedule
                    # switch to default, and carry-out default until timestamp
                    interim_value = action.act(self.timestamp, end_time, value, **kwargs)

                    # remove scheduled action
                    self.scheduled_actions.pop(name)

                    action = self._default_action(name, property_, timestamp)
                    setattr(self, name, action.act(end_time, timestamp, interim_value, **kwargs))
                elif end_time == timestamp:
                    # complete action and remove from schedule
                    setattr(self, name, action.act(self.timestamp, timestamp, value, **kwargs))
                    self.scheduled_actions.pop(name)
                else:
                    # carry-out action to timestamp
                    setattr(self, name, action.act(self.timestamp, timestamp, value, **kwargs))

        self.timestamp = timestamp

    @abstractmethod
    def validate_timestamp(self) -> bool:
        """Method to validate the timestamp of the actionable.

        Returns
        -------
        bool
            True if timestamp is valid, False otherwise.
        """

        raise NotImplementedError
