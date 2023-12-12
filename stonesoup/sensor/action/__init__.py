import datetime
from abc import abstractmethod
from typing import Iterator, Any

from ...base import Base, Property


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
