import datetime
from abc import abstractmethod
from typing import Iterator, Any

from ...base import Base, Property


class Action(Base):

    generator: Any = Property(default=None,
                              readonly=True,
                              doc="Action generator that created the action.")
    end_time: datetime.datetime = Property(readonly=True)

    @property
    def current_value(self):
        return getattr(self.owner, self.attribute)

    def act(self, current_time, timestamp, init_value):
        """Return the attribute modified.
        Parameters
        ----------
        duration: datetime.timedelta
            Duration of modification of attribute
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
    owner: object = Property(doc="Actionable object that has the attribute to be modified.")
    attribute: str = Property(doc="The name of the attribute to be modified.")
    start_time: datetime.datetime = Property()
    end_time: datetime.datetime = Property()

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[Action]:
        raise NotImplementedError()

    @property
    def current_value(self):
        return getattr(self.owner, self.attribute)

    @property
    def default_action(self):
        """The default action to modify the property if there is no given action."""
        raise NotImplementedError()


class RealNumberActionGenerator(ActionGenerator):

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
