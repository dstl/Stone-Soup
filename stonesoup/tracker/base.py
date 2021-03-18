# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """Returns a generator of tracks for each time step.

        Returns
        -------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError
