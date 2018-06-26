# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self):
        """The tracks at the current time step.

        This is the set of tracks last returned by the
        :meth:`tracks_gen` generator, to allow other components, like
        metrics, to access the data.
        """
        raise NotImplementedError

    @abstractmethod
    def tracks_gen(self):
        """Returns a generator of tracks for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError
