# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base
from ..buffered_generator import BufferedGenerator


class Tracker(Base, BufferedGenerator):
    """Tracker base class"""

    @property
    def tracks(self):
        return self.current[1]

    @abstractmethod
    @BufferedGenerator.generator_method
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
