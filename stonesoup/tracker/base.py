# -*- coding: utf-8 -*-
from abc import abstractmethod
import warnings

from ..base import Base
from ..buffered_generator import BufferedGenerator


class Tracker(Base, BufferedGenerator):
    """Tracker base class"""

    @property
    def tracks(self):
        return self.current[1]

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        warnings.warn('Track generators may be removed in future releases', DeprecationWarning, stacklevel=2)
        tracker_iter = iter(self)
        for time, tracks in tracker_iter:
            yield (time, tracks)

    '''
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
    '''
