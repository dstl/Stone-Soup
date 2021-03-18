# -*- coding: utf-8 -*-
from abc import abstractmethod
import warnings

from ..base import Base
from ..buffered_generator import BufferedGenerator


class Tracker(Base, BufferedGenerator):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        Returns
        -------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        warnings.warn('Track generators may be removed in future releases', DeprecationWarning, stacklevel=2)
        tracker_iter = iter(self)
        for time, tracks in tracker_iter:
            yield (time, tracks)

