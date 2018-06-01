# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Deletor(Base):
    """Deletor base class.

    Proposes tracks for deletion."""

    @abstractmethod
    def delete(self, tracks, **kwargs):
        """Returns set of tracks for deletion.

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Tracks to be checked for possible deletion.

        Returns
        -------
        : set of :class:`~.Track`
            Tracks proposed for deletion.

        """
        raise NotImplemented
