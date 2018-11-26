# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Deleter(Base):
    """Deleter base class.

    Proposes tracks for deletion.
    """

    @abstractmethod
    def check_for_deletion(self, track, **kwargs):
        """Abstract method to check if a given track should be deleted"""
        pass

    def delete_tracks(self, tracks, **kwargs):
        """Generic/Base track deletion method.

        Iterates through all tracks in a given list and calls
        :py:meth:`~check_for_deletion` to determine which
        tracks should be deleted and which should survive.

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            A set of :class:`~.Track` objects

        Returns
        -------
        : set of :class:`~.Track`
            Set of tracks proposed for deletion.
        """

        return {track
                for track in tracks
                if self.check_for_deletion(track, **kwargs)}
