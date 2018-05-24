# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Deletor(Base):
    """Deletor base class"""

    @abstractmethod
    def check_for_deletion(self):
        """Abstract method to check if a given track should be deleted"""
        pass

    def delete_tracks(self, tracks, **kwargs):
        """Generic/Base track deletion method.

        Iterates through all tracks in a given list and calls
        :py:meth:`~check_for_deletion` to determine which
        tracks should be deleted and which should survive.

        Parameters
        ----------
        tracks : :class:`sets.Set`
            A set of :class:`stonesoup.types.Track` objects

        Returns
        -------
        : (:class:`sets.Set`, :class:`sets.Set`) :class:`tuple`
            A tuple whose first entry contains the set of survining tracks,\
            while the second contains the set of deleted tracks.
        """

        surviving_tracks = set()
        deleted_tracks = set()

        for track in tracks:
            if(self.check_for_deletion(track, **kwargs)):
                deleted_tracks.add(track)
            else:
                surviving_tracks.add(track)

        return (surviving_tracks, deleted_tracks)
