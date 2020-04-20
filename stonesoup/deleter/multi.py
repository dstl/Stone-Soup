# -*- coding: utf-8 -*-
"""Contains the multi-deleter feeders"""

from ..base import Property
from .base import Deleter


class MultiDeleterIntersect(Deleter):
    """ Track deleter composed of multiple deleters.

    Deletes tracks if they satisfy the conditions of all deleters applied.
    """

    deleters = Property([Deleter], doc="List of deleters to be applied to the"
                                       "track")

    def check_for_deletion(self, track, **kwargs):

        """Check if a given track should be deleted

        A track is flagged for deletion if it satisfies the deletion
        conditions given by :py:meth:`check_for_deletion` of all deleters
        listed in :py:attr:`deleters`.

        Parameters
        ----------
        track : :class:`stonesoup.types.Track`
            A track object to be checked for deletion.

        Returns
        -------
        : :class:`bool`
            ``True`` if track should be deleted, ``False`` otherwise.
        """

        for deleter in self.deleters:
            if not deleter.check_for_deletion(track, **kwargs):
                return False
        return True


class MultiDeleterUnion(Deleter):
    """ Track deleter composed of multiple deleters.

    Deletes tracks if they satisfy the conditions of at least one deleter
    applied.
    """

    deleters = Property([Deleter], doc="List of deleters to be applied to the"
                                       "track")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if it satisfies the deletion
        conditions given by :py:meth:`check_for_deletion` of at least one
        deleter listed in :py:attr:`deleters`

        Parameters
        ----------
        track : :class:`stonesoup.types.Track`
            A track object to be checked for deletion.

        Returns
        -------
        : :class:`bool`
            ``True`` if track should be deleted, ``False`` otherwise.
        """

        for deleter in self.deleters:
            if deleter.check_for_deletion(track, **kwargs):
                return True
        return False
