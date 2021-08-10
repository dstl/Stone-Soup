# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Set

from ..base import Base, Property
from ..types.track import Track
from ..types.update import Update


class Deleter(Base):
    """Deleter base class.

    Proposes tracks for deletion.
    """

    delete_last_pred: bool = Property(default=False, doc="Remove the state that caused a track to "
                                                         "be deleted if it is a prediction.")

    @abstractmethod
    def check_for_deletion(self, track: Track, **kwargs) -> bool:
        """Check if a given track should be deleted.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if track should be deleted, `False` otherwise.
        """
        pass

    def delete_tracks(self, tracks: Set[Track], **kwargs) -> Set[Track]:
        """Generic/Base track deletion method.

        Iterates through all tracks in a given list and calls
        :meth:`~check_for_deletion` to determine which
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

        tracks_to_delete = {track for track in tracks if self.check_for_deletion(track, **kwargs)}

        if self.delete_last_pred:
            for track in tracks_to_delete:
                if not isinstance(track[-1], Update):
                    del track[-1]
                    del track.metadatas[-1]

        return tracks_to_delete
