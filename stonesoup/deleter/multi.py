"""Contains deleters which use a composite of deleters to decide whether a track is to be deleted
"""

from typing import Collection

from ..base import Property
from .base import Deleter


class CompositeDeleter(Deleter):
    """ Track deleter composed of multiple deleters.

    If :attr:`intersect` is True, deletes tracks if they satisfy the deletion conditions of each
    deleter listed in :attr:`deleters`. Otherwise deletes tracks if they satisfy the conditions of
    at least one deleter listed.
    """

    deleters: Collection[Deleter] = Property(doc="List of deleters to be applied to the track")
    intersect: bool = Property(
        default=True,
        doc="Boolean that determines whether the composite deleter will intersect or unify "
            "deletion results. Default is `True`, applying an intersection.")

    def check_for_deletion(self, track, **kwargs):
        if self.intersect:
            for deleter in self.deleters:
                if not deleter.check_for_deletion(track, **kwargs):
                    return False
            return True
        else:
            for deleter in self.deleters:
                if deleter.check_for_deletion(track, **kwargs):
                    return True
            return False
