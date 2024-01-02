from abc import abstractmethod
from typing import Any, Collection

from ..base import Base
from ..types.track import Track


class BaseMeasure(Base):
    """Abstract Measure Base Type

    A measure provides a means to assess the separation between two
    objects item1 and item2.
    """

    @abstractmethod
    def __call__(self, item1: Any, item2: Any) -> float:
        r"""
        Compute the distance between a pair of objects

        Parameters
        ----------
        item1 : Any
        item2 : Any

        Returns
        -------
        float
            distance measure between a pair of input objects

        """
        raise NotImplementedError


class TrackMeasure(BaseMeasure):
    """TrackMeasure base class.

    A measure provides a means to assess the separation between two :class:`~.Track` objects
    `track_1` and `track_2`. It should return a float value of the distance measure between the
    two tracks.
    """

    @abstractmethod
    def __call__(self, track_1: Track, track_2: Track) -> float:
        """Compute the distance between a pair of :class:`~.Track` objects."""
        raise NotImplementedError


class SetComparisonMeasure(BaseMeasure):
    """
    This class measures how many items are present in both collections. The type of the collections
    is ignored and duplicate items are ignored.

     * The measure output is between 0 and 1 (inclusive).
     * An output of 1 is for both collections to contain the same items.
     * An output of 0 is when there are zero items in common between the two sets.

    This class compares an object/item’s identity. It doesn't directly compare objects
    (``obj1 is obj2`` rather than ``obj1 == obj2``).

    """

    def __call__(self, collection_1: Collection, collection_2: Collection) -> float:
        """
        The measure is calculated by finding the number of items in common between the two
        collections and divides it by the total number of unique items in the combined collection.

        Parameters
        ----------
        collection_1 :
        collection_2 :

        Returns
        -------
        float
            distance measure between a pair of input objects.

        """
        set_1 = set(collection_1)
        set_2 = set(collection_2)
        all_item = set_1 | set_2
        items_in_both = set_1 & set_2

        if len(all_item) == 0:
            return 0
        else:
            return len(items_in_both)/len(all_item)
