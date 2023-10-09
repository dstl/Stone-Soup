import heapq
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Sequence, List, Optional, Collection

from .base import Property
from .measures import Measure, BaseMeasure
from .types.state import StateMutableSequence
from .types.track import Track


class MultipleMeasure(BaseMeasure):
    """ MultipleMeasure base class.

    This measure produces a list of ``float`` values instead of a singular ``float`` value. This
    can be used when comparing objects that contain multiple other objects.
    """
    @abstractmethod
    def __call__(self, item1, item2) -> List[float]:
        raise NotImplementedError


class TrackMeasure(BaseMeasure):
    """TrackMeasure base class.

    A measure provides a means to assess the separation between two track objects ``track_1``
    and ``track_2``. It should return a float value of the distance measure between the two tracks.
    """

    @abstractmethod
    def __call__(self, track_1: Track, track_2: Track) -> float:
        """Compute the distance between a pair of tracks."""
        raise NotImplementedError


class StateSequenceMeasure(MultipleMeasure):
    """
    Applies a state measure to each state in the state sequence with matching times.
    """

    state_measure: Measure = Property(doc="The measure used to compare individual states.")

    def __call__(self, state_sequence_1: StateMutableSequence,
                 state_sequence_2: StateMutableSequence,
                 times_to_measure: Sequence[datetime] = None) -> List[float]:
        """
        Compare the states from each state sequence for every time in ``times_to_measure``.

        If ``times_to_measure`` is None. Find all times that both state sequences have in common.

        Parameters
        ----------
        state_sequence_1 : :class:`.~StateMutableSequence`
            a state sequence to compare against ``state_sequence_2``.
        state_sequence_2 : :class:`.~StateMutableSequence`
            a state sequence to compare against ``state_sequence_1``.
        times_to_measure : Sequence of :class:`.~datetime`
            Calculate the state measure for states in the state sequences at these times. Default
            value is ``None``. If ``None``, ``times_to_measure`` is calculated as all the times
            that both state sequences have in common.

        Returns
        -------
        List[float]
            a list of distance measures between a states in the state sequence inputs.

        """

        if times_to_measure is None:
            track_1_times = {state.timestamp for state in state_sequence_1.states}
            track_2_times = {state.timestamp for state in state_sequence_2.states}
            times_to_measure = sorted(track_1_times & track_2_times)

            if len(times_to_measure) == 0:
                warnings.warn("No measures are calculated as there are not any times that match "
                              "between the two state sequences.")

        measures = [self.state_measure(state_sequence_1[time], state_sequence_2[time])
                    for time in times_to_measure]

        return measures


class RecentStateSequenceMeasure(MultipleMeasure):
    """
    Applies a state measure to each state in the state sequence with for the most recent *n*
    matching times. It will return less than ``n_states_to_compare`` values if there are less
    matching times.
    """

    state_measure: Measure = Property(doc="The measure used to compare individual states.")
    n_states_to_compare: int = Property(doc="Maximum number of states to be compared.")

    def __call__(self, state_sequence_1: StateMutableSequence,
                 state_sequence_2: StateMutableSequence) -> List[float]:
        """
        Compare the states from each state sequence for the most recent ``n_states_to_compare``
        times.

        Parameters
        ----------
        state_sequence_1 : :class:`.~StateMutableSequence`
            a state sequence to compare against ``state_sequence_2``.
        state_sequence_2 : :class:`.~StateMutableSequence`
            a state sequence to compare against ``state_sequence_1``.

        Returns
        -------
        float
            a list of distance measures between a states in the state sequence inputs. These are
            returned in ascending state time order.

        """

        track_1_times = {state.timestamp for state in state_sequence_1.states}
        track_2_times = {state.timestamp for state in state_sequence_2.states}

        times_in_both = track_1_times & track_2_times

        times_to_measure = heapq.nlargest(self.n_states_to_compare, times_in_both)

        # Not strictly needed but means output will be in ascending time order
        times_to_measure = list(reversed(times_to_measure))

        state_sequence_measure = StateSequenceMeasure(self.state_measure)
        return state_sequence_measure(state_sequence_1, state_sequence_2, times_to_measure)


class MeanMeasure(BaseMeasure):
    """
    This class converts multiple measures into one mean average measure.
    """
    measure: MultipleMeasure = Property()

    def __call__(self, *args, **kwargs) -> Optional[float]:
        measures: List[float] = self.measure(*args, **kwargs)

        if len(measures) == 0:
            return None
        else:
            return sum(measures)/len(measures)


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
