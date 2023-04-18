import heapq
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Sequence, List, Optional, Collection

from .base import Property
from .measures import Measure, GenericMeasure
from .types.state import StateMutableSequence
from .types.track import Track


class MultipleMeasure(GenericMeasure):
    @abstractmethod
    def __call__(self, item1, item2) -> List[float]:
        raise NotImplementedError


class TrackMeasure(GenericMeasure):
    """Track Measure base type
    A measure provides a means to assess the separation between two track objects track1 and
    track2. It should return the float of the distance measure between the two tracks
    """

    @abstractmethod
    def __call__(self, track_1: Track, track_2: Track) -> float:
        r"""Compute the distance between a pair of tracks"""
        raise NotImplementedError


class StateSequenceMeasure(MultipleMeasure):
    """
    Applies a state measure to each state in the state sequence with matching times
    """

    measure: Measure = Property()

    def __call__(self, state_sequence_1: StateMutableSequence,
                 state_sequence_2: StateMutableSequence,
                 times_to_measure: Sequence[datetime] = None) -> List[float]:
        """
        Compare the states from each state sequence for every time in `times_to_measure`.
        If `times_to_measure` is None. Find all times that both state sequences have in common
        """

        if times_to_measure is None:
            track_1_times = {state.timestamp for state in state_sequence_1.states}
            track_2_times = {state.timestamp for state in state_sequence_2.states}
            times_to_measure = track_1_times & track_2_times

            if len(times_to_measure) == 0:
                warnings.warn("No measures are calculated as there are not any times that match "
                              "between the two state sequences ")

        measures = [self.measure(state_sequence_1[time], state_sequence_2[time])
                    for time in times_to_measure]

        return measures


class RecentStateSequenceMeasure(MultipleMeasure):
    """
    Applies a state measure to each state in the state sequence with for the most recent n matching
    times
    """

    measure: Measure = Property()
    n_states_to_compare: int = Property()

    def __call__(self, state_sequence_1: StateMutableSequence,
                 state_sequence_2: StateMutableSequence) -> List[float]:

        track_1_times = {state.timestamp for state in state_sequence_1.states}
        track_2_times = {state.timestamp for state in state_sequence_2.states}

        times_in_both = track_1_times & track_2_times

        times_to_measure = heapq.nlargest(self.n_states_to_compare, times_in_both)

        return StateSequenceMeasure(self.measure)(state_sequence_1, state_sequence_2,
                                                  times_to_measure)


class MeanMeasure(GenericMeasure):
    """
    This class converts multiple measures into one mean average measure
    """
    measure: MultipleMeasure = Property()

    def __call__(self, *args, **kwargs) -> Optional[float]:
        measures: List[float] = self.measure(*args, **kwargs)

        if len(measures) == 0:
            return None
        else:
            return sum(measures)/len(measures)


class SetComparisonMeasure(GenericMeasure):
    """
    This class measures how many items are present in both collections. The type of the collections
    is ignored and duplicate items are ignored.
        The measure output is between 0 and 1 (inclusive).
        An output of 1 is for both collections to contain the same items.
        An output of 0 is when there are zero items in common between the two sets.

    This class compares an object/item’s identity. It doesn't directly compare objects (obj1 is
    obj2 rather than obj1 == obj2).

    """

    def __call__(self, collection_1: Collection, collection_2: Collection) -> float:
        r"""
        The measure is calculated by finding the number of items in common between the two
        collections and divides it by the total number of unique items in the combined collection

        Parameters
        ----------
        collection_1 :
        collection_2 :

        Returns
        -------
        float
            distance measure between a pair of input objects

        """
        set_1 = set(collection_1)
        set_2 = set(collection_2)
        all_item = set_1 | set_2
        items_in_both = set_1 & set_2

        if len(all_item) == 0:
            return 0
        else:
            return len(items_in_both)/len(all_item)
