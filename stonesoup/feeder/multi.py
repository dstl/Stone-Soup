import datetime
import heapq
import warnings
from itertools import tee
from typing import Collection, Iterable, Tuple, Set, Iterator, Sequence, Dict

from .base import DetectionFeeder, GroundTruthFeeder, Feeder, TrackFeeder, MultipleTrackFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader import Reader
from ..types.track import Track


class MultiDataFeeder(DetectionFeeder, GroundTruthFeeder):
    """Multi-data Feeder

    This returns states from multiple data readers as a single stream,
    yielding from the reader yielding the lowest timestamp first.
    """
    reader = None
    readers: Collection[Reader] = Property(doc='Readers to yield from')

    @BufferedGenerator.generator_method
    def data_gen(self):
        yield from heapq.merge(*self.readers)


class SyncMultipleTrackFeedersToOneFeeder(MultipleTrackFeeder):
    readers: Collection[TrackFeeder] = Property(doc='Readers to yield from')

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Sequence[Set[Track]]]]:
        for output in zip(*self.readers):
            all_times = set()
            all_track_sets = []
            for time, tracks in output:
                all_times.add(time)
                all_track_sets.append(tracks)
            if len(all_times) != 1:
                raise Exception("Track Feeders are outputting different times")
            time = all_times.pop()
            yield time, all_track_sets


class FeederToMultipleFeeders(Feeder):
    """
    Todo make formatting better
    This takes output from one feeder/reader and distributes it to multiple feeders

    Process:
    > Object Creation:
        > A empty dictionary 'iterator_clones_dict' is set up to contain all iterators clones
    > [Repeated 'x' times] create_feeder is called
        > A generator function is created 'iterator_clone_generator'
        > An entry is added to the 'iterator_clone_generator' with the key of this function and the
        value is 'None
        > The generator function is returned
    > First next call on any of the iterator clones
        > Tries to retrieve a copy of the iterator from the 'iterator_clones_dict' using function as
        it's key. This returns None.
        > As the returned value from the dictionary is None. '_activate_feeders' is called
        > _activate_feeders
            > Generates 'x' iterator copies using the 'tee' function
            > Each copy is assigned to a value in the dictionary
        > Tries to retrieve a copy of the iterator from the 'iterator_clones_dict' using function as
        it's key. This now returns an iterator clone
        > This iterator is now iterated over
    """
    reader: Iterable = Property(readonly=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator_clones_dict = dict()
        self._has_been_activated = False

    def _activate_feeders(self):
        feeder_clones = tee(self.reader, len(self.iterator_clones_dict))
        for sub_feeder, feeder_key in zip(feeder_clones, self.iterator_clones_dict):
            self.iterator_clones_dict[feeder_key] = sub_feeder
        self._has_been_activated = True

    def create_feeder(self):
        if self._has_been_activated:
            warnings.warn("Feeders have already been activated")

        def iterator_clone_generator():
            feeder_clone = self.iterator_clones_dict.get(iterator_clone_generator)
            if feeder_clone is None:
                self._activate_feeders()
                feeder_clone = self.iterator_clones_dict.get(iterator_clone_generator)

            yield from feeder_clone

        self.iterator_clones_dict[iterator_clone_generator] = None
        return iterator_clone_generator()

    @BufferedGenerator.generator_method
    def data_gen(self):
        yield from self.create_feeder()
