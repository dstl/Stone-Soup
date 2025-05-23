import datetime
import itertools
import sys
from collections import defaultdict
from collections.abc import Generator, Iterable, MutableSequence
from typing import Any

import numpy as np
import scipy
from ordered_set import OrderedSet

from ..base import Property
from ..measures import Euclidean, Measure
from ..types.association import AssociationSet, TimeRangeAssociation
from ..types.groundtruth import GroundTruthPath
from ..types.state import State
from ..types.time import TimeRange
from ..types.track import Track
from .base import TwoTrackToTrackAssociator

StatesFromTimeIdLookup = dict[datetime.datetime, dict[str, State]]
TrackFromIdLookup = dict[str, Track]


class ClearMotAssociator(TwoTrackToTrackAssociator):
    """Track to truth associator used in the CLEAR MOT metrics paper[1].

    Compares two sets of :class:`~.Track`, each formed of a sequence of
    :class:`~.State` objects and returns an :class:`~.Association` object for
    each time at which a the two :class:`~.State` within the :class:`~.Track`
    are assessed to be associated. A track keeps its association with the
    truth from previous timestep,even if there is a new track which is closer to the truth.
    Unassigned tracks and truths are matched using Munkres algorithm if they
    are below the specified distance threshold.

    Note
    ----
    A track can only be associated with one Truth (one-2-one relationship) at a
    given time step and vice versa.

    Reference
        [1] Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics,
            Bernardin et al, 2008
    """

    association_threshold: float = Property(
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded")
    measure: Measure = Property(
        default=Euclidean(),
        doc="Distance measure to use. Default :class:`~.measures.Euclidean()`")

    def associate_tracks(self, tracks_set: set[Track], truth_set: set[GroundTruthPath])\
            -> AssociationSet:
        """Associate Tracks

        Method compares to sets of :class:`~.Track` objects and will determine
        associations between the two sets.

        Parameters
        ----------
        tracks_set : set of :class:`~.Track` objects
            Tracks to associate to truth
        truth_set : set of :class:`~.GroundTruthPath` objects
            Truth to associate to tracks

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects
        """

        timestamps, truth_states_by_time_id, track_states_by_time_id, \
            truth_tracks_by_id, estim_tracks_by_id = \
            self._prepare_timestamps_and_helpful_lookups(tracks_set, truth_set)

        # we use this to collect match sets over time
        matches_over_time: list[set[tuple[str, str]]] = []

        # holds the match set of the previous timestep in (truth_id, track_id) format
        matches_previous: set[tuple[str, str]] = set()

        for current_time in timestamps:

            truth_ids_at_current_time = OrderedSet(truth_states_by_time_id[current_time])
            track_ids_at_current_time = OrderedSet(track_states_by_time_id[current_time])
            truth_states_by_id = truth_states_by_time_id[current_time]
            track_states_by_id = track_states_by_time_id[current_time]

            matches_current, remaining_truth_ids_at_current_time, \
                remaining_track_ids_at_current_time = \
                self._forward_matches_from_previous_timestep(truth_states_by_id,
                                                             track_states_by_id,
                                                             matches_previous,
                                                             truth_ids_at_current_time,
                                                             track_ids_at_current_time,
                                                             )

            # in case either the truth or tracks are empty, continue with the next timestep,
            # since there is nothing left to associate anymore
            if not remaining_truth_ids_at_current_time or not remaining_track_ids_at_current_time:
                matches_over_time.append(matches_current)
                matches_previous = matches_current
                continue

            matches_from_unassigned = \
                self._match_unassigned_tracks(truth_states_by_id,
                                              track_states_by_id,
                                              remaining_truth_ids_at_current_time,
                                              remaining_track_ids_at_current_time)
            matches_current |= matches_from_unassigned

            matches_over_time.append(matches_current)
            matches_previous = matches_current

        associations = self._create_associations_from_matches_over_time(estim_tracks_by_id,
                                                                        truth_tracks_by_id,
                                                                        timestamps,
                                                                        matches_over_time)

        return AssociationSet(associations)

    def _prepare_timestamps_and_helpful_lookups(self, tracks_set: set[Track],
                                                truth_set: set[Track]) -> \
        tuple[list[datetime.datetime], StatesFromTimeIdLookup, StatesFromTimeIdLookup,
              TrackFromIdLookup, TrackFromIdLookup]:
        """Helper function to prepare lookups and determine unique timestamps across
        both truth and tracks.
        """

        timestamps = set()

        truth_tracks_by_id = {truth.id: truth for truth in truth_set}
        estim_tracks_by_id = {track.id: track for track in tracks_set}

        truth_states_by_time_id: StatesFromTimeIdLookup = defaultdict(dict)
        for truth in truth_set:
            for state in truth.last_timestamp_generator():
                truth_states_by_time_id[state.timestamp][truth.id] = state
                timestamps.add(state.timestamp)

        track_states_by_time_id: StatesFromTimeIdLookup = defaultdict(dict)
        for track in tracks_set:
            for state in track.last_timestamp_generator():
                track_states_by_time_id[state.timestamp][track.id] = state
                timestamps.add(state.timestamp)

        # Make a sorted list of all the unique timestamps used
        timestamps = sorted(timestamps)
        return timestamps, truth_states_by_time_id, track_states_by_time_id, \
            truth_tracks_by_id, estim_tracks_by_id

    def _create_associations_from_matches_over_time(self,
                                                    estim_tracks_by_id: dict[str, Track],
                                                    truth_tracks_by_id: dict[str, GroundTruthPath],
                                                    timestamps: MutableSequence[datetime.datetime],
                                                    matches_over_time: list[set[tuple[str, str]]]):
        unique_matches = {
            match for matches_timestamp in matches_over_time for match in matches_timestamp}

        associations = set()
        for match in unique_matches:
            timesteps_where_match_exists = list()
            for i, matches_timestamp in enumerate(matches_over_time):
                if match in matches_timestamp:
                    timesteps_where_match_exists.append(i)

            # deal with temporal gaps in associations
            time_intervals = get_strictly_monotonously_increasing_intervals(
                timesteps_where_match_exists)

            for (start_idx, end_idx) in time_intervals:
                associations.add(TimeRangeAssociation(OrderedSet(
                    (estim_tracks_by_id[match[0]], truth_tracks_by_id[match[1]])),
                    TimeRange(timestamps[timesteps_where_match_exists[start_idx]],
                              timestamps[timesteps_where_match_exists[end_idx-1]])))

        return associations

    def _forward_matches_from_previous_timestep(self,
                                                truth_states_by_id: dict[str, State],
                                                track_states_by_id: dict[str, State],
                                                matches_previous: set[tuple[str, str]],
                                                truth_ids_at_current_time: OrderedSet[str],
                                                track_ids_at_current_time: OrderedSet[str]) \
            -> tuple[set[tuple[str, str]], OrderedSet[str], OrderedSet[str]]:
        """Checks if matches from the previous timestep are still valid by their distance and
        adds them to the returned set of matches. Note that, the variables
        """

        matches_current = set()

        if not matches_previous:
            return matches_current, truth_ids_at_current_time, track_ids_at_current_time

        # we iterate over each match and check if it is still valid (i.e. below the
        # assication threshold - if true, we keep it and add it to current set,
        # if not we do not maintain the match
        for (track_id, truth_id) in matches_previous:
            try:
                truth_state_current = truth_states_by_id[truth_id]
                track_state_current = track_states_by_id[track_id]
            except KeyError:
                continue

            distance = self.measure(track_state_current, truth_state_current)

            # if distance is still lower than the threshold, keep the match
            if distance < self.association_threshold:
                matches_current.add((track_id, truth_id))

                truth_ids_at_current_time.remove(truth_id)
                track_ids_at_current_time.remove(track_id)
        return matches_current, truth_ids_at_current_time, track_ids_at_current_time

    def _match_unassigned_tracks(self, truth_states_by_id: dict[str, State],
                                 track_states_by_id: dict[str, State],
                                 truth_ids_at_current_time: OrderedSet[str],
                                 track_ids_at_current_time: OrderedSet[str]
                                 ) -> set[tuple[str, str]]:
        """Match unassigned tracks using Munkers algorithm and distance threshold.
        """
        num_truth_unassigned = len(truth_ids_at_current_time)
        num_tracks_unassigned = len(track_ids_at_current_time)
        cost_matrix = np.zeros((num_truth_unassigned, num_tracks_unassigned), dtype=float)

        matches = set()

        for i in range(num_truth_unassigned):
            for j in range(num_tracks_unassigned):
                truth_id, track_id = truth_ids_at_current_time[i], track_ids_at_current_time[j]

                truth_state_current = truth_states_by_id[truth_id]
                track_state_current = track_states_by_id[track_id]
                distance = self.measure(track_state_current, truth_state_current)
                cost_matrix[i, j] = distance

        # Munkers / Hungarian Method for the assignment problem
        row_ind, col_in = scipy.optimize.linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_in):
            if cost_matrix[i, j] < self.association_threshold:
                matches.add((track_ids_at_current_time[j], truth_ids_at_current_time[i]))
        return matches


def get_strictly_monotonously_increasing_intervals(arr: MutableSequence[int])\
        -> list[tuple[int, int]]:
    """Return (start <= t < end) index intervals where array elements are increasing by 1.

    Args:
        timesteps (MutableSequence[int]): array

    Returns:
        list[tuple[int, int]]: intervals with indices, where
            array elements are increasing monotonically by 1
    """
    time_jumps = np.diff(arr) > 1
    valid_interval_start_indices = np.r_[0, 1+np.where(time_jumps)[0], len(arr)]
    intervals = []
    for start_idx, end_idx in pairwise(valid_interval_start_indices):
        intervals.append((start_idx, end_idx))
    return intervals


def pairwise(iterable: Iterable[Any]) -> Generator[Any, None, None]:
    """pairwise('ABCDEFG') → AB BC CD DE EF FG
    """

    if sys.version_info >= (3, 10):
        yield from itertools.pairwise(iterable)
    else:
        # Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b
