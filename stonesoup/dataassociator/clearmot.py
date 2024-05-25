import datetime
from itertools import chain
from typing import MutableSequence, Set

import numpy as np
import scipy
from ordered_set import OrderedSet

from ..base import Property
from ..measures import Euclidean, Measure
from ..types.association import AssociationSet, TimeRangeAssociation
from ..types.groundtruth import GroundTruthPath
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange
from ..types.track import Track
from .base import TwoTrackToTrackAssociator


class ClearMotAssociator(TwoTrackToTrackAssociator):
    """
    2. Deal with gaps in matches over time
    3. Adapt docs the class
    4. Remove self.time_interval
    """

    association_threshold: float = Property(
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded")
    time_interval: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=1),
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded")
    measure: Measure = Property(
        default=Euclidean(),
        doc="Distance measure to use. Default :class:`~.measures.Euclidean()`")

    def associate_tracks(self, tracks_set: Set[Track], truth_set: Set[GroundTruthPath]) -> AssociationSet:
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

        truth_states_by_id = {truth.id: truth.states for truth in truth_set}
        track_states_by_id = {track.id: track.states for track in tracks_set}

        # Make a sorted list of all the unique timestamps used
        timestamps = self.determine_unique_timestamps(tracks_set, truth_set)

        # we use this to collect match sets over time
        matches_over_time = []

        # holds the match set of the previous timestep in (truth_id, track_id) format
        matches_previous: set[tuple[Any, Any]] = set()

        for current_time in timestamps:

            truth_ids_at_current_time, track_ids_at_current_time = \
                self.get_truth_and_track_ids_at_a_specific_time(truth_states_by_id, 
                                                                track_states_by_id, current_time)
            
            matches_current = set()

            if matches_previous:

                # we iterate over each match and check if it is still valid (i.e. below the
                # assication threshold - if true, we keep it, if not we do not maintain the match
                self.verify_if_previos_matches_are_still_valid(truth_states_by_id,
                                                               track_states_by_id,
                                                               matches_previous,
                                                               current_time,
                                                               truth_ids_at_current_time,
                                                               track_ids_at_current_time,
                                                               matches_current)

            # continue, in case either the truth or tracks are empty, since there is nothing
            # left anymore to associate
            if not truth_ids_at_current_time or not track_ids_at_current_time:
                matches_over_time.append(matches_current)
                matches_previous = matches_current
                continue
            
            self.match_unassigned_tracks(truth_states_by_id, track_states_by_id,
                                         current_time, truth_ids_at_current_time,
                                         track_ids_at_current_time, matches_current)

            matches_over_time.append(matches_current)
            matches_previous = matches_current

        truth_tracks_by_id = {truth.id: truth for truth in truth_set}
        estim_tracks_by_id = {track.id: track for track in tracks_set}

        unique_matches = {match for matches_timestamp in matches_over_time for match in matches_timestamp}

        associations = set()
        for match in unique_matches:

            timesteps_where_match_exists = list()
            for i, matches_timestamp in enumerate(matches_over_time):
                if match in matches_timestamp:
                    timesteps_where_match_exists.append(i)

            # TODO: deal with gaps in associations

            associations.add(TimeRangeAssociation(OrderedSet(
                    (estim_tracks_by_id[match[0]], truth_tracks_by_id[match[1]])),
                    TimeRange(timestamps[timesteps_where_match_exists[0]],
                              timestamps[timesteps_where_match_exists[-1]])))

        return AssociationSet(associations)

    def get_truth_and_track_ids_at_a_specific_time(self, truth_states_by_id, track_states_by_id, current_time):
        truth_ids_at_current_time = [truth_id for (truth_id, truth_states)
                                         in truth_states_by_id.items()
                                         if get_state_at_time(truth_states, current_time)]
        track_ids_at_current_time = [track_id for (track_id, track_states)
                                         in track_states_by_id.items()
                                         if get_state_at_time(track_states, current_time)]
                                     
        return truth_ids_at_current_time, track_ids_at_current_time

    @staticmethod
    def _create_associations_from_sequence_of_match_sets(truth_states_by_id, timestamps,
                                                        matches_over_time, truth_by_id, track_by_id) -> Set[TimeRangeAssociation]:
        associations = set()
        for truth_id in truth_states_by_id.keys():
            assigned_track_ids_over_time = list()

            for t, match_set in zip(timestamps, matches_over_time):
                track_id_at_t = None
                for truth_id_in_match, track_id_in_match in match_set:
                    if truth_id_in_match == truth_id:
                        track_id_at_t = track_id_in_match
                        break
                assigned_track_ids_over_time.append(track_id_at_t)

            start_time = None
            current_track_id = None

            for i, assigned_track_id in enumerate(assigned_track_ids_over_time):
                if (not current_track_id) and assigned_track_id:
                    current_track_id = assigned_track_id
                    start_time = timestamps[i]

                if assigned_track_id != current_track_id:
                    associations.add(TimeRangeAssociation(OrderedSet(
                        (track_by_id[current_track_id], truth_by_id[truth_id])),
                        TimeRange(start_time, timestamps[i])))
                    start_time = timestamps[i] if assigned_track_id else None
                    current_track_id = assigned_track_id

                # end of timeseries
                if i == (len(assigned_track_ids_over_time)-1):
                    
                    if current_track_id is None:
                        continue

                    associations.add(TimeRangeAssociation(OrderedSet(
                        (track_by_id[current_track_id], truth_by_id[truth_id])),
                        TimeRange(start_time, timestamps[i])))
                    break
        return associations

    def match_unassigned_tracks(self, truth_states_by_id, track_states_by_id, current_time, truth_ids_at_current_time,
                                    track_ids_at_current_time, matches_current: Set):
        num_truth_unassigned = len(truth_ids_at_current_time)
        num_tracks_unassigned = len(track_ids_at_current_time)
        cost_matrix = np.zeros((num_truth_unassigned, num_tracks_unassigned), dtype=float)

        for i in range(num_truth_unassigned):
            for j in range(num_tracks_unassigned):
                truth_id, track_id = truth_ids_at_current_time[i], track_ids_at_current_time[j]

                truth_states = truth_states_by_id[truth_id]
                track_states = track_states_by_id[track_id]
                truth_state_current = get_state_at_time(truth_states, current_time)
                track_state_current = get_state_at_time(track_states, current_time)
                distance = self.measure(track_state_current, truth_state_current)
                cost_matrix[i, j] = distance

        # Munkers / Hungarian Method for the assignment problem
        row_ind, col_in = scipy.optimize.linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_in):
            if cost_matrix[i, j] < self.association_threshold:
                matches_current.add((track_ids_at_current_time[j], truth_ids_at_current_time[i]))

    def verify_if_previos_matches_are_still_valid(self, truth_states_by_id, track_states_by_id,
                                                  matches_previous, current_time,
                                                  truth_ids_at_current_time,
                                                  track_ids_at_current_time,
                                                  matches_current):
        for (track_id, truth_id) in matches_previous:
            # get
            truth_states = truth_states_by_id[truth_id]
            truth_state_current = get_state_at_time(truth_states, current_time)

            if not truth_state_current:
                continue

            track_states = track_states_by_id[track_id]
            track_state_current = get_state_at_time(track_states, current_time)

            # if hypothesis is not available anymore
            if not track_state_current:
                continue

            distance = self.measure(track_state_current, truth_state_current)

            # if distance is still lower than the threshold, keep the match
            if distance < self.association_threshold:
                matches_current.add((track_id, truth_id))

                truth_ids_at_current_time.remove(truth_id)
                track_ids_at_current_time.remove(track_id)


    def determine_unique_timestamps(self, tracks_set, truth_set) -> list[datetime.datetime]:
        track_states = self.extract_states(tracks_set)
        truth_states = self.extract_states(truth_set)
        timestamps = sorted({
            state.timestamp
            for state in chain(track_states, truth_states)})
        return timestamps

    @staticmethod
    def extract_states(object_with_states, return_ids=False):
        """
        Extracts a list of states from a list of (or single) objects
        containing states. This method is defined to handle :class:`~.StateMutableSequence`
        and :class:`~.State` types.

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object
        return_ids: If we should return obj ids as well.

        Returns
        -------
        : list of :class:`~.State`
        """

        state_list = StateMutableSequence()
        ids = []
        for i, element in enumerate(list(object_with_states)):
            if isinstance(element, StateMutableSequence):
                state_list.extend(element.states)
                ids.extend([i]*len(element.states))
            elif isinstance(element, State):
                state_list.append(element)
                ids.extend([i])
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))
        if return_ids:
            return state_list, ids
        return state_list


def get_state_at_time(state_sequence: MutableSequence[State],
                      timestamp: datetime.datetime) -> State | None:
    try:
        return Track(state_sequence)[timestamp]
    except IndexError:
        return None
