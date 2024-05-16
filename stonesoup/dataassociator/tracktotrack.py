import datetime
from itertools import chain
from operator import attrgetter
from typing import MutableSequence, Set

import numpy as np
import scipy
from ordered_set import OrderedSet

from stonesoup.types.state import State, StateMutableSequence

from ..base import Property
from ..measures import Euclidean, EuclideanWeighted, Measure
from ..measures.base import TrackMeasure
from ..types.association import Association, AssociationSet, TimeRangeAssociation
from ..types.groundtruth import GroundTruthPath
from ..types.time import TimeRange
from ..types.track import Track
from ._assignment import multidimensional_deconfliction
from .base import TwoTrackToTrackAssociator
from .general import OneToOneAssociator


class TrackToTrackCounting(TwoTrackToTrackAssociator):
    """Track to track associator based on the Counting Technique

    Compares two sets of :class:`~.tracks`, each formed of a sequence of
    :class:`~.State` objects and returns an :class:`~.Association` object for
    each time at which the two :class:`~.State` within the :class:`~.tracks`
    are assessed to be associated.

    Uses an algorithm called the Counting Technique [1]_.
    Associations are triggered by track states being within a threshold
    distance for a given number of timestamps. Associations are terminated when
    either the two :class:`~.tracks` end or the two :class:`~.State` are
    separated by a distance greater than the threshold at the next time step.

    References
    ----------
    .. [1] J. Å. Sagild, A. Gullikstad Hem and E. F. Brekke,
           "Counting Technique versus Single-Time Test for Track-to-Track Association,"
           2021 IEEE 24th International Conference on Information Fusion (FUSION), 2021, pp. 1-7
    Note
    ----
    Association is not prioritised based on historic associations or distance.
    If, at a specific time step, the :class:`~.State` of one of the
    :class:`~.tracks` is assessed as close to more than one track then an
    :class:`~.Association` object will be return for all possible association
    combinations.


    """

    association_threshold: float = Property(
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded")
    consec_pairs_confirm: int = Property(
        default=3,
        doc="Number of consecutive time instances which track pairs are "
            "required to be within a specified threshold in order for an "
            "association to be formed. Default is 3")
    consec_misses_end: int = Property(
        default=2,
        doc="Number of consecutive time instances which track pairs are "
            "required to exceed a specified threshold in order for an "
            "association to be ended. Default is 2")
    measure: Measure = Property(
        default=None,
        doc="Distance measure to use. Must use :class:`~.measures.EuclideanWeighted()` if "
            "`use_positional_only` set to True.  Default  is "
            ":class:`~.measures.EuclideanWeighted()` using :attr:`use_positional_only` "
            "and :attr:`pos_map`.  Note if neither are provided this is equivalent to a "
            "standard Euclidean")
    pos_map: list = Property(
        default=None,
        doc="List of items specifying the mapping of the position components "
            "of the state space for :attr:`tracks_set_1`.  "
            "Defaults to whole :class:`~.array.StateVector()`, but must be provided whenever "
            ":attr:`use_positional_only` is set to True")
    use_positional_only: bool = Property(
        default=True,
        doc="If `True`, the differences in velocity/acceleration values for each state are "
            "ignored in the calculation for the association threshold.  Default is `True`"
    )
    position_weighting: float = Property(
        default=0.6,
        doc="If :attr:`use_positional_only` is set to False, this decides how much to weight "
            "position components compared to others (such as velocity).  "
            "Default is 0.6"
    )
    one_to_one: bool = Property(
        default=False,
        doc="If True, it is ensured no two associations ever contain the same track "
            "at the same time"
    )

    def associate_tracks(self, tracks_set_1: Set[Track], tracks_set_2: Set[Track]):
        """Associate two sets of tracks together.

        Parameters
        ----------
        tracks_set_1 : set of :class:`~.Track` objects
            Tracks to associate to track set 2
        tracks_set_2 : set of :class:`~.Track` objects
            Tracks to associate to track set 1

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """
        if self.position_weighting > 1 or self.position_weighting < 0:
            raise ValueError("Position weighting must be between 0 and 1")
        if not self.pos_map and self.use_positional_only:
            raise ValueError("Must provide mapping of position components to pos_map")

        if not self.measure:
            state1 = next(iter(tracks_set_1))[0]
            total = len(state1.state_vector)
            if not self.pos_map:
                self.pos_map = [i for i in range(total)]

            pos_map_len = len(self.pos_map)
            if not self.use_positional_only and total - pos_map_len > 0:
                v_weight = (1 - self.position_weighting) / (total - pos_map_len)
                p_weight = self.position_weighting / pos_map_len
            else:
                p_weight = 1 / pos_map_len
                v_weight = 0

            weights = [p_weight if i in self.pos_map else v_weight
                       for i in range(total)]

            self.measure = EuclideanWeighted(weighting=weights)

        associations = set()
        for track2 in tracks_set_2:
            truth_timestamps = [state.timestamp for state in track2.states]
            for track1 in tracks_set_1:

                track1_states = sorted(
                    (state
                     for state in track1
                     if state.timestamp in truth_timestamps),
                    key=attrgetter('timestamp'))
                track_timestamps = [state.timestamp for state in track1_states]

                track2_states = sorted(
                    (state
                     for state in track2
                     if state.timestamp in track_timestamps),
                    key=attrgetter('timestamp'))

                if not (track1_states and track2_states):
                    continue

                # At this point we should have two lists of states from
                # track1 and 2 only at the times that they both existed

                n_successful = 0
                n_unsuccessful = 0
                start_timestamp = None
                end_timestamp = None
                # Loop through every detection pair and form associations
                for state1, state2 in zip(track1_states, track2_states):

                    distance = self.measure(state1, state2)

                    if distance <= self.association_threshold:
                        n_successful += 1
                        n_unsuccessful = 0

                        if n_successful == 1:
                            first_timestamp = state1.timestamp
                        if n_successful == self.consec_pairs_confirm:
                            start_timestamp = first_timestamp
                    else:
                        n_successful = 0
                        n_unsuccessful += 1

                        if n_unsuccessful == 1:
                            end_timestamp = state1.timestamp

                        if n_unsuccessful >= self.consec_misses_end and \
                                start_timestamp:
                            associations.add(TimeRangeAssociation(
                                OrderedSet((track1, track2)),
                                TimeRange(start_timestamp, end_timestamp)))
                            start_timestamp = None

                # close any open associations
                if start_timestamp:
                    end_timestamp = track1_states[-1].timestamp
                    associations.add(TimeRangeAssociation(
                        OrderedSet((track1, track2)),
                        TimeRange(start_timestamp, end_timestamp)))

        if self.one_to_one:
            return multidimensional_deconfliction(AssociationSet(associations))
        else:
            return AssociationSet(associations)


class TrackToTruth(TwoTrackToTrackAssociator):
    """Track to truth associator

    Compares two sets of :class:`~.Track`, each formed of a sequence of
    :class:`~.State` objects and returns an :class:`~.Association` object for
    each time at which a the two :class:`~.State` within the :class:`~.Track`
    are assessed to be associated. Tracks are considered to be associated with
    the Truth if the true :class:`~.State` is the closest to the track and
    within the specified distance for a specified number of time steps.

    Associations between Truth and Track if the Truth is no longer the
    'closest' to the track or the distance exceeds the specified threshold for
    a specified number of consecutive time steps.

    Associates will be ended by consec_misses_end before any new associations
    are considered even if consec_pairs_confirm < consec_misses_end

    Note
    ----
    Tracks can only be associated with one Truth (one-2-one relationship) at a
    given time step however a Truth track can be associated with multiple
    Tracks (one-2-many relationship).
    """

    association_threshold: float = Property(
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded")
    consec_pairs_confirm: int = Property(
        default=3,
        doc="Number of consecutive time instances which track-truth pairs are "
            "required to be within a specified threshold in order for an "
            "association to be formed. Default is 3")
    consec_misses_end: int = Property(
        default=2,
        doc="Number of consecutive time instances which track-truth pairs are "
            "required to exceed a specified threshold in order for an "
            "association to be ended. Default is 2")
    measure: Measure = Property(
        default=Euclidean(),
        doc="Distance measure to use. Default :class:`~.measures.Euclidean()`")

    def associate_tracks(self, tracks_set: Set[Track], truth_set: Set[GroundTruthPath]):
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

        associations = set()

        for track in tracks_set:

            current_truth = None
            potential_truth = None
            n_potential_successes = 0
            n_failures = 0
            potential_start_timestep = None
            start_timestamp = None
            end_timestamp = None

            truth_state_iters = {truth: GroundTruthPath.last_timestamp_generator(truth)
                                 for truth in truth_set}
            truth_states = {truth: next(truth_state_iter)
                            for truth, truth_state_iter in truth_state_iters.items()}

            for track_state in Track.last_timestamp_generator(track):

                min_dist = None
                min_truth = None

                for truth in truth_set:
                    if truth[0].timestamp > track_state.timestamp \
                            or truth[-1].timestamp < track_state.timestamp:
                        continue

                    while truth_states[truth].timestamp < track_state.timestamp:
                        truth_states[truth] = next(truth_state_iters[truth])
                    truth_state = truth_states[truth]
                    if truth_state.timestamp != track_state.timestamp:
                        continue

                    distance = self.measure(track_state, truth_state)
                    if min_dist and distance < min_dist:
                        min_dist = distance
                        min_truth = truth
                    elif not min_dist \
                            and distance < self.association_threshold:
                        min_dist = distance
                        min_truth = truth

                # If there is not a truth track currently
                # considered to be associated to the track
                if not current_truth:
                    # If no truth associated then there's nothing to consider
                    if min_truth is None:
                        n_potential_successes = 0
                        potential_truth = None
                        potential_start_timestep = None
                    # If the latest closest truth is not being assessed
                    # as the likely truth make it so
                    elif potential_truth is not min_truth:
                        potential_truth = min_truth
                        n_potential_successes = 1
                        potential_start_timestep = track_state.timestamp

                    # Otherwise increse the number of times
                    # this truth appears in a row
                    else:
                        n_potential_successes += 1
                    # If the threshold of continuous
                    # similar matches has been made
                    if n_potential_successes >= self.consec_pairs_confirm:
                        current_truth = min_truth
                        start_timestamp = potential_start_timestep
                        end_timestamp = track_state.timestamp
                        potential_start_timestep = None
                        potential_truth = None
                        n_potential_successes = 0

                # Otherwise if there is a track currently
                # considered as the association
                else:
                    # If the closest track this time is the same
                    # update the end time (time of last association)
                    if min_truth == current_truth:
                        n_failures = 0
                        end_timestamp = track_state.timestamp
                    # Otherwise record the failed match and how
                    # many times it's been the same different
                    # potential track in a row
                    else:
                        n_failures += 1
                        if min_truth and min_truth is potential_truth:
                            n_potential_successes += 1
                        else:
                            potential_truth = min_truth
                            potential_start_timestep = track_state.timestamp
                            n_potential_successes = 1

                    # If there have been enough failed matches
                    # in a row end the association and record
                    if n_failures >= self.consec_misses_end:
                        associations.add(TimeRangeAssociation(
                            OrderedSet((track, current_truth)),
                            TimeRange(start_timestamp, end_timestamp)))

                        # If the current potential association
                        # is strong enough to confirm then do so
                        if n_potential_successes >= self.consec_pairs_confirm:

                            current_truth = potential_truth
                            start_timestamp = potential_start_timestep
                            end_timestamp = track_state.timestamp

                        else:
                            # Otherwise wait for a new
                            # association to be good enough
                            current_truth = None
                            start_timestamp = None
                            end_timestamp = None

            # Close any open associations when the track ends
            if current_truth:

                associations.add(TimeRangeAssociation(
                    OrderedSet((track, current_truth)),
                    TimeRange(start_timestamp, end_timestamp)))

        return AssociationSet(associations)


class TrackIDbased(TwoTrackToTrackAssociator):
    """Track ID based associator

        Compares set of :class:`~.Track` objects to set of :class:`~.GroundTruth` objects,
        each formed of a sequence of :class:`~.State` objects and returns an
        :class:`~.Association` object for each time at which a the two :class:`~.State`
        within the :class:`~.Track` and :class:`~.GroundTruthPath` are assessed to be associated.
        Tracks are considered to be associated with the Ground Truth if the ID of the Track
        is the same as the ID of the Ground Truth.
        """

    def associate_tracks(self, tracks_set, truths_set):
        """Associate two sets of tracks together.

               Parameters
               ----------
               tracks_set : list of :class:`~.Track` objects
                   Tracks to associate to ground truths set
               truths_set: list of :class:`~.GroundTruthPath` objects
                   Ground truths to associate to tracks set

               Returns
               -------
               AssociationSet
                   Contains a set of :class:`~.Association` objects

               """

        associations = set()

        for track in tracks_set:
            for truth in truths_set:
                if track.id == truth.id:
                    try:
                        associations.add(
                            TimeRangeAssociation(OrderedSet((track, truth)),
                                                 TimeRange(max(track[0].timestamp,
                                                               truth[0].timestamp),
                                                           min(track[-1].timestamp,
                                                               truth[-1].timestamp))))
                    except (TypeError, ValueError):
                        # A timestamp is None, or non-overlapping timestamps (start > end)
                        associations.add(Association(OrderedSet((track, truth))))

        return AssociationSet(associations)


class OneToOneTrackAssociator(TwoTrackToTrackAssociator, OneToOneAssociator):
    """ Uses the :class:`~.OneToOneAssociator` to associate tracks together """

    measure: TrackMeasure = Property()

    def associate_tracks(self, *tracks_sets: Set[Track]) -> AssociationSet:
        if len(tracks_sets) != 2:  # Should have two sets of tracks
            raise ValueError("There should be two sources of tracks to compare")

        tracks_a, tracks_b = tracks_sets

        associated_tracks, _, _ = self.associate(tracks_a, tracks_b)

        return associated_tracks


class ClearMotAssociator(TwoTrackToTrackAssociator):
    """TODO
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

    def associate_tracks(self, tracks_set: Set[Track], truth_set: Set[GroundTruthPath]):
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

        # Make a list of all the unique timestamps used
        # Make a sorted list of all the unique timestamps used
        track_states = self.extract_states(tracks_set)
        truth_states = self.extract_states(truth_set)
        timestamps = sorted({
            state.timestamp
            for state in chain(track_states, truth_states)})
        start_time = timestamps[0]
        end_time = timestamps[-1]

        timesteps = []
        matches_over_time = []
        matches_previous = set()

        current_time = start_time
        while current_time <= end_time:

            truth_ids_at_current_time = [truth_id for (truth_id, truth_states)
                                         in truth_states_by_id.items()
                                         if get_state_at_time(truth_states, current_time)]
            track_ids_at_current_time = [track_id for (track_id, track_states)
                                         in track_states_by_id.items()
                                         if get_state_at_time(track_states, current_time)]
            matches_current = set()

            if matches_previous:

                for (truth_id, track_id) in matches_previous:
                    truth_states = truth_states_by_id[truth_id]
                    track_states = track_states_by_id[track_id]

                    truth_state_current = get_state_at_time(truth_states, current_time)

                    if not truth_state_current:
                        continue

                    track_state_current = get_state_at_time(track_states, current_time)
                    if not track_state_current:
                        continue

                    distance = self.measure(track_state_current, truth_state_current)

                    if distance < self.association_threshold:
                        matches_current.add((truth_id, track_id))

                        truth_ids_at_current_time.remove(truth_id)
                        track_ids_at_current_time.remove(track_id)

            if not truth_ids_at_current_time or not track_ids_at_current_time:
                matches_over_time.append(matches_current)
                timesteps.append(current_time)

                matches_previous = matches_current
                current_time += self.time_interval
                continue

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
                    matches_current.add((truth_id, track_id))

            # save
            matches_over_time.append(matches_current)
            timesteps.append(current_time)

            matches_previous = matches_current
            current_time += self.time_interval

        truth_by_id = {truth.id: truth for truth in truth_set}
        track_by_id = {track.id: track for track in tracks_set}

        # --- create associations ---
        associations = set()
        for truth_id in truth_states_by_id.keys():

            assigned_track_ids_over_time = list()

            for t, match_set in zip(timesteps, matches_over_time):

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
                    start_time = timesteps[i]

                if assigned_track_id != current_track_id:
                    associations.append(TimeRangeAssociation(OrderedSet(
                        (track_by_id[current_track_id], truth_by_id[truth_id])),
                        TimeRange(start_time, timesteps[i])))
                    start_time = timesteps[i] if assigned_track_id else None
                    current_track_id = assigned_track_id

                # end of timeseries
                if i == (len(assigned_track_ids_over_time)-1):
                    associations.add(TimeRangeAssociation(OrderedSet(
                        (track_by_id[current_track_id], truth_by_id[truth_id])),
                        TimeRange(start_time, timesteps[i])))
                    break

        return AssociationSet(associations)

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


def get_state_at_time(states: MutableSequence[State], timestamp: datetime.datetime,
                      max_error: datetime.timedelta = datetime.timedelta(milliseconds=10)) -> State | None:

    state_timestamps = [state.timestamp for state in states]

    for state_timestamp in state_timestamps:
        if state_timestamp >= timestamp:
            break

    error = state_timestamp - timestamp
    if error < max_error:
        return states[state_timestamps.index(state_timestamp)]
    else:
        return None
