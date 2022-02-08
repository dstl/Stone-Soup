# -*- coding: utf-8 -*-
from operator import attrgetter
from typing import Set

from .base import TrackToTrackAssociator
from ..base import Property
from ..measures import Measure, Euclidean
from ..types.association import AssociationSet, TimeRangeAssociation, Association
from ..types.groundtruth import GroundTruthPath
from ..types.track import Track
from ..types.time import TimeRange


class TrackToTrack(TrackToTrackAssociator):
    """Track to track associator

    Compares two sets of :class:`~.tracks`, each formed of a sequence of
    :class:`~.State` objects and returns an :class:`~.Association` object for
    each time at which a the two :class:`~.State` within the :class:`~.tracks`
    are assessed to be associated.

    Associations are triggered by track states being within a threshold
    distance for a given number of timestamps. Associations are terminated when
    either the two :class:`~.tracks` end or the two :class:`~.State` are
    separated by a distance greater than the threshold at the next time step.

    Note
    ----
    Association is not prioritised based on historic associations or distance.
    If, at a specific time step, the :class:`~.State` of one of the
    :class:`~.tracks` is assessed as close to more than one track then an
    :class:`~.Association` object will be return for all possible association
    combinations
    """

    association_threshold: float = Property(
        default=10,
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded.Default is 10")
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
        default=Euclidean(),
        doc="Distance measure to use. Default :class:`~.measures.Euclidean()`")

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

                n_succesful = 0
                n_unsuccesful = 0
                start_timestamp = None
                end_timestamp = None
                # Loop through every detection pair and form associations
                for state1, state2 in zip(track1_states, track2_states):

                    distance = self.measure(state1, state2)

                    if distance <= self.association_threshold:
                        n_succesful += 1
                        n_unsuccesful = 0

                        if n_succesful == 1:
                            first_timestamp = state1.timestamp
                        if n_succesful == self.consec_pairs_confirm:
                            start_timestamp = first_timestamp
                    else:
                        n_succesful = 0
                        n_unsuccesful += 1

                        if n_unsuccesful == 1:
                            end_timestamp = state1.timestamp

                        if n_unsuccesful >= self.consec_misses_end and \
                                start_timestamp:
                            associations.add(TimeRangeAssociation(
                                (track1, track2),
                                TimeRange(start_timestamp, end_timestamp)))
                            start_timestamp = None

                # close any open associations
                if start_timestamp:
                    end_timestamp = track1_states[-1].timestamp
                    associations.add(TimeRangeAssociation(
                        (track1, track2),
                        TimeRange(start_timestamp, end_timestamp)))

        return AssociationSet(associations)


class TrackToTruth(TrackToTrackAssociator):
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
        default=10,
        doc="Threshold distance measure which states must be within for an "
            "association to be recorded.Default is 10")
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
                            (track, current_truth),
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
                    (track, current_truth),
                    TimeRange(start_timestamp, end_timestamp)))

        return AssociationSet(associations)


class TrackIDbased(TrackToTrackAssociator):
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
                            TimeRangeAssociation((track, truth),
                                                 TimeRange(max(track[0].timestamp,
                                                               truth[0].timestamp),
                                                           min(track[-1].timestamp,
                                                               truth[-1].timestamp))))
                    except (TypeError, ValueError):
                        # A timestamp is None, or non-overlapping timestamps (start > end)
                        associations.add(Association((track, truth)))

        return AssociationSet(associations)
