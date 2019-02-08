# -*- coding: utf-8 -*-
from operator import attrgetter

import numpy as np

from ..base import Property
from .base import TrackToTrackAssociator
from ..models.measurement import MeasurementModel
from ..types.association import AssociationSet, TimeRangeAssociation
from ..types.time import TimeRange


class EuclideanTrackToTrack(TrackToTrackAssociator):
    """Euclidean track to track associator

    Compares two sets of tracks, each formed of a sequence of :class:`~.State`
    and returns an association for each time a track from one set is
    associated with a track from the other. Associations are triggered by track
    states being within a threshold for a given number of timestamps and ended
    by one track ending or the states being outside of the threshold for a
    given number of timestamps.

    No prioritisation of tracks is performed. If one track is near two tracks
    from the other set then associations will be created for both
    """

    association_threshold = Property(
        float, default=10,
        doc="Distance between states within which an association occurs")
    consec_pairs_confirm = Property(
        int, default=3,
        doc="Number of consecutive track-truth states within threshold to "
            "confirm association of track to truth")
    consec_misses_end = Property(
        int, default=2,
        doc="Number of consecutive track-truth states ouside threshold to end "
            "association of track to truth")
    measurement_model_track1 = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the "
            "track state are to be used to calculate distance over")
    measurement_model_track2 = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the "
            "track state are to be used to calculate distance over")

    def associate_tracks(self, tracks_set_1, tracks_set_2):
        """Associate two sets of tracks together.

        Parameters
        ----------
        tracks_set_1 : list of :class:`~.Track` objects
            Tracks to associate to track set 2
        tracks_set_2 : list of :class:`~.Track` objects
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

                    distance = np.linalg.norm(
                        self.measurement_model_track1.function(
                            state1.state_vector, noise=0)
                        - self.measurement_model_track2.function(
                            state2.state_vector, noise=0))

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


class EuclideanTrackToTruth(TrackToTrackAssociator):
    """Euclidean track to truth associator

    Returns an association for each track that is following a truth path.
    Tracks are assumed to be following a truth if the truth is the closest
    state to the track (and within a threshold) for a given number of
    consecutive timestamps. Associations end if the truth is not the closest
    to the track (or the distance is outside the threshold) for a given number
    of consecutive tracks. Each track can only be associated to one truth at
    a given timestamp but truths may have multiple tracks associated to them.

    Associates will be ended by attr:`consec_misses_end` before any new
    associations are considered even if :attr:`consec_pairs_confirm` <
    :attr:`consec_misses_end`
    """

    association_threshold = Property(
        float, default=10,
        doc="Distance between states within which an association occurs")

    consec_pairs_confirm = Property(
        int, default=3,
        doc="Number of consecutive track-truth states within threshold to "
            "confirm association of track to truth")

    consec_misses_end = Property(
        int, default=2,
        doc="Number of consecutive track-truth states ouside threshold to end "
            "association of track to truth")
    measurement_model_track = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the "
            "track state are to be used to calculate distance over")

    measurement_model_truth = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the "
            "truth state are to be used to calculate distance over")

    def associate_tracks(self, tracks_set, truth_set):
        """Associate two sets of tracks together.

        Parameters
        ----------
        tracks_set : list of :class:`~.Track` objects
            Tracks to associate to truth
        truth_set : list of :class:`~.Track` objects
            Truth to associated tracks to

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

            for track_state in track:

                min_dist = None
                min_truth = None

                for truth in truth_set:

                    try:
                        truth_state = truth[track_state.timestamp]
                    except IndexError:
                        continue

                    distance = np.linalg.norm(
                        self.measurement_model_track.function(
                            track_state.state_vector, noise=0)
                        - self.measurement_model_truth.function(
                            truth_state.state_vector, noise=0))
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
