import numpy as np
from itertools import groupby
from ..base import Property
from .base import TrackToTrackAssociator
from ..types import TimePeriodAssociation, AssociationSet


class EuclidianTrackToTrack(TrackToTrackAssociator):
    """Assuming that the trakcs have states at the same timestamps

    Compares two sets of tracks and returns an association for each time a track
    from one set is associated with a track from the other. Associations are
    triggered by track states being within a threshold for a given number of
    timesteps and ended by one track ending or the states being outside of the
    threshold for a given number of timesteps.

     No prioritisation of tracks is performed. If a track is near two tracks
     from the other set then associations will be created for both"""

    association_threshold = Property(float, default=10,
                                     doc="Distance between states within which an association occurs")

    consec_pairs_confirm = Property(int, default=3,
                                    doc="Number of consecutive track-truth states within threshold to"
                                        "confirm association of track to truth")

    consec_misses_end = Property(int, default=2,
                                 doc="Number of consecutive track-truth states ouside threshold to "
                                     "end association of track to truth")
    measurement_matrix_track1 = Property(np.ndarray,
                                         doc='Measurement matrix for the track states to extract parameters to'
                                             ' calculate distance over')
    measurement_matrix_track2 = Property(np.ndarray,
                                         doc='Measurement matrix for the track states to extract parameters to'
                                             ' calculate distance over')

    def associate_tracks(self, tracks_set_1, tracks_set_2):

        associations = set()
        for track2 in tracks_set_2:
            for track1 in tracks_set_1:

                truth_timestamps = [state.timestamp for state in track2.states]
                track1_states = [state for state in track1.states if
                                 state.timestamp in truth_timestamps]
                track_timestamps = [state.timestamp for state in track1_states]
                track2_states = [state for state in track2.states if
                                 state.timestamp in track_timestamps]

                if track2_states != [] and track1_states != []:

                    # Ensure everything is in chronological order
                    track1_states.sort(key=lambda x: x.timestamp)
                    track2_states.sort(key=lambda x: x.timestamp)

                    # At this point we should have two lists of states from track1 and 2 only at the times that they both existed

                    # Get distances between track 1 and 2 at each timestamp
                    distances = [np.linalg.norm(
                        self.measurement_matrix_track1 @ track1_states[
                            i].state_vector.__array__()
                        - self.measurement_matrix_track2 @ track2_states[
                            i].state_vector.__array__())
                        for i in range(len(track1_states))]

                    n_succesful = 0
                    n_unsuccesful = 0
                    start_timestamp = None
                    end_timestamp = None
                    # Loop through every detection pair and form associations
                    for i in range(len(track1_states)):

                        if distances[i] <= self.association_threshold:

                            n_succesful += 1
                            n_unsuccesful = 0

                            if n_succesful >= self.consec_pairs_confirm and not start_timestamp:
                                start_timestamp = track1_states[
                                    i - (self.consec_misses_end - 1)].timestamp

                        else:

                            n_succesful = 0
                            n_unsuccesful += 1

                            if n_unsuccesful >= self.consec_misses_end and start_timestamp:
                                end_timestamp = track1_states[
                                    i - self.consec_misses_end - 1].timestamp

                                associations.add(
                                    TimePeriodAssociation((track1, track2),
                                                          start_timestamp=start_timestamp,
                                                          end_timestamp=end_timestamp))
                                start_timestamp = None
                                end_timestamp = None

                    # close any open associations
                    if start_timestamp:
                        end_timestamp = track1_states[-1].timestamp
                        associations.add(TimePeriodAssociation((track1, track2),
                                                               start_timestamp=start_timestamp,
                                                               end_timestamp=end_timestamp))

        return AssociationSet(associations)


class EuclidianTrackToTruth(TrackToTrackAssociator):
    """Assuming that the tracks and truth have states at the same timestamps

    Returns an association for each track that is following a truth path. Tracks
    are assumed to be following a truth if the truth is the closest state to the
    track (and within a threshold) for a given number of consecutive timestamps.
    Associations end if the truth is not the closest to the track (or the
     distance is outside the threshold) for a given number of consecutive tracks
    Each track can only be associated to one truth at a given timestamp.

     No prioritisation of tracks is performed. If 3 tracks are within the
     threshold then each will be associated to both the others.

     Associates will be ended by consec_misses_end before any new associations are
     considered even if consec_pairs_confirm < consec_misses_end"""

    association_threshold = Property(float, default=10,
                                     doc="Distance between states within which an association occurs")

    consec_pairs_confirm = Property(int, default=3,
                                    doc="Number of consecutive track-truth states within threshold to"
                                        "confirm association of track to truth")

    consec_misses_end = Property(int, default=2,
                                 doc="Number of consecutive track-truth states ouside threshold to "
                                     "end association of track to truth")
    measurement_matrix_truth = Property(np.ndarray,
                                        doc='Measurement matrix for the truth states to extract parameters to'
                                            ' calculate distance over')
    measurement_matrix_track = Property(np.ndarray,
                                        doc='Measurement matrix for the track states to extract parameters to'
                                            ' calculate distance over')

    def associate_tracks(self, tracks_set, truth_set):
        associations = set()

        for track in tracks_set:

            closest_truths = []
            current_truth = None
            potential_truth = None
            n_potential_successes = 0
            n_failures = 0
            potential_start_timestep = None
            start_timestamp = None
            end_timestamp = None

            for track_state in track.states:

                min_dist = None
                min_truth = None

                for truth in truth_set:

                    if track_state.timestamp in [i.timestamp for i in
                                                 truth.states]:
                        truth_state = truth[track_state.timestamp]
                    else:
                        truth_state = None

                    if truth_state:
                        distance = np.linalg.norm(
                            self.measurement_matrix_track @ track_state.state_vector.__array__()
                            - self.measurement_matrix_truth @ truth_state.state_vector.__array__())
                        if min_dist and distance < min_dist:
                            min_dist = distance
                            min_truth = truth
                        elif not min_dist and distance < self.association_threshold:
                            min_dist = distance
                            min_truth = truth

                # If there is not a truth track currently considered to be associated to the track
                if not current_truth:
                    # If no truth associated then there's nothing to consider
                    if min_truth == None:
                        n_potential_successes = 0
                        potential_start_timestep = None
                    # If the latest closest truth is not being assessed as the likely truth make it so
                    elif potential_truth != min_truth:
                        potential_truth = min_truth
                        n_potential_successes = 1
                        potential_start_timestep = track_state.timestamp

                    # Otherwise increse the number of times this truth appears in a row
                    else:
                        n_potential_successes += 1
                    # If the threshold of continuous similar matches has been made
                    if n_potential_successes >= self.consec_pairs_confirm:
                        current_truth = min_truth
                        start_timestamp = potential_start_timestep
                        end_timestamp = track_state.timestamp
                        potential_start_timestep = None
                        potential_truth = None
                        n_potential_successes = 0

                # Otherwise if there is a track currently considered as the association
                else:
                    # If the closest track this time is the same update the end time (time of last association)
                    if min_truth == current_truth:
                        n_failures = 0
                        end_timestamp = track_state.timestamp
                    # Otherwise record the failed match and how many times it's been the same different
                    # potential track in a row
                    else:
                        n_failures += 1
                        if min_truth == potential_truth:
                            n_potential_successes += 1
                        else:
                            potential_truth = min_truth
                            potential_start_timestep = track_state.timestamp
                            n_potential_successes = 1

                    # If there have been enough failed matches in a row end the association and recod
                    if n_failures >= self.consec_misses_end:

                        associations.add(TimePeriodAssociation(
                            (track, current_truth),
                            start_timestamp=start_timestamp,
                            end_timestamp=end_timestamp))

                        # If the current potential association is strong enough to confirm then do so
                        if n_potential_successes >= self.consec_pairs_confirm:

                            current_truth = potential_truth
                            start_timestamp = potential_start_timestep
                            end_timestamp = track_state.timestamp

                        else:
                            # Otherwise wait for a new association to be good enough
                            current_truth = None
                            start_timestamp = None
                            end_timestamp = None

            # Close any open associations when the track ends
            if current_truth:
                associations.add(TimePeriodAssociation(
                    (track, current_truth),
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp
                ))
        return AssociationSet(associations)
