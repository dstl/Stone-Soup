import copy
import datetime
from typing import Tuple, Set

import numpy as np

from ..tracker2 import SingleTargetTracker2, MultiTargetTracker2, \
    MultiTargetMixtureTracker2, SingleTargetMixtureTracker2, PointProcessMultiTargetTracker2
from ... import measures
from ...hypothesiser.distance import DistanceHypothesiser
from ...hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from ...mixturereducer.gaussianmixture import GaussianMixtureReducer
from ...models.measurement.linear import LinearGaussian
from ...types.angle import Bearing
from ...types.detection import Detection
from ...types.state import TaggedWeightedGaussianState
from ...types.track import Track
from ...updater.kalman import KalmanUpdater
from ...updater.pointprocess import PHDUpdater


def test_single_target_tracker(initiator, deleter, detector, data_associator, updater):
    tracker = SingleTargetTracker2(initiator=initiator, deleter=deleter, detector=detector,
                                   data_associator=data_associator, updater=updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    total_tracks = set()
    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert len(tracks) <= 1  # Shouldn't have more than one track
        for track in tracks:
            assert len(track.states) <= 10  # Deleter should delete these
        total_tracks |= tracks

        previous_time = time

    assert len(total_tracks) >= 2  # Should of had at least 2 over all steps


def test_single_target_mixture_tracker(
        initiator, deleter, detector, data_mixture_associator, updater):
    tracker = SingleTargetMixtureTracker2(initiator=initiator, deleter=deleter, detector=detector,
                                          data_associator=data_mixture_associator, updater=updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    total_tracks = set()
    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert len(tracks) <= 1  # Shouldn't have more than one track
        for track in tracks:
            assert len(track.states) <= 10  # Deleter should delete them
        total_tracks |= tracks

        previous_time = time
    assert len(total_tracks) >= 2


def test_multi_target_tracker(
        initiator, deleter, detector, data_associator, updater):
    tracker = MultiTargetTracker2(initiator=initiator, deleter=deleter, detector=detector,
                                  data_associator=data_associator, updater=updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    max_tracks = 0
    total_tracks = set()
    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert len(tracks) <= 3  # Shouldn't have more than three tracks
        for track in tracks:
            assert len(track.states) <= 10  # Deleter should delete these
        if len(tracks) == 3:
            sorted_vectors = sorted(track.state_vector for track in tracks)
            assert sorted_vectors[0] + 10 == sorted_vectors[1]
            assert sorted_vectors[1] + 10 == sorted_vectors[2]

        max_tracks = max(max_tracks, len(tracks))
        total_tracks |= tracks

        previous_time = time

    assert max_tracks >= 3  # Should of had at least 3 tracks in single step
    assert len(total_tracks) >= 6  # Should of had at least 6 over all steps


def test_multi_target_mixture_tracker(
        initiator, deleter, detector, data_mixture_associator, updater):
    tracker = MultiTargetMixtureTracker2(initiator=initiator, deleter=deleter, detector=detector,
                                         data_associator=data_mixture_associator, updater=updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    max_tracks = 0
    total_tracks = set()
    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        # assert len(tracks) <= 3  # Shouldn't have more than three tracks
        for track in tracks:
            assert len(track.states) <= 10  # Deleter should delete these
        # if len(tracks) == 3:
        #    sorted_vectors = sorted(track.state_vector for track in tracks)
        #    assert sorted_vectors[0] + 10 == sorted_vectors[1]
        #    assert sorted_vectors[1] + 10 == sorted_vectors[2]

        max_tracks = max(max_tracks, len(tracks))
        total_tracks |= tracks

        previous_time = time

    assert max_tracks >= 3  # Should of had at least 3 tracks in single step
    assert len(total_tracks) >= 6  # Should of had at least 6 over all steps


def test_point_process_multi_target_tracker_cycle(detector, predictor):
    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    timestamp = datetime.datetime.now()
    birth_mean = np.array([[40]])
    birth_covar = np.array([[1000]])
    birth_component = TaggedWeightedGaussianState(
        birth_mean,
        birth_covar,
        weight=0.3,
        tag=TaggedWeightedGaussianState.BIRTH,
        timestamp=timestamp)

    # Initialise a Kalman Updater
    measurement_model = LinearGaussian(ndim_state=1, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    updater = KalmanUpdater(measurement_model=measurement_model)
    # Initialise a Gaussian Mixture hypothesiser
    measure = measures.Mahalanobis()
    base_hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=16)
    hypothesiser = GaussianMixtureHypothesiser(hypothesiser=base_hypothesiser,
                                               order_by_detection=True)

    # Initialise a Gaussian Mixture reducer
    merge_threshold = 4
    prune_threshold = 1e-5
    reducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                     merge_threshold=merge_threshold)

    # Initialise a Point Process updater
    phd_updater = PHDUpdater(updater=updater, prob_detection=0.8)

    tracker = PointProcessMultiTargetTracker2(
        detector=detector,
        updater=phd_updater,
        hypothesiser=hypothesiser,
        reducer=reducer,
        birth_component=birth_component
        )

    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert tracker.estimated_number_of_targets > 0
        assert tracker.estimated_number_of_targets < 4
        previous_time = time
        # Shouldn't have more than three active tracks
        assert (len(tracks) >= 1) & (len(tracks) <= 3)
        # All tracks should have unique IDs
        assert len(tracker.gaussian_mixture.component_tags) == len(tracker.gaussian_mixture)


def test_tracker_copies(initiator, deleter, detector, data_associator, updater):
    # Test is trackers can be copied and the copies perform like the originals

    trackers = [MultiTargetTracker2(initiator=initiator, deleter=deleter, detector=detector,
                                    data_associator=data_associator, updater=updater)]

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)

    for tracker_input in detector:
        for tracker_idx, tracker in enumerate(trackers):
            time, tracks = tracker.update_tracker(*tracker_input)

            # standard checks
            assert time == previous_time + datetime.timedelta(minutes=1)

            if tracker_idx == 0:
                track0_state_vectors = {tuple(state.state_vector[0] for state in track.states)
                                        for track in tracks}
            else:
                track_state_vectors = {tuple(state.state_vector[0] for state in track.states)
                                       for track in tracks}

                assert track0_state_vectors == track_state_vectors

        previous_time = time
        trackers.append(copy.deepcopy(trackers[-1]))


class DummyAngleTracker(MultiTargetTracker2):

    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:

        angle_detections = copy.deepcopy(detections)
        for det in angle_detections:
            det.state_vector[0] = np.deg2rad(det.state_vector[0])

        time, tracks = super().update_tracker(time, angle_detections)

        for track in tracks:
            track.state.state_vector[0] = Bearing(track.state.state_vector[0])

        return time, tracks


def test_angle_tracker(initiator, deleter, detector, data_associator, updater):
    # Test the update_tracker function can be overwritten

    tracker = DummyAngleTracker(initiator=initiator, deleter=deleter, detector=detector,
                                data_associator=data_associator, updater=updater)

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)

    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)

        for track in tracks:
            assert isinstance(track.state.state_vector[0], Bearing)
            assert track.state.state_vector[0] < 10

        previous_time = time
