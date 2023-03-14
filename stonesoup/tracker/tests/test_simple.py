import pytest
import datetime
from ..simple import SingleTargetTracker, MultiTargetTracker, \
    MultiTargetMixtureTracker


def test_single_target_manual_update(
        initiator, deleter, detector, data_associator, updater):

    tracker = SingleTargetTracker(initiator=initiator, deleter=deleter,
                                  data_associator=data_associator, updater=updater)

    with pytest.raises(ValueError):
        for _ in tracker:
            pass

    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    total_tracks = set()
    for detection_time, detections in detector:
        tracker_time, tracks = tracker.update_tracker(detection_time, detections)
        assert detection_time == tracker_time == previous_time + datetime.timedelta(minutes=1)

        assert len(tracks) <= 1  # Shouldn't have more than one track
        total_tracks |= tracks

        previous_time = tracker_time

    assert len(total_tracks) >= 2  # Should of had at least 2 over all steps


def test_single_target_tracker(
        initiator, deleter, detector, data_associator, updater):

    tracker = SingleTargetTracker(initiator=initiator, deleter=deleter, detector=detector,
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


def test_multi_target_tracker(
        initiator, deleter, detector, data_associator, updater):

    tracker = MultiTargetTracker(initiator=initiator, deleter=deleter, detector=detector,
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

    tracker = MultiTargetMixtureTracker(initiator=initiator, deleter=deleter, detector=detector,
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
