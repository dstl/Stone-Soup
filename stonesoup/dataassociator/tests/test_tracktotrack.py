import datetime
from typing import List

import pytest

from ...measures.state import Euclidean
from ...types.association import Association
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State
from ...types.track import Track
from ..clearmot import ClearMotAssociator
from ..tracktotrack import (
    TrackIDbased,
    TrackToTrackCounting,
    TrackToTruth,
)


@pytest.fixture
def tracks():
    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    tracks = [Track(id="Track-1", states=[
        State(state_vector=[[i], [i]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(10)])]

    # 2nd track should be associated with track1 from the second timestamp to
    # the 6th
    tracks.append(Track(id="Track-2", states=[
        State(state_vector=[[20], [20]],
              timestamp=start_time)]
        + [State(state_vector=[[i + 2], [i + 2]],
                 timestamp=start_time + datetime.timedelta(seconds=i))
           for i in range(1, 7)]))

    # 3rd is at a different time so should not associate with anything
    tracks.append(Track(id="Track-3", states=[
        State(state_vector=[[i], [i]],
              timestamp=start_time + datetime.timedelta(seconds=i + 20))
        for i in range(10)]))

    # 4th is outside the association threshold
    tracks.append(Track(id="Track-4", states=[
        State(state_vector=[[i + 20], [i + 20]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(10)]))

    # 5th should match with the first if only the first element of the state_vector is used
    tracks.append(Track(id="Track-5", states=[
        State(state_vector=[[i], [i+100]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(10)]))

    # 6th should match with 1st from the 2nd timestamp to the 7th, but continues after this point
    tracks.append(Track(id="Track-6", states=[
        State(state_vector=[[20], [20]],
              timestamp=start_time)]
        + [State(state_vector=[[i + 2], [i + 2]],
                 timestamp=start_time + datetime.timedelta(seconds=i))
            for i in range(1, 7)]
        + [State(state_vector=[[i + 1000], [i + 1000]],
                 timestamp=start_time + datetime.timedelta(seconds=i))
           for i in range(7, 10)]))
    
    # 7th crosses the 1st at the middle
    tracks.append(Track(id="Track-7", states=[
        State(state_vector=[[i], [10-i]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(10)]))

    # 8th first equals the 1st and later follows the 7th
    tracks.append(Track(id="Track-8", states=[State(state_vector=[[i], [i]],
                                      timestamp=start_time + datetime.timedelta(seconds=i))
                                for i in range(0, 5)]
                        + [State(state_vector=[[i], [10-i]],
                                 timestamp=start_time + datetime.timedelta(seconds=i))
                           for i in range(5, 10)]))

    return tracks


def test_euclidiantracktotrack(tracks):
    complete_associator = TrackToTrackCounting(
        association_threshold=10,
        consec_pairs_confirm=3,
        consec_misses_end=2,
        use_positional_only=False)
    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    association_set_1 = complete_associator.associate_tracks({tracks[0], tracks[2]},
                                                             {tracks[1], tracks[3], tracks[4]})

    positional_associator = TrackToTrackCounting(
        association_threshold=10,
        consec_pairs_confirm=3,
        consec_misses_end=2,
        pos_map=[0],
        use_positional_only=True)

    association_set_2 = positional_associator.associate_tracks({tracks[0], tracks[2]},
                                                               {tracks[1], tracks[3], tracks[4]})

    # Should give same results as the positional_associator for these tracks
    heavily_weighted_associator = TrackToTrackCounting(
        association_threshold=10,
        consec_pairs_confirm=3,
        consec_misses_end=2,
        pos_map=[0],
        use_positional_only=False,
        position_weighting=0.999)

    association_set_3 = heavily_weighted_associator.associate_tracks(
        {tracks[0], tracks[2]}, {tracks[1], tracks[3], tracks[4]})

    association_set_4 = complete_associator.associate_tracks({tracks[0]}, {tracks[5]})

    complete_associator_one2one = TrackToTrackCounting(
        association_threshold=10,
        consec_pairs_confirm=3,
        consec_misses_end=2,
        use_positional_only=False)
    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    association_set_one2one = complete_associator_one2one.associate_tracks(
        {tracks[0], tracks[2]}, {tracks[1], tracks[3], tracks[4]})

    assert len(association_set_1.associations) == 1
    assoc1 = list(association_set_1.associations)[0]
    assert set(assoc1.objects) == {tracks[0], tracks[1]}
    assert assoc1.time_range.start_timestamp \
        == start_time + datetime.timedelta(seconds=1)
    assert assoc1.time_range.end_timestamp \
        == start_time + datetime.timedelta(seconds=6)

    assert len(association_set_2.associations) == 2
    assoc20 = list(association_set_2.associations)[0]
    assoc21 = list(association_set_2.associations)[1]
    assert set(assoc20.objects) == {tracks[0], tracks[4]} or \
        set(assoc21.objects) == {tracks[0], tracks[4]}

    assert len(association_set_3.associations) == 2
    assoc30 = list(association_set_3.associations)[0]
    assoc31 = list(association_set_3.associations)[1]
    assert set(assoc30.objects) == {tracks[0], tracks[4]} or \
        set(assoc31.objects) == {tracks[0], tracks[4]}

    assert len(association_set_4) == 1
    assoc4 = list(association_set_4)[0]
    assert assoc4.time_range.start_timestamp \
        == start_time + datetime.timedelta(seconds=1)
    assert assoc4.time_range.end_timestamp \
        == start_time + datetime.timedelta(seconds=7)

    assert len(association_set_one2one) == 1
    assoc5 = list(association_set_one2one)[0]
    # assoc5 should be equal to assoc1
    assert set(assoc5.objects) == {tracks[0], tracks[1]}
    assert assoc5.time_range.start_timestamp \
        == start_time + datetime.timedelta(seconds=1)
    assert assoc5.time_range.end_timestamp \
        == start_time + datetime.timedelta(seconds=6)


def test_euclidiantracktotruth(tracks):
    associator = TrackToTruth(
        association_threshold=10,
        consec_pairs_confirm=3,
        consec_misses_end=2)
    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    association_set = associator.associate_tracks(
        truth_set={tracks[0]}, tracks_set={tracks[2], tracks[1], tracks[3]})

    assert len(association_set.associations) == 1
    assoc = list(association_set.associations)[0]
    assert set(assoc.objects) == {tracks[0], tracks[1]}
    assert assoc.time_range.start_timestamp == start_time + datetime.timedelta(
        seconds=1)
    assert assoc.time_range.end_timestamp == start_time + datetime.timedelta(
        seconds=6)


def test_trackidbased():
    associator = TrackIDbased()
    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    tracksA = [Track(states=[State(state_vector=[[i]],
                                   timestamp=start_time),
                             State(state_vector=[[i]],
                                   timestamp=start_time + datetime.timedelta(
                                       seconds=1)),
                             State(state_vector=[[i]],
                                   timestamp=start_time + datetime.timedelta(
                                       seconds=2))],
                     id=f"id{i}")
               for i in range(0, 5)]

    tracksB = [Track(states=[State(state_vector=[[i]])],
                     id=f"id{i}")
               for i in range(0, 5)]

    tracksC = [Track(states=[State(state_vector=[[i]],
                                   timestamp=start_time + datetime.timedelta(
                                       seconds=4)),
                             State(state_vector=[[i]],
                                   timestamp=start_time + datetime.timedelta(
                                       seconds=5))],
                     id=f"id{i}")
               for i in range(0, 5)]

    truths = [GroundTruthPath(
        states=[GroundTruthState(state_vector=[[i+0.5]],
                                 timestamp=start_time + datetime.timedelta(seconds=1)),
                GroundTruthState(state_vector=[[i+0.5]],
                                 timestamp=start_time + datetime.timedelta(seconds=2)),
                GroundTruthState(state_vector=[[i + 0.5]],
                                 timestamp=start_time + datetime.timedelta(seconds=3))
                ],
        id=f"id{i}")
        for i in range(1, 6)]

    association_setA = associator.associate_tracks(tracksA, truths)
    assert len(association_setA.associations) == 4

    assocA = list(association_setA.associations)[0]
    assoc_track = assocA.objects[0]
    assoc_truth = assocA.objects[1]
    assert assoc_track.id == assoc_truth.id

    association_setB = associator.associate_tracks(tracksB, truths)
    assocB = list(association_setB.associations)[0]
    assert isinstance(assocB, Association)

    association_setC = associator.associate_tracks(tracksC, truths)
    assocC = list(association_setC.associations)[0]
    assert isinstance(assocC, Association)


def test_clear_mot(tracks: List[Track]):

    associator = ClearMotAssociator(measure=Euclidean(mapping=[0, 1]), association_threshold=3.0)

    track = tracks[0]
    truth_track = tracks[1]

    association_set = associator.associate_tracks({track}, {truth_track})

    assoc = list(association_set.associations)[0]
    assoc_track = assoc.objects[0]
    assoc_truth = assoc.objects[1]
    assert assoc_track.id == track.id
    assert assoc_truth.id == truth_track.id

    assert assoc.time_range.start == datetime.datetime(2019, 1, 1, 14, 0, 1)
    assert assoc.time_range.end == datetime.datetime(2019, 1, 1, 14, 0, 6)


def test_clear_mot_two_truths_crossing(tracks: List[Track]):

    associator = ClearMotAssociator(measure=Euclidean(mapping=[0, 1]), association_threshold=2.0)

    # truths cross each other
    truth_tracks = {tracks[0], tracks[6]}

    estimated_tracks = {tracks[7], tracks[2], tracks[3]}

    association_set = associator.associate_tracks(estimated_tracks, truth_tracks)

    # sort by truth-ID
    associations = sorted(list(association_set.associations),
                          key=lambda assoc: assoc.objects[1].id)

    assert len(associations) == 2

    assocA = associations[0]
    assert assocA.objects[0].id == "Track-8"
    assert assocA.objects[1].id == "Track-1"
    assert assocA.time_range.start == datetime.datetime(2019, 1, 1, 14, 0, 0)
    assert assocA.time_range.end == datetime.datetime(2019, 1, 1, 14, 0, 5)

    assocB = associations[1]
    assert assocB.objects[0].id == "Track-8"
    assert assocB.objects[1].id == "Track-7"
    assert assocB.time_range.start == datetime.datetime(2019, 1, 1, 14, 0, 6)
    assert assocB.time_range.end == datetime.datetime(2019, 1, 1, 14, 0, 9)


def test_clear_mot_measure_dimensions(tracks: List[Track]):

    track = tracks[0]
    truth_track = tracks[4]

    associator = ClearMotAssociator(measure=Euclidean(mapping=[0, 1]), association_threshold=3.0)

    association_set = associator.associate_tracks({track}, {truth_track})

    assert not len(association_set), \
        "Track-1 and Track-5 should not be associated when looking at all" +\
        " dimensions of the state vector."
    
    associator = ClearMotAssociator(measure=Euclidean(mapping=[0]), association_threshold=3.0)
    
    association_set = associator.associate_tracks({track}, {truth_track})
    
    assoc = list(association_set.associations)[0]
    assoc_track = assoc.objects[0]
    assoc_truth = assoc.objects[1]
    assert assoc_track.id == track.id
    assert assoc_truth.id == truth_track.id
