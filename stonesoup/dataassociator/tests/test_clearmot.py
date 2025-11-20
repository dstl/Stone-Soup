import datetime

import pytest

from ...measures.state import Euclidean
from ...types.state import State
from ...types.track import Track
from ..clearmot import ClearMotAssociator


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
                                                    timestamp=start_time +
                                                    datetime.timedelta(seconds=i))
                                              for i in range(0, 5)]
                        + [State(state_vector=[[i], [10-i]],
                                 timestamp=start_time + datetime.timedelta(seconds=i))
                           for i in range(5, 10)]))

    return tracks


def test_clear_mot(tracks: list[Track]):

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


def test_clear_mot_two_truths_crossing(tracks: list[Track]):

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


def test_clear_mot_measure_dimensions(tracks: list[Track]):

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
