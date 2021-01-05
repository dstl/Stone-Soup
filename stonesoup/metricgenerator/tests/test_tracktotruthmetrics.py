import datetime

import pytest

from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.update import StateUpdate
from ..tracktotruthmetrics import SIAPMetrics
from ...metricgenerator.manager import SimpleManager
from ...types.association import TimeRangeAssociation, AssociationSet
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.metric import TimeRangeMetric
from ...types.state import State
from ...types.time import TimeRange
from ...types.track import Track


def metric_generators():
    """A list of metric generators to be used in tests"""
    return [SIAPMetrics(position_mapping=[0, 2],
                        velocity_mapping=[1, 3],
                        truth_id='identify',
                        track_id='ident')]


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_num_tracks_tracks(generator):
    manager = SimpleManager()
    metric = SIAPMetrics(position_mapping=[0, 1])
    tstart = datetime.datetime.now()
    truths = {
        GroundTruthPath(
            states=[GroundTruthState([[i], [0], [0], [0]],
                                     timestamp=tstart + datetime.timedelta(seconds=i))
                    for i in range(5)]),
        GroundTruthPath(
            states=[GroundTruthState([[i], [0], [i], [0]],
                                     timestamp=tstart + datetime.timedelta(seconds=i))
                    for i in range(5)])
    }
    tracks = {
        Track(
            states=[State([[i], [0], [1], [0]],
                          timestamp=tstart + datetime.timedelta(seconds=i))
                    for i in range(5)]),
        Track(
            states=[State([[i], [0], [-1], [0]],
                          timestamp=tstart + datetime.timedelta(seconds=i))
                    for i in range(5)]),
        Track(
            states=[State([[i], [0], [i + 1], [0]],
                          timestamp=tstart + datetime.timedelta(seconds=i))
                    for i in range(5)])
    }

    manager.groundtruth_paths = truths
    manager.tracks = tracks

    num_tracks = metric.num_tracks(manager)
    num_truths = metric.num_truths(manager)

    metrics = {num_tracks, num_truths}
    for met in metrics:
        assert isinstance(met, TimeRangeMetric)
        assert met.time_range.start_timestamp == tstart
        assert met.time_range.end_timestamp == tstart + datetime.timedelta(seconds=4)
        assert met.generator == metric
    assert num_tracks.title == 'SIAP nt'
    assert num_tracks.value == 3
    assert num_truths.title == 'SIAP nj'
    assert num_truths.value == 2


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
@pytest.mark.parametrize("mapping", ['position_mapping', 'velocity_mapping'])
def test_assoc_distances_sum_t(generator, mapping):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(
        states=[GroundTruthState([[i], [i], [0], [0]],
                                 timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)]
    )
    track1 = Track(
        states=[State([[i], [i], [1], [1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)]
    )
    track2 = Track(
        states=[State([[i], [i], [-1], [-1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)]
    )
    assoc1 = TimeRangeAssociation(
        {truth, track1},
        time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))
    )

    assoc2 = TimeRangeAssociation(
        {truth, track2},
        time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))
    )
    associations = {assoc1}

    manager.groundtruth_paths = {truth}
    manager.tracks = {track1, track2}
    manager.association_set = AssociationSet(associations)

    for i in range(5):
        distance_sum = generator._assoc_distances_sum_t(manager,
                                                        tstart + datetime.timedelta(seconds=i),
                                                        getattr(generator, mapping),
                                                        1)
        assert distance_sum == 1

    associations = {assoc1, assoc2}
    manager.association_set = AssociationSet(associations)

    for i in range(5):
        distance_sum = generator._assoc_distances_sum_t(manager,
                                                        tstart + datetime.timedelta(seconds=i),
                                                        getattr(generator, mapping),
                                                        1)
        assert distance_sum == 2


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_j_t(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truths = {GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]],
                         timestamp=tstart + datetime.timedelta(seconds=i))
        for i in range(5)]),
        GroundTruthPath(states=[
            GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
                seconds=i))
            for i in range(3)])}
    manager.groundtruth_paths = truths

    assert generator._j_t(manager, tstart + datetime.timedelta(seconds=1)) == 2
    assert generator._j_t(manager, tstart + datetime.timedelta(seconds=4)) == 1


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_j_sum(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truths = {GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(5)]),
        GroundTruthPath(states=[
            GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
                seconds=i))
            for i in range(3)])
    }
    manager.groundtruth_paths = truths

    assert generator._j_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                      for i in range(7)]) == 8


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_jt(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truth1 = GroundTruthPath(
        states=[GroundTruthState([[i], [0], [0], [0]],
                                 timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(20)]
    )
    truth2 = GroundTruthPath(
        states=[GroundTruthState([[i], [0], [10], [0]],
                                 timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(20)]
    )
    track1 = Track(
        states=[State([[i], [0], [1], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(20)]
    )
    track2 = Track(
        states=[State([[i], [0], [10.5], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(20)]
    )
    assocs = [TimeRangeAssociation(
        {truth1, track1},
        time_range=TimeRange(start_timestamp=tstart + datetime.timedelta(seconds=5),
                             end_timestamp=tstart + datetime.timedelta(seconds=5 + i)))
        for i in range(15)]

    assocs.extend([TimeRangeAssociation(
        {truth2, track2},
        time_range=TimeRange(start_timestamp=tstart + datetime.timedelta(seconds=10),
                             end_timestamp=tstart + datetime.timedelta(seconds=10 + i)))
        for i in range(10)])

    manager.groundtruth_paths = {truth1, truth2}
    manager.tracks = {track1, track2}
    manager.association_set = AssociationSet(assocs)

    timestamps = [tstart + datetime.timedelta(seconds=i) for i in range(20)]

    count = 0

    # test _jt_t
    for timestamp in timestamps:
        value = generator._jt_t(manager, timestamp)
        if timestamps.index(timestamp) < 5:
            assert value == 0
        elif timestamps.index(timestamp) < 10:
            assert value == 1
        else:
            assert value == 2
        count += value

    # test _jt_sum
    assert generator._jt_sum(manager, timestamps) == count


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test__na(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(5)])
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)])
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    associations = {TimeRangeAssociation({truth, track1}, time_range=TimeRange(
        start_timestamp=tstart,
        end_timestamp=tstart + datetime.timedelta(seconds=4))),
                    TimeRangeAssociation({truth, track2}, time_range=TimeRange(
                        start_timestamp=tstart,
                        end_timestamp=tstart + datetime.timedelta(seconds=2)))}
    manager.tracks = {track1, track2}
    manager.association_set = AssociationSet(associations)

    # Test _na_t
    assert generator._na_t(manager, tstart + datetime.timedelta(seconds=1)) == 2
    assert generator._na_t(manager, tstart + datetime.timedelta(seconds=4)) == 1
    assert generator._na_t(manager, tstart + datetime.timedelta(seconds=7)) == 0

    # Test _na_sum

    assert generator._na_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                       for i in range(4)]) == 7


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_n(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)])
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(2, 7)])
    manager.tracks = {track1, track2, track3}

    # test _n_t
    assert generator._n_t(manager, tstart + datetime.timedelta(seconds=2)) == 3
    assert generator._n_t(manager, tstart + datetime.timedelta(seconds=4)) == 2
    assert generator._n_t(manager, tstart + datetime.timedelta(seconds=6)) == 1
    assert generator._n_t(manager, tstart + datetime.timedelta(seconds=10)) == 0

    # test _n_sum
    assert generator._n_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                      for i in range(5)]) == 11


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_tt_j(generator):
    manager = SimpleManager()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(20)])
    # Idea is track 1, then no track then 2 then 2 and 3 then 3
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    associations = {TimeRangeAssociation({truth, track1}, time_range=TimeRange(
        start_timestamp=tstart,
        end_timestamp=tstart + datetime.timedelta(seconds=2))),
                    TimeRangeAssociation({truth, track2}, time_range=TimeRange(
                        start_timestamp=tstart + datetime.timedelta(seconds=5),
                        end_timestamp=tstart + datetime.timedelta(seconds=9))),
                    TimeRangeAssociation({truth, track3}, time_range=TimeRange(
                        start_timestamp=tstart + datetime.timedelta(seconds=7),
                        end_timestamp=tstart + datetime.timedelta(
                            seconds=14)))}
    manager.tracks = {track1, track2, track3}
    manager.association_set = AssociationSet(associations)

    assert generator._tt_j(manager, truth) == datetime.timedelta(seconds=11)


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_nu_j(generator):
    manager = SimpleManager()
    tstart = datetime.datetime(2020, 1, 1, 0)
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])
    # Single track
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    # Overlapping tracks
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    track4 = Track(
        states=[State([[4], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(13, 20)])
    # Long track with shorter unnecessary one internal
    track5 = Track(
        states=[State([[5], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(18, 28)])
    track6 = Track(
        states=[State([[6], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(22, 26)])
    # 2 tracks covered by same range as 1
    track7 = Track(
        states=[State([[7], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 40)])
    track8 = Track(
        states=[State([[8], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 35)])
    track9 = Track(
        states=[State([[9], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(35, 40)])

    manager.tracks = {track1, track2, track3, track4, track5, track6, track7,
                      track8, track9}
    manager.groundtruth_paths = {truth}
    associations = {TimeRangeAssociation({truth, track}, time_range=TimeRange(
        start_timestamp=min([state.timestamp for state in track.states]),
        end_timestamp=max([state.timestamp for state in track.states])))
                    for track in manager.tracks}
    manager.association_set = AssociationSet(associations)

    # test nu_j
    # The fewest tracks to cover the whole length should be tracks 1,2,3,5 & 7
    assert generator._nu_j(manager, truth) == 6

    # test _tl_j
    # longest track on truth is track 7
    assert generator._tl_j(manager, truth) == datetime.timedelta(seconds=9)

    # test _r
    assert generator._r(manager) == 5 / 33


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_t_j(generator):
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])

    assert generator._t_j(truth) == datetime.timedelta(seconds=39)


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_get_track_id(generator):
    tstart = datetime.datetime.now()
    track = Track([
        StateUpdate([0], hypothesis=SingleHypothesis(None, Detection([0])), timestamp=tstart),
        StateUpdate([0], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'colour': 'red'})),
                    timestamp=tstart + datetime.timedelta(seconds=1)),
        StateUpdate([0], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=2)),
        StateUpdate([0],
                    hypothesis=SingleHypothesis(None, Detection([0], metadata={'ident': 'enemy'})),
                    timestamp=tstart + datetime.timedelta(seconds=3))])

    assert generator._get_track_id(track, tstart) is None
    assert generator._get_track_id(track, tstart + datetime.timedelta(seconds=1)) is None
    assert generator._get_track_id(track, tstart + datetime.timedelta(seconds=2)) == 'ally'
    assert generator._get_track_id(track, tstart + datetime.timedelta(seconds=3)) == 'enemy'


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_constant_id(generator):
    tstart = datetime.datetime.now()
    manager = SimpleManager()

    truth1 = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(5)])
    truth2 = GroundTruthPath(states=[
        GroundTruthState([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(5)])
    truth3 = GroundTruthPath(states=[
        GroundTruthState([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(5)])
    truth4 = GroundTruthPath(states=[
        GroundTruthState([[4], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(5)])

    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)],
        init_metadata={'ident': 'ally'})

    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)],
        init_metadata={'ident': 'enemy'})

    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)])

    manager.tracks = {track1, track2, track3}
    manager.groundtruth_paths = {truth1, truth2, truth3, truth4}

    associations = {
        TimeRangeAssociation({truth1, track1}, time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))),
        TimeRangeAssociation({truth2, track2}, time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))),
        TimeRangeAssociation({truth3, track3}, time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))),
        TimeRangeAssociation({truth4, track1}, time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5))),
        TimeRangeAssociation({truth4, track2}, time_range=TimeRange(
            start_timestamp=tstart,
            end_timestamp=tstart + datetime.timedelta(seconds=5)))
    }
    manager.association_set = AssociationSet(associations)

    for i in range(5):
        timestamp = tstart + datetime.timedelta(seconds=i)
        assert generator._ju_t(manager, timestamp) == 1
        assert generator._jc_t(manager, timestamp) == 1
        assert generator._ji_t(manager, timestamp) == 1
        assert generator._ja_t(manager, timestamp) == 1

    assert generator._ju_sum(manager, manager.list_timestamps()) == 5
    assert generator._jc_sum(manager, manager.list_timestamps()) == 5
    assert generator._ji_sum(manager, manager.list_timestamps()) == 5
    assert generator._ja_sum(manager, manager.list_timestamps()) == 5


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_variable_id(generator):
    tstart = datetime.datetime.now()
    manager = SimpleManager()

    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(5)])

    vary_track = Track([
        StateUpdate([1], hypothesis=SingleHypothesis(None, Detection([0])), timestamp=tstart),
        StateUpdate([1], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'colour': 'red'})),
                    timestamp=tstart + datetime.timedelta(seconds=1)),
        StateUpdate([1], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=2)),
        StateUpdate([1],
                    hypothesis=SingleHypothesis(None, Detection([0], metadata={'ident': 'enemy'})),
                    timestamp=tstart + datetime.timedelta(seconds=3)),
        StateUpdate([1],
                    hypothesis=SingleHypothesis(None, Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=4))])
    correct_track = Track([
        StateUpdate([1], hypothesis=SingleHypothesis(None, Detection([0])), timestamp=tstart),
        StateUpdate([1], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=1)),
        StateUpdate([1], hypothesis=SingleHypothesis(None,
                                                     Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=2)),
        StateUpdate([1],
                    hypothesis=SingleHypothesis(None, Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=3)),
        StateUpdate([1],
                    hypothesis=SingleHypothesis(None, Detection([0], metadata={'ident': 'ally'})),
                    timestamp=tstart + datetime.timedelta(seconds=4))])

    tend = tstart + datetime.timedelta(seconds=4)
    manager.tracks = {vary_track, correct_track}
    manager.groundtruth_paths = {truth}

    associations = {
        TimeRangeAssociation({truth, vary_track}, time_range=TimeRange(tstart, tend)),
        TimeRangeAssociation({truth, correct_track}, time_range=TimeRange(tstart, tend))
    }

    manager.association_set = AssociationSet(associations)

    timestamp = tstart
    assert generator._ju_t(manager, timestamp) == 1
    assert generator._jc_t(manager, timestamp) == 0
    assert generator._ji_t(manager, timestamp) == 0
    assert generator._ja_t(manager, timestamp) == 0

    timestamp = tstart + datetime.timedelta(seconds=1)
    assert generator._ju_t(manager, timestamp) == 0
    assert generator._jc_t(manager, timestamp) == 0
    assert generator._ji_t(manager, timestamp) == 0
    assert generator._ja_t(manager, timestamp) == 1

    timestamp = tstart + datetime.timedelta(seconds=2)
    assert generator._ju_t(manager, timestamp) == 0
    assert generator._jc_t(manager, timestamp) == 1
    assert generator._ji_t(manager, timestamp) == 0
    assert generator._ja_t(manager, timestamp) == 0

    timestamp = tstart + datetime.timedelta(seconds=3)
    assert generator._ju_t(manager, timestamp) == 0
    assert generator._jc_t(manager, timestamp) == 0
    assert generator._ji_t(manager, timestamp) == 0
    assert generator._ja_t(manager, timestamp) == 1

    timestamp = tstart + datetime.timedelta(seconds=4)
    assert generator._ju_t(manager, timestamp) == 0
    assert generator._jc_t(manager, timestamp) == 1
    assert generator._ji_t(manager, timestamp) == 0
    assert generator._ja_t(manager, timestamp) == 0

    assert generator._ju_sum(manager, manager.list_timestamps()) == 1
    assert generator._jc_sum(manager, manager.list_timestamps()) == 2
    assert generator._ji_sum(manager, manager.list_timestamps()) == 0
    assert generator._ja_sum(manager, manager.list_timestamps()) == 2


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_compute_metric(generator):
    manager = SimpleManager()
    # Create truth, tracks and associations, same as test_nu_j
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(40)])
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)],
        init_metadata={'ident': 'ally'})
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)],
        init_metadata={'ident': 'enemy'})
    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    track4 = Track(
        states=[State([[4], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(13, 20)],
        init_metadata={'ident': 'ally'})
    track5 = Track(
        states=[State([[5], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(18, 28)],
        init_metadata={'something': 'ally'})
    track6 = Track(
        states=[State([[6], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(22, 26)])
    track7 = Track(
        states=[State([[7], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 40)],
        init_metadata={'ident': None})
    track8 = Track(
        states=[State([[8], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 35)])
    track9 = Track(
        states=[State([[9], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(35, 40)])
    manager.tracks = {track1, track2, track3, track4, track5, track6, track7,
                      track8, track9}
    manager.groundtruth_paths = {truth}
    associations = {TimeRangeAssociation({truth, track}, time_range=TimeRange(
        start_timestamp=min([state.timestamp for state in track.states]),
        end_timestamp=max([state.timestamp for state in track.states])))
                    for track in manager.tracks}
    manager.association_set = AssociationSet(associations)

    metrics = generator.compute_metric(manager)
    tend = tstart + datetime.timedelta(seconds=39)

    assert len(metrics) == 20

    c = metrics[0]
    assert c.title == "SIAP C"
    assert c.value == generator._jt_sum(manager,
                                        manager.list_timestamps()) / (
               generator._j_sum(manager, manager.list_timestamps()))
    assert c.time_range.start_timestamp == tstart
    assert c.time_range.end_timestamp == tend
    assert c.generator == generator

    a = metrics[1]
    assert a.title == "SIAP A"
    assert a.value == generator._na_sum(manager,
                                        manager.list_timestamps()) / (
               generator._jt_sum(manager, manager.list_timestamps()))
    assert a.time_range.start_timestamp == tstart
    assert a.time_range.end_timestamp == tend
    assert a.generator == generator

    s = metrics[2]
    assert s.title == "SIAP S"
    assert s.value == sum([generator._n_t(manager, timestamp) -
                           generator._na_t(manager, timestamp)
                           for timestamp in manager.list_timestamps()]) / (
               generator._n_sum(manager, manager.list_timestamps()))
    assert s.time_range.start_timestamp == tstart
    assert s.time_range.end_timestamp == tend
    assert s.generator == generator

    lt = metrics[3]
    assert lt.title == "SIAP LT"
    assert lt.value == 1 / generator._r(manager)
    assert lt.time_range.start_timestamp == tstart
    assert lt.time_range.end_timestamp == tend
    assert lt.generator == generator

    ls = metrics[4]
    assert ls.title == "SIAP LS"
    assert ls.value == sum([generator._tl_j(manager, truth).total_seconds()
                            for truth in manager.groundtruth_paths]) / sum(
        [generator._t_j(truth).total_seconds()
         for truth in manager.groundtruth_paths])
    assert ls.time_range.start_timestamp == tstart
    assert ls.time_range.end_timestamp == tend
    assert ls.generator == generator

    nt = metrics[5]
    assert nt.title == "SIAP nt"
    assert nt.value == len({track1, track2, track3, track4, track5, track6, track7, track8,
                            track9})
    assert nt.time_range.start_timestamp == tstart
    assert nt.time_range.end_timestamp == tend
    assert nt.generator == generator

    nj = metrics[6]
    assert nj.title == "SIAP nj"
    assert nj.value == len({truth})
    assert nj.time_range.start_timestamp == tstart
    assert nj.time_range.end_timestamp == tend
    assert nj.generator == generator

    pa = metrics[7]
    assert pa.title == "SIAP PA"
    assert pa.value == \
           sum(generator._assoc_distances_sum_t(manager,
                                                timestamp,
                                                generator.position_mapping,
                                                1)
               for timestamp in manager.list_timestamps()) \
           / generator._na_sum(manager, manager.list_timestamps())
    assert pa.time_range.start_timestamp == tstart
    assert pa.time_range.end_timestamp == tend
    assert pa.generator == generator

    tpa = metrics[8]
    assert tpa.title == "T PA"
    for i in range(len(manager.list_timestamps())):
        t_metric = tpa.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP PA at timestamp"
        numerator = generator._assoc_distances_sum_t(manager,
                                                     timestamp,
                                                     generator.position_mapping,
                                                     1)
        if generator._na_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._na_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert pa.time_range.start_timestamp == tstart
    assert pa.time_range.end_timestamp == tend
    assert pa.generator == generator

    va = metrics[9]
    assert va.title == "SIAP VA"
    numerator = sum(generator._assoc_distances_sum_t(manager,
                                                     timestamp,
                                                     generator.velocity_mapping,
                                                     1)
                    for timestamp in manager.list_timestamps())
    assert va.value == numerator / generator._na_sum(manager, manager.list_timestamps())
    assert va.time_range.start_timestamp == tstart
    assert va.time_range.end_timestamp == tend
    assert va.generator == generator

    tva = metrics[10]
    assert tva.title == "T VA"
    for i in range(len(manager.list_timestamps())):
        t_metric = tva.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP VA at timestamp"
        numerator = generator._assoc_distances_sum_t(manager,
                                                     timestamp,
                                                     generator.velocity_mapping,
                                                     1)
        if generator._na_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._na_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert tva.time_range.start_timestamp == tstart
    assert tva.time_range.end_timestamp == tend
    assert tva.generator == generator

    tc = metrics[11]
    assert tc.title == "T C"
    for i in range(len(manager.list_timestamps())):
        t_metric = tc.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP C at timestamp"
        numerator = generator._jt_t(manager, timestamp)
        if generator._na_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._j_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert tc.time_range.start_timestamp == tstart
    assert tc.time_range.end_timestamp == tend
    assert tc.generator == generator

    ta = metrics[12]
    assert ta.title == "T A"
    for i in range(len(manager.list_timestamps())):
        t_metric = ta.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP A at timestamp"
        numerator = generator._na_t(manager, timestamp)
        if generator._na_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._jt_t(manager, timestamp)
        else:
            assert t_metric.value == 1
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert ta.time_range.start_timestamp == tstart
    assert ta.time_range.end_timestamp == tend
    assert ta.generator == generator

    ts = metrics[13]
    assert ts.title == "T S"
    for i in range(len(manager.list_timestamps())):
        t_metric = ts.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP S at timestamp"
        numerator = generator._n_t(manager, timestamp) - generator._na_t(manager, timestamp)
        if generator._na_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._n_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert ts.time_range.start_timestamp == tstart
    assert ts.time_range.end_timestamp == tend
    assert ts.generator == generator

    cid = metrics[14]
    assert cid.title == "SIAP CID"
    assert cid.value == \
           sum({generator._jt_t(manager, timestamp) - generator._ju_t(manager, timestamp)
                for timestamp in manager.list_timestamps()}) \
           / generator._jt_sum(manager, manager.list_timestamps())
    assert cid.time_range.start_timestamp == tstart
    assert cid.time_range.end_timestamp == tend
    assert cid.generator == generator

    tcid = metrics[15]
    assert tcid.title == "T CID"
    for i in range(len(manager.list_timestamps())):
        t_metric = tcid.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP CID at timestamp"
        numerator = generator._jt_t(manager, timestamp) - generator._ju_t(manager, timestamp)
        if generator._jt_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._jt_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert tcid.time_range.start_timestamp == tstart
    assert tcid.time_range.end_timestamp == tend
    assert tcid.generator == generator

    idc = metrics[16]
    assert idc.title == "SIAP IDC"
    assert idc.value == generator._jc_sum(manager, manager.list_timestamps()) \
           / generator._jt_sum(manager, manager.list_timestamps())
    assert idc.time_range.start_timestamp == tstart
    assert idc.time_range.end_timestamp == tend
    assert idc.generator == generator

    ida = metrics[17]
    assert ida.title == "SIAP IDA"
    assert ida.value == generator._ja_sum(manager, manager.list_timestamps()) \
           / generator._jt_sum(manager, manager.list_timestamps())
    assert ida.time_range.start_timestamp == tstart
    assert ida.time_range.end_timestamp == tend
    assert ida.generator == generator

    tidc = metrics[18]
    assert tidc.title == "T IDC"
    for i in range(len(manager.list_timestamps())):
        t_metric = tidc.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP IDC at timestamp"
        numerator = generator._jc_t(manager, timestamp)
        if generator._jt_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._jt_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert tidc.time_range.start_timestamp == tstart
    assert tidc.time_range.end_timestamp == tend
    assert tidc.generator == generator

    tida = metrics[19]
    assert tida.title == "T IDA"
    for i in range(len(manager.list_timestamps())):
        t_metric = tida.value[i]
        timestamp = manager.list_timestamps()[i]
        assert t_metric.title == "SIAP IDA at timestamp"
        numerator = generator._ja_t(manager, timestamp)
        if generator._jt_t(manager, timestamp) != 0:
            assert t_metric.value == numerator / generator._jt_t(manager, timestamp)
        else:
            assert t_metric.value == 0
        assert t_metric.timestamp == timestamp
        assert t_metric.generator == generator
    assert tida.time_range.start_timestamp == tstart
    assert tida.time_range.end_timestamp == tend
    assert tida.generator == generator


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_no_truth_divide_by_zero(generator):
    manager = SimpleManager()
    # Create truth, tracks and associations, same as test_nu_j
    tstart = datetime.datetime.now()
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    track4 = Track(
        states=[State([[4], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(13, 20)])
    track5 = Track(
        states=[State([[5], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(18, 28)])
    track6 = Track(
        states=[State([[6], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(22, 26)])
    track7 = Track(
        states=[State([[7], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 40)])
    track8 = Track(
        states=[State([[8], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 35)])
    track9 = Track(
        states=[State([[9], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(35, 40)])
    manager.tracks = {track1, track2, track3, track4, track5, track6, track7,
                      track8, track9}
    manager.groundtruth_paths = set()
    associations = set()
    manager.association_set = AssociationSet(associations)

    with pytest.warns(UserWarning) as warning:
        metrics = generator.compute_metric(manager)

    assert warning[0].message.args[0] == "No truth to generate SIAP Metric"

    assert len(metrics) == 20

    print([metric.title for metric in metrics])


@pytest.mark.parametrize("generator", metric_generators(), ids=["SIAP"])
def test_no_track_divide_by_zero(generator):
    manager = SimpleManager()
    # Create truth, tracks and associations, same as test_nu_j
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])
    manager.tracks = set()
    manager.groundtruth_paths = {truth}
    associations = {TimeRangeAssociation({truth}, time_range=TimeRange(
        start_timestamp=min([state.timestamp for state in track.states]),
        end_timestamp=max([state.timestamp for state in track.states])))
                    for track in manager.tracks}
    manager.association_set = AssociationSet(associations)

    with pytest.warns(UserWarning) as warning:
        metrics = generator.compute_metric(manager)

    assert warning[0].message.args[0] == "No tracks to generate SIAP Metric"

    assert len(metrics) == 20


def test_absent_params():
    manager = SimpleManager()
    generator = SIAPMetrics()

    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(
            seconds=i), metadata={'identify': 'ally'})
        for i in range(40)])
    track1 = Track(
        states=[State([[1], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)],
        init_metadata={'ident': 'ally'})
    track2 = Track(
        states=[State([[2], [0], [0], [0]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)],
        init_metadata={'ident': 'enemy'})

    manager.tracks = {track1, track2}
    manager.groundtruth_paths = {truth}
    associations = {TimeRangeAssociation({truth, track}, time_range=TimeRange(
        start_timestamp=min([state.timestamp for state in track.states]),
        end_timestamp=max([state.timestamp for state in track.states])))
                    for track in manager.tracks}
    manager.association_set = AssociationSet(associations)

    metrics = generator.compute_metric(manager)

    metric_names = {'SIAP C', 'SIAP A', 'SIAP S', 'SIAP LT', 'SIAP LS', 'SIAP nt', 'SIAP nj',
                    'SIAP PA', 'T PA', 'SIAP VA', 'T VA', 'T C', 'T A', 'T S', 'SIAP CID',
                    'T CID', 'SIAP IDC', 'SIAP IDA', 'T IDC', 'T IDA'}

    absent_names = {'SIAP PA', 'T PA', 'SIAP VA', 'T VA', 'SIAP CID', 'T CID', 'SIAP IDC',
                    'SIAP IDA', 'T IDC', 'T IDA'}

    assert all(metric.title in (metric_names - absent_names) for metric in metrics)

    generator = SIAPMetrics(track_id='ident')

    metrics = generator.compute_metric(manager)

    absent_names = {'SIAP PA', 'T PA', 'SIAP VA', 'T VA', 'SIAP IDC', 'T IDC', 'SIAP IDA', 'T IDA'}

    assert all(metric.title in (metric_names - absent_names) for metric in metrics)
