import datetime

from ..tracktotruthmetrics import SIAPMetrics
from ...types.association import TimeRangeAssociation, AssociationSet
from ...types.track import Track
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...metricgenerator.manager import SimpleManager
from ...types.time import TimeRange
from ...types.state import State


def test_j_t():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    truths = {GroundTruthPath(states=[
        GroundTruthState([[1]],
                         timestamp=tstart + datetime.timedelta(seconds=i))
        for i in range(5)]),
        GroundTruthPath(states=[
            GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
                seconds=i))
            for i in range(3)])}
    manager.groundtruth_paths = truths

    assert metric._j_t(manager, tstart + datetime.timedelta(seconds=1)) == 2
    assert metric._j_t(manager, tstart + datetime.timedelta(seconds=4)) == 1


def test_j_sum():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    truths = {GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(5)]),
        GroundTruthPath(states=[
            GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
                seconds=i))
            for i in range(3)])
    }
    manager.groundtruth_paths = truths

    assert metric._j_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                   for i in range(7)]) == 8


def test_jt():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    associations = {TimeRangeAssociation([], time_range=TimeRange(
        start_timestamp=tstart,
        end_timestamp=tstart + datetime.timedelta(seconds=3))),
                    TimeRangeAssociation([], time_range=TimeRange(
                        start_timestamp=tstart,
                        end_timestamp=tstart + datetime.timedelta(seconds=5)))}
    manager.association_set = AssociationSet(associations)

    # test _jt_t
    assert metric._jt_t(manager, tstart) == 2
    assert metric._jt_t(manager, tstart + datetime.timedelta(seconds=5)) == 1

    # test _jt_sum
    assert metric._jt_sum(manager,
                          [tstart + datetime.timedelta(seconds=i)
                           for i in range(5)]) == 9


def test__na():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(5)])
    track1 = Track(
        states=[State([[1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)])
    track2 = Track(
        states=[State([[2]], timestamp=tstart + datetime.timedelta(seconds=i))
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
    assert metric._na_t(manager, tstart + datetime.timedelta(seconds=1)) == 2
    assert metric._na_t(manager, tstart + datetime.timedelta(seconds=4)) == 1
    assert metric._na_t(manager, tstart + datetime.timedelta(seconds=7)) == 0

    # Test _na_sum

    assert metric._na_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                    for i in range(4)]) == 7


def test_n():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    track1 = Track(
        states=[State([[1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5)])
    track2 = Track(
        states=[State([[2]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track3 = Track(
        states=[State([[3]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(2, 7)])
    manager.tracks = {track1, track2, track3}

    # test _n_t
    assert metric._n_t(manager, tstart + datetime.timedelta(seconds=2)) == 3
    assert metric._n_t(manager, tstart + datetime.timedelta(seconds=4)) == 2
    assert metric._n_t(manager, tstart + datetime.timedelta(seconds=6)) == 1
    assert metric._n_t(manager, tstart + datetime.timedelta(seconds=10)) == 0

    # test _n_sum
    assert metric._n_sum(manager, [tstart + datetime.timedelta(seconds=i)
                                   for i in range(5)]) == 11


def test_tt_j():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(20)])
    # Idea is track 1, then no track then 2 then 2 and 3 then 3
    track1 = Track(
        states=[State([[1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track2 = Track(
        states=[State([[2]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3]], timestamp=tstart + datetime.timedelta(seconds=i))
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

    assert metric._tt_j(manager, truth) == datetime.timedelta(seconds=11)


def test_nu_j():
    manager = SimpleManager()
    metric = SIAPMetrics()
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])
    # Single track
    track1 = Track(
        states=[State([[1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    # Overlapping tracks
    track2 = Track(
        states=[State([[2]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    track4 = Track(
        states=[State([[4]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(13, 20)])
    # Long track with shorter unnecessary one internal
    track5 = Track(
        states=[State([[5]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(18, 28)])
    track6 = Track(
        states=[State([[6]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(22, 26)])
    # 2 tracks covered by same range as 1
    track7 = Track(
        states=[State([[7]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 40)])
    track8 = Track(
        states=[State([[8]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 35)])
    track9 = Track(
        states=[State([[9]], timestamp=tstart + datetime.timedelta(seconds=i))
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
    assert metric._nu_j(manager, truth) == 6

    # test _tl_j
    # longest track on truth is track 7
    assert metric._tl_j(manager, truth) == datetime.timedelta(seconds=9)

    # test _r
    assert metric._r(manager) == 5 / 33


def test_t_j():
    metric = SIAPMetrics()

    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])

    assert metric._t_j(truth) == datetime.timedelta(seconds=39)


def test_compute_metric():
    manager = SimpleManager()
    generator = SIAPMetrics()
    # Create truth, tracks and associations, same as test_nu_j
    tstart = datetime.datetime.now()
    truth = GroundTruthPath(states=[
        GroundTruthState([[1]], timestamp=tstart + datetime.timedelta(
            seconds=i))
        for i in range(40)])
    track1 = Track(
        states=[State([[1]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(3)])
    track2 = Track(
        states=[State([[2]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(5, 10)])
    track3 = Track(
        states=[State([[3]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(7, 15)])
    track4 = Track(
        states=[State([[4]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(13, 20)])
    track5 = Track(
        states=[State([[5]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(18, 28)])
    track6 = Track(
        states=[State([[6]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(22, 26)])
    track7 = Track(
        states=[State([[7]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 40)])
    track8 = Track(
        states=[State([[8]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(30, 35)])
    track9 = Track(
        states=[State([[9]], timestamp=tstart + datetime.timedelta(seconds=i))
                for i in range(35, 40)])
    manager.groundtruth_paths = truth
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

    assert len(metrics) == 5

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
