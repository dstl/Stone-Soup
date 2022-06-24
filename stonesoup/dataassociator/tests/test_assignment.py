from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.time import TimeRange, CompoundTimeRange
from ...types.track import Track
from .._assignment import multidimensional_deconfliction
import datetime
import pytest


def test_multi_deconfliction():
    test = AssociationSet()
    tested = multidimensional_deconfliction(test)
    assert test == tested
    tracks = [Track(), Track(), Track(), Track()]
    times = [datetime.datetime(year=2022, month=6, day=1, hour=0),
             datetime.datetime(year=2022, month=6, day=1, hour=1),
             datetime.datetime(year=2022, month=6, day=1, hour=5),
             datetime.datetime(year=2022, month=6, day=2, hour=5),
             datetime.datetime(year=2022, month=6, day=1, hour=9)]
    ranges = [TimeRange(times[0], times[1]),
              TimeRange(times[1], times[2]),
              TimeRange(times[2], times[3]),
              TimeRange(times[0], times[4]),
              TimeRange(times[2], times[4])]

    assoc1 = TimeRangeAssociation({tracks[0], tracks[1]}, time_range=ranges[0])
    assoc2 = TimeRangeAssociation({tracks[2], tracks[3]}, time_range=ranges[0])
    assoc_fail = TimeRangeAssociation({tracks[0]}, time_range=ranges[0])
    with pytest.raises(ValueError):
        multidimensional_deconfliction(AssociationSet({assoc_fail, assoc1, assoc2}))

    # Objects do not conflict, so should do nothing
    test2 = AssociationSet({assoc1, assoc2})
    assert multidimensional_deconfliction(test2).associations == test2.associations
