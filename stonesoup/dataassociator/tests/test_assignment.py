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
    tracks = [Track(id=0), Track(id=1), Track(id=2), Track(id=3)]
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
    assoc3 = TimeRangeAssociation({tracks[0], tracks[3]},
                                  time_range=CompoundTimeRange([ranges[0], ranges[4]]))
    assoc4 = TimeRangeAssociation({tracks[0], tracks[1]},
                                  time_range=CompoundTimeRange([ranges[1], ranges[4]]))
    # Will fail as there is only one track, rather than two
    assoc_fail = TimeRangeAssociation({tracks[0]}, time_range=ranges[0])
    with pytest.raises(ValueError):
        multidimensional_deconfliction(AssociationSet({assoc_fail, assoc1, assoc2}))

    #  Objects do not conflict, so should do nothing
    test2 = AssociationSet({assoc1, assoc2})
    assert multidimensional_deconfliction(test2).associations == {assoc1, assoc2}

    # Objects do conflict, so remove the shorter one
    test3 = AssociationSet({assoc1, assoc3})
    # Should entirely remove assoc1
    tested3 = multidimensional_deconfliction(test3)
    assert len(tested3) == 1
    test_assoc3 = next(iter(tested3.associations))
    for var in vars(test_assoc3):
        assert getattr(test_assoc3, var) == getattr(assoc3, var)

    test4 = AssociationSet({assoc1, assoc2, assoc3, assoc4})
    # assoc1 and assoc4 should merge together, assoc3 should be removed, and assoc2 should remain
    tested4 = multidimensional_deconfliction(test4)
    assert len(tested4) == 2
    assert assoc2 in tested4
    merged = tested4.associations_including_objects({tracks[0], tracks[1]})
    assert len(merged) == 1
    merged = merged.associations.pop()
    assert merged.time_range == CompoundTimeRange([ranges[0], ranges[1], ranges[4]])
