import datetime

import numpy as np
import pytest

from ..association import Association, AssociationPair, AssociationSet, \
    SingleTimeAssociation, TimeRangeAssociation
from ..detection import Detection
from ..time import TimeRange, CompoundTimeRange


def test_association():
    with pytest.raises(TypeError):
        Association()

    objects = {Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))}

    assoc = Association(objects)

    assert assoc.objects == objects


def test_associationpair():
    with pytest.raises(TypeError):
        AssociationPair()

    objects = [Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))]

    # Over 3 objects
    with pytest.raises(ValueError):
        AssociationPair(set(objects))

    # Under 2 objects
    with pytest.raises(ValueError):
        AssociationPair({objects[0]})

    # 2 objects
    assoc = AssociationPair(set(objects[:2]))
    assert np.array_equal(assoc.objects, set(objects[:2]))


def test_singletimeassociation():
    with pytest.raises(TypeError):
        SingleTimeAssociation()

    objects = {Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))}
    timestamp = datetime.datetime(2018, 3, 1, 5, 3, 35)

    assoc = SingleTimeAssociation(objects=objects, timestamp=timestamp)

    assert assoc.objects == objects
    assert assoc.timestamp == timestamp


def test_timerangeassociation():
    with pytest.raises(TypeError):
        TimeRangeAssociation()

    objects = {Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))}
    timestamp1 = datetime.datetime(2018, 3, 1, 5, 3, 35)
    timestamp2 = datetime.datetime(2018, 3, 1, 5, 8, 35)
    timerange = TimeRange(start=timestamp1, end=timestamp2)
    ctimerange = CompoundTimeRange([timerange])

    assoc = TimeRangeAssociation(objects=objects, time_range=timerange)
    cassoc = TimeRangeAssociation(objects=objects, time_range=ctimerange)

    assert assoc.objects == objects
    assert assoc.time_range == timerange
    assert cassoc.objects == objects
    assert cassoc.time_range == ctimerange


def test_associationset():
    # Set up some dummy variables
    timestamp1 = datetime.datetime(2018, 3, 1, 5, 3, 35)
    timestamp2 = datetime.datetime(2018, 3, 1, 5, 8, 35)
    timestamp3 = datetime.datetime(2020, 3, 1, 1, 1, 1)
    time_range = TimeRange(start=timestamp1, end=timestamp2)
    time_range2 = TimeRange(start=timestamp2, end=timestamp3)

    objects_list = [Detection(np.array([[1], [2]])),
                    Detection(np.array([[3], [4]])),
                    Detection(np.array([[5], [6]]))]

    assoc1 = SingleTimeAssociation(objects=set(objects_list),
                                   timestamp=timestamp1)

    assoc2 = TimeRangeAssociation(objects=set(objects_list[1:]),
                                  time_range=time_range)
    assoc2_same_objects = TimeRangeAssociation(objects=set(objects_list[1:]),
                                               time_range=time_range2)

    assoc_set = AssociationSet({assoc1, assoc2})

    assert assoc_set.associations == {assoc1, assoc2}

    assert assoc1 in assoc_set
    assert assoc2 in assoc_set
    assert 'a' not in assoc_set

    assoc_iter = iter(assoc_set)
    for assoc in assoc_iter:
        assert assoc in {assoc1, assoc2}

    assert len(assoc_set) == 2

    # test _simplify method

    simplify_test = AssociationSet({assoc1, assoc2, assoc2_same_objects})

    assert len(simplify_test.associations) == 2

    # Test associations including objects

    # Object only present in object 1
    assert assoc_set.associations_including_objects(objects_list[0]) \
        == AssociationSet({assoc1})
    # Object present in both
    assert assoc_set.associations_including_objects(objects_list[1]) \
        == AssociationSet({assoc1, assoc2})
    # Object present in neither
    assert assoc_set.associations_including_objects(Detection(np.array([[0], [0]]))) \
        == AssociationSet()

    # Test associations including timestamp

    # Timestamp present in one object
    assert assoc_set.associations_at_timestamp(timestamp2) \
        == AssociationSet({assoc2})
    # Timestamp present in both
    assert assoc_set.associations_at_timestamp(timestamp1) \
        == AssociationSet({assoc1, assoc2})
    # Timestamp not present in either
    timestamp4 = datetime.datetime(2022, 3, 1, 6, 8, 35)
    assert assoc_set.associations_at_timestamp(timestamp4) \
        == AssociationSet()


def test_association_set_add_remove():
    test = AssociationSet()
    with pytest.raises(TypeError):
        test.add("a string")
    with pytest.raises(TypeError):
        test.remove("a string")
    objects = {Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))}

    assoc = Association(objects)
    assert assoc not in test.associations
    with pytest.raises(ValueError):
        test.remove(assoc)
    test.add(assoc)
    assert assoc in test.associations
    test.remove(assoc)
    assert assoc not in test.associations


def test_association_set_properties():
    test = AssociationSet()
    assert test.key_times == []
    assert test.overall_time_range == CompoundTimeRange()
    assert test.object_set == set()
    objects = [Detection(np.array([[1], [2]])),
               Detection(np.array([[3], [4]])),
               Detection(np.array([[5], [6]]))]
    assoc = Association(set(objects))
    test2 = AssociationSet({assoc})
    assert test2.key_times == []
    assert test2.overall_time_range == CompoundTimeRange()
    assert test2.object_set == set(objects)
    timestamp1 = datetime.datetime(2018, 3, 1, 5, 3, 35)
    timestamp2 = datetime.datetime(2018, 3, 1, 5, 8, 35)
    time_range = TimeRange(start=timestamp1, end=timestamp2)
    com_time_range = CompoundTimeRange([time_range])
    assoc2 = TimeRangeAssociation(objects=set(objects[1:]),
                                  time_range=com_time_range)
    test3 = AssociationSet({assoc, assoc2})
    assert test3.key_times == [timestamp1, timestamp2]
    assert test3.overall_time_range == com_time_range
    assert test3.object_set == set(objects)
