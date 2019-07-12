# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pytest

from ..association import Association, AssociationPair, AssociationSet, \
    SingleTimeAssociation, TimeRangeAssociation
from ..detection import Detection
from ..time import TimeRange


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
    np.array_equal(assoc.objects, set(objects[:2]))


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
    timerange = TimeRange(start_timestamp=timestamp1, end_timestamp=timestamp2)

    assoc = TimeRangeAssociation(objects=objects, time_range=timerange)

    assert assoc.objects == objects
    assert assoc.time_range == timerange


def test_associationset():
    # Set up some dummy variables
    timestamp1 = datetime.datetime(2018, 3, 1, 5, 3, 35)
    timestamp2 = datetime.datetime(2018, 3, 1, 5, 8, 35)
    timerange = TimeRange(start_timestamp=timestamp1, end_timestamp=timestamp2)

    objects_list = [Detection(np.array([[1], [2]])),
                    Detection(np.array([[3], [4]])),
                    Detection(np.array([[5], [6]]))]

    assoc1 = SingleTimeAssociation(objects=set(objects_list),
                                   timestamp=timestamp1)

    assoc2 = TimeRangeAssociation(objects=set(objects_list[1:]),
                                  time_range=timerange)

    assoc_set = AssociationSet({assoc1, assoc2})

    assert assoc_set.associations == {assoc1, assoc2}

    # Test associations including objects

    # Object only present in object 1
    assert assoc_set.associations_including_objects(objects_list[0]) \
        == {assoc1}
    # Object present in both
    assert assoc_set.associations_including_objects(objects_list[1]) \
        == {assoc1, assoc2}
    # Object present in neither
    assert not assoc_set.associations_including_objects(
        Detection(np.array([[0], [0]])))

    # Test associations including timestamp

    # Timestamp present in one object
    assert assoc_set.associations_at_timestamp(timestamp2) \
        == {assoc2}
    # Timestamp present in both
    assert assoc_set.associations_at_timestamp(timestamp1) \
        == {assoc1, assoc2}
    # Timestamp not present in either
    timestamp3 = datetime.datetime(2018, 3, 1, 6, 8, 35)
    assert not assoc_set.associations_at_timestamp(timestamp3)
