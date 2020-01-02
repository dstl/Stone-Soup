# -*- coding: utf-8 -*-
import datetime
import numpy as np

import pytest

from ..filter import (MetadataReducer,
                      MetadataValueFilter,
                      BoundingBoxDetectionReducer)


def test_metadata_reducer(detector):
    feeder = MetadataReducer(detector, metadata_field="colour")

    multi_none = False
    for time, detections in feeder:
        all_colours = [detection.metadata.get('colour')
                       for detection in detections]
        if not multi_none:
            multi_none = len(
                [colour for colour in all_colours if colour is None]) > 1

        colours = [colour for colour in all_colours if colour is not None]
        assert len(colours) == len(set(colours))

        assert "red" in colours
        assert "blue" in colours
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert "green" in colours
        else:
            assert "green" not in colours

        assert all(time == detection.timestamp for detection in detections)

    assert multi_none


def test_metadata_value_filter(detector):
    feeder = MetadataValueFilter(detector,
                                 metadata_field="score",
                                 operator=lambda x: x >= 0.1)
    # Discard unmatched
    nones = False
    for time, detections in feeder:
        all_scores = [detection.metadata.get('score')
                      for detection in detections]
        nones = nones | (len([score for score in all_scores
                              if score is None]) > 0)
        scores = [score for score in all_scores if score is not None]
        assert len(scores) == len(set(scores))
        assert 0 not in scores
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert all([score <= 0.5 for score in scores])
        assert all(time == detection.timestamp for detection in detections)
    assert not nones

    # Keep unmatched
    feeder.keep_unmatched = True
    nones = False
    for time, detections in feeder:
        all_scores = [detection.metadata.get('score')
                      for detection in detections]
        nones = nones | (len([score for score in all_scores
                              if score is None]) > 0)
        scores = [score for score in all_scores if score is not None]
        assert len(scores) == len(set(scores))
        assert 0 not in scores
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert all([score <= 0.5 for score in scores])
        assert all(time == detection.timestamp for detection in detections)
    assert nones


def test_boundingbox_reducer(detector):

    # Simple 2D rectangle/bounding box
    limits = np.array([[-1, 1],
                       [-2, 2]])
    mapping = [1, 0]

    # Confirm errors raised on improper instantiation attempt
    with pytest.raises(TypeError):
        BoundingBoxDetectionReducer()
    with pytest.raises(TypeError):
        BoundingBoxDetectionReducer(detector)

    feeder = BoundingBoxDetectionReducer(detector, limits, mapping)

    # Assert correct constructor assignments
    assert np.array_equal(limits, feeder.limits)
    assert np.array_equal(mapping, feeder.mapping)

    # Ensure only measurements within box are returned
    multi_check = True
    for time, detections in feeder:

        for detection in detections:
            num_dims = len(limits)
            for i in range(num_dims):
                min = limits[i, 0]
                max = limits[i, 1]
                value = detection.state_vector[mapping[i]]
                multi_check &= not (value < min or value > max)

        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert len(detections) == 1
        elif time == datetime.datetime(2019, 4, 1, 14, 0, 7):
            assert not (len(detections))

        assert all(time == detection.timestamp for detection in detections)

    assert multi_check


def test_boundingbox_reducer_default_mapping(detector):
    # Simple 2D rectangle/bounding box
    limits = np.array([[-1, 1],
                       [-2, 2]])

    feeder = BoundingBoxDetectionReducer(detector, limits)

    assert feeder.mapping == (0, 1)
