# -*- coding: utf-8 -*-

import pytest

from ...sensor.composite import CompositeSensor
from ...types.detection import CompositeDetection, Detection

composite_sensor = CompositeSensor(sensors=3 * [None], mapping=[1, 0, 2])


def test_composite_detection_mapping_errors():
    with pytest.raises(ValueError, match="Cannot define mapping and sensor"):
        CompositeDetection(sub_states=[Detection([0]), Detection([0]), Detection([0])],
                           sensor=composite_sensor,
                           default_mapping=[0, 1, 2])

    with pytest.raises(ValueError, match="Must have mapping for each sub-detection"):
        CompositeDetection(sub_states=[Detection([0]), Detection([0]), Detection([0])],
                           default_mapping=[0, 1])

    with pytest.raises(ValueError, match="Must have mapping for each sub-detection"):
        CompositeDetection(sub_states=[Detection([0]), Detection([0]), Detection([0])],
                           default_mapping=[0, 1, 2, 3])


def test_composite_detection():
    a = Detection([0], metadata={'colour': 'blue'})
    b = Detection([1], metadata={'speed': 'fast'})
    c = Detection([2], metadata={'size': 'big'})
    detections = [a, b, c]

    # Test mapping
    assert (CompositeDetection(detections,
                               sensor=composite_sensor).mapping == composite_sensor.mapping).all()
    assert (CompositeDetection(detections, default_mapping=[1, 0, 2]).mapping == [1, 0, 2]).all()
    assert (CompositeDetection(detections).mapping == [0, 1, 2]).all()

    detection = CompositeDetection(sub_states=detections, default_mapping=[1, 0, 2])

    # Test metadata
    assert detection.metadata == {'colour': 'blue', 'speed': 'fast', 'size': 'big'}

    d = Detection([3], metadata={'colour': 'red'})
    detections = [a, b, c, d]
    detection = CompositeDetection(sub_states=detections,
                                   default_mapping=[1, 0, 2, 3])
    # Last detection should overwrite metadata of earlier
    assert detection.metadata == {'colour': 'red', 'speed': 'fast', 'size': 'big'}
