# -*- coding: utf-8 -*-
import pytest

from ..detection import Detection, CompositeDetection


def test_composite_detection():
    # Test mapping error

    with pytest.raises(ValueError, match="Mappings and sub-detections must have same count"):
        CompositeDetection(sub_states=[Detection([0]), Detection([0]), Detection([0])],
                           mapping=[0, 1])

    with pytest.raises(ValueError, match="Mappings and sub-detections must have same count"):
        CompositeDetection(sub_states=[Detection([0]), Detection([0]), Detection([0])],
                           mapping=[0, 1, 2, 3])

    a = Detection([0], metadata={'colour': 'blue'})
    b = Detection([1], metadata={'speed': 'fast'})
    c = Detection([2], metadata={'size': 'big'})
    detections = [a, b, c]

    # Test default mapping
    assert CompositeDetection(detections).mapping == [0, 1, 2]

    detection = CompositeDetection(sub_states=detections, mapping=[1, 0, 2])

    # Test metadata
    assert detection.metadata == {'colour': 'blue', 'speed': 'fast', 'size': 'big'}

    d = Detection([3], metadata={'colour': 'red'})
    detections = [a, b, c, d]
    detection = CompositeDetection(sub_states=detections,
                                   mapping=[1, 0, 2, 3])
    # Last detection should overwrite metadata of earlier
    assert detection.metadata == {'colour': 'red', 'speed': 'fast', 'size': 'big'}
