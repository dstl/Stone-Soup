# -*- coding: utf-8 -*-
import pymap3d
import pytest
import utm

from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader
from ...types.detection import Detection
from ..geo import LongLatToUTMConverter, LLAtoENUConverter, LLAtoNEDConverter


@pytest.fixture()
def detector():
    class Detector(DetectionReader):

        @BufferedGenerator.generator_method
        def detections_gen(self):
            for i in range(-3, 4):
                detections = {Detection([[i], [50], [5000 + i*10]])}
                yield None, detections

    return Detector()


@pytest.mark.parametrize(
    'converter_class,reverse_func',
    [
        (LLAtoENUConverter, pymap3d.enu2geodetic),
        (LLAtoNEDConverter, pymap3d.ned2geodetic),
    ])
def test_lla_reference_converter(detector, converter_class, reverse_func):
    converter = converter_class(detector, reference_point=(0, 50, 5000))

    for i, (time, detections) in zip(range(-3, 4), converter):
        detection = detections.pop()

        assert pytest.approx((50, i, 5000 + i*10), abs=1e-2, rel=1e-3) == \
            reverse_func(*detection.state_vector[:, 0], 50, 0, 5000)


def test_utm_converter(detector):
    converter = LongLatToUTMConverter(detector)

    p_east = float('-inf')
    assert converter.zone_number is None

    for long, (time, detections) in zip(range(-3, 4), converter):
        detection = detections.pop()

        assert converter.zone_number == 30
        assert converter.northern

        assert p_east < detection.state_vector[0]
        p_east = detection.state_vector[0]

        assert pytest.approx((50, long), rel=1e-2, abs=1e-4) == utm.to_latlon(
            *detection.state_vector[0:2, 0], zone_number=30, northern=True)
