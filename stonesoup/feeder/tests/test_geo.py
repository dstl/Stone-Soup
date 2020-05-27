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
                detections = {Detection([[i], [50 + i], [5000 + i*10]])}
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

    for long, (time, detections) in enumerate(converter, -3):
        detection = detections.pop()
        lat = 50 + long
        alt = 5000 + long * 10

        assert pytest.approx((lat, long, alt), abs=1e-2, rel=1e-3) == \
            reverse_func(*detection.state_vector, 50, 0, 5000)


def test_utm_converter(detector):
    converter = LongLatToUTMConverter(detector)

    p_east = float('-inf')
    assert converter.zone_number is None
    assert converter.zone_letter is None

    for long, (time, detections) in enumerate(converter, -3):
        detection = detections.pop()
        lat = 50 + long

        assert converter.zone_number == 30
        assert converter.zone_letter == 'T'
        assert detection.metadata['utm_zone'] == (30, 'T')

        assert p_east < detection.state_vector[0]
        p_east = detection.state_vector[0]

        assert pytest.approx((lat, long), rel=1e-2, abs=1e-4) == utm.to_latlon(
            *detection.state_vector[0:2], zone_number=30, northern=True)


def test_utm_set_zone(detector):
    # Note, zone letter doesn't matter unless it's north vs. south of the equator
    converter = LongLatToUTMConverter(detector, zone_number=31, zone_letter='U')

    p_east = float('-inf')
    assert converter.zone_number == 31
    assert converter.zone_letter == 'U'

    for long, (time, detections) in enumerate(converter, -3):
        detection = detections.pop()
        lat = 50 + long

        assert converter.zone_number == 31
        assert converter.zone_letter == 'U'
        assert detection.metadata['utm_zone'] == (31, 'U')

        assert p_east < detection.state_vector[0]
        p_east = detection.state_vector[0]

        assert pytest.approx((lat, long), rel=1e-2, abs=1e-4) == utm.to_latlon(
            *detection.state_vector[0:2], zone_number=31, zone_letter='U', strict=False)

        # Correct zone shouldn't match, as conversion is wrong
        assert pytest.approx((lat, long)) != utm.to_latlon(
            *detection.state_vector[0:2], zone_number=30, zone_letter='U', strict=False)


def test_utm_converter_dynamic(detector):
    converter = LongLatToUTMConverter(detector, dynamic=True)

    p_east = float('-inf')
    assert converter.zone_number is None
    assert converter.zone_letter is None

    for long, (time, detections) in enumerate(converter, -3):
        detection = detections.pop()
        lat = 50 + long

        assert converter.zone_number is None
        assert converter.zone_letter is None
        if lat < 48:
            assert detection.metadata['utm_zone'] == (30, 'T')
        elif lat < 50:
            assert detection.metadata['utm_zone'] == (30, 'U')
        else:  # Long > 0, zone change
            assert detection.metadata['utm_zone'] == (31, 'U')

        if lat == 50:
            # Switched zones, so east value will be smaller than previous
            assert p_east > detection.state_vector[0]
        else:
            assert p_east < detection.state_vector[0]
        p_east = detection.state_vector[0]

        assert pytest.approx((lat, long), rel=1e-2, abs=1e-4) == utm.to_latlon(
            *detection.state_vector[0:2], *detection.metadata['utm_zone'])
