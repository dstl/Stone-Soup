# -*- coding: utf-8 -*-
import pymap3d
import pytest
import utm

from ..geo import LongLatToUTMConverter, LLAtoENUConverter, LLAtoNEDConverter
from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader, GroundTruthReader
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthState, GroundTruthPath


@pytest.fixture(params=['detector', 'groundtruth'])
def reader(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def detector():
    class Detector(DetectionReader):

        @BufferedGenerator.generator_method
        def detections_gen(self):
            for i in range(-3, 4):
                detections = {Detection([[i], [50], [5000 + i*10]])}
                yield None, detections

    return Detector()


@pytest.fixture()
def groundtruth():
    class GroundTruth(GroundTruthReader):

        @BufferedGenerator.generator_method
        def groundtruth_paths_gen(self):
            path = GroundTruthPath()
            for i in range(-3, 4):
                path.append(GroundTruthState([[i], [50], [5000 + i*10]]))
                truths = {path}
                yield None, truths

    return GroundTruth()


@pytest.mark.parametrize(
    'converter_class,reverse_func',
    [
        (LLAtoENUConverter, pymap3d.enu2geodetic),
        (LLAtoNEDConverter, pymap3d.ned2geodetic),
    ])
def test_lla_reference_converter(reader, converter_class, reverse_func):
    converter = converter_class(reader, reference_point=(0, 50, 5000))

    for i, (time, detections) in zip(range(-3, 4), converter):
        detection = detections.pop()

        assert pytest.approx((50, i, 5000 + i*10), abs=1e-2, rel=1e-3) == \
            reverse_func(*detection.state_vector[:, 0], 50, 0, 5000)


def test_utm_converter(reader):
    converter = LongLatToUTMConverter(reader)

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
