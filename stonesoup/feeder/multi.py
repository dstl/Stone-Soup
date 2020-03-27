# -*- coding: utf-8 -*-
from .base import Feeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..detector import Detector


class MultiDetectorFeeder(Feeder):
    """Multi-detector Feeder

    This returns detections from multiple detectors as a single stream,
    yielding from the detector yielding the lowest timestamp first.
    """
    detector = None
    detectors = Property([Detector], doc="Detectors to yield from")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detector_iters = (iter(detector) for detector in self.detectors)
        iter_detections = {
            detector_iter: next(detector_iter)
            for detector_iter in detector_iters}

        min_time = None
        while iter_detections:  # Whilst still iterators left
            for detector_iter, (time, detections) in iter_detections.items():
                if min_time is None or time < min_time:
                    min_time = time
                    min_detector_iter = detector_iter
                    min_detections = detections

            yield min_time, min_detections
            min_time = None

            try:
                # Grab next set for this iter
                iter_detections[min_detector_iter] = next(min_detector_iter)
            except StopIteration:
                # Empty iterator, remove from dictionary
                del iter_detections[min_detector_iter]
