# -*- coding: utf-8 -*-

from ..base import Property
from ..reader import GroundTruthReader
from .base import DetectionSimulator
from ..buffered_generator import BufferedGenerator
from ..platform import Platform


class PlatformDetectionSimulator(DetectionSimulator):
    """A simple platform detection simulator.

    Processes ground truth data and generates :class:`~.Detection` data
    according to a list of platforms by calling each sensor in these platforms.

    """
    groundtruth = Property(GroundTruthReader,
                           doc='Source of ground truth tracks used to generate'
                               ' detections for.')
    platforms = Property([Platform],
                         doc='List of platforms in :class:`~.Platform` to '
                             'generate sensor detections from.')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, truths in self.groundtruth:
            for platform in self.platforms:
                platform.move(time)
            for platform in self.platforms:
                for sensor in platform.sensors:
                    detections = set()
                    for truth in truths.union(self.platforms):

                        # Make sure platform's sensors do not measure itself
                        if truth is platform:
                            continue

                        detection = sensor.measure(truth)
                        if detection is not None:
                            detections.add(detection)
                    yield time, detections
