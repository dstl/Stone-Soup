from typing import Sequence

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
    groundtruth: GroundTruthReader = Property(
        doc='Source of ground truth tracks used to generate detections for.')
    platforms: Sequence[Platform] = Property(
        doc='List of platforms in :class:`~.Platform` to generate sensor detections from.')

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, truths in self.groundtruth:

            # Move platforms and carry out sensor actions.
            for platform in self.platforms:
                platform.move(time)
                for sensor in platform.sensors:
                    sensor.act(time)

            # Make measurements from sensors
            for platform in self.platforms:
                for sensor in platform.sensors:
                    truths_to_be_measured = truths.union(self.platforms) - {platform}
                    detections = sensor.measure(truths_to_be_measured)
                    yield time, detections
