from typing import Sequence

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.platform import Platform
from stonesoup.simulator.platform import PlatformDetectionSimulator


class PlatformTargetDetectionSimulator(PlatformDetectionSimulator):
    """A simple platform detection simulator.

    Processes ground truth data and generates :class:`~.Detection` data
    according to a list of platforms by calling each sensor in these platforms.

    """
    targets: Sequence[Platform] = Property(
        doc='List of target platforms to be detected'
    )

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, truths in self.groundtruth:
            for platform in self.platforms:
                platform.move(time)
            for platform in self.targets:
                platform.move(time)
            for platform in self.platforms:
                for sensor in platform.sensors:
                    truths_to_be_measured = truths.union(self.targets)
                    detections = sensor.measure(truths_to_be_measured, timestamp=time)
                    yield time, detections