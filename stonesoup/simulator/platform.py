from collections.abc import Sequence

from ..base import Property
from ..reader import GroundTruthReader
from .base import DetectionSimulator
from ..buffered_generator import BufferedGenerator
from ..platform import Platform


class PlatformDetectionSimulator(DetectionSimulator):
    """A simple platform detection simulator.

    Processes ground truth data and generates :class:`~.Detection` data
    according to a list of platforms by calling each sensor in these platforms.
    Can append information of the sensors to the metadata of their corresponding detections.

    """
    groundtruth: GroundTruthReader = Property(
        doc='Source of ground truth tracks used to generate detections for.')
    platforms: Sequence[Platform] = Property(
        doc='List of platforms in :class:`~.Platform` to generate sensor detections from.')
    platforms_detectable: bool = Property(
        default=True, doc='If sensor platforms are able to detect each other. Default ``True``')
    attributes_inform: set[str] = Property(
        default_factory=set, doc="Names of attributes to store the value of at time of detection."
    )

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
                    if self.platforms_detectable:
                        truths_to_be_measured = truths.union(self.platforms) - {platform}
                    else:
                        truths_to_be_measured = truths
                    detections = sensor.measure(truths_to_be_measured)

                    # Store metadata:
                    attributes_dict = {attribute_name: sensor.__getattribute__(attribute_name)
                                       for attribute_name in self.attributes_inform}
                    for detection in detections:
                        detection.metadata.update(attributes_dict)

                    yield time, detections
