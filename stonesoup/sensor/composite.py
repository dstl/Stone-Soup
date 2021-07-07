from typing import Set, Union, Sequence

import numpy as np

from ..base import Property
from ..sensor.sensor import Sensor
from ..types.detection import TrueDetection, CompositeDetection
from ..types.groundtruth import CompositeGroundTruthState


class CompositeSensor(Sensor):
    sensors: Sequence[Sensor] = Property()
    mapping: Sequence = Property(default=None,
                                 doc="Mapping of which component states in the composite truth "
                                     "state is measured.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping is None:
            self.mapping = list(np.arange(len(self.sensors)))

    def measure(self, ground_truths: Set[CompositeGroundTruthState],
                noise: Sequence[Union[np.ndarray, bool]] = True,
                **kwargs) -> Set[TrueDetection]:

        if isinstance(noise, bool) or len(noise) == 1:
            noise = len(self.sensors) * [noise]

        detections = set()
        for truth in ground_truths:

            sub_detections = list()

            states = [truth.sub_states[i] for i in self.mapping]

            for state, sensor, sub_noise in zip(states, self.sensors, noise):
                sub_detection = sensor.measure(ground_truths={state},
                                               noise=sub_noise).pop()  # returns a set
                sub_detections.append(sub_detection)

            detection = CompositeDetection(sub_states=sub_detections, groundtruth_path=truth,
                                           sensor=self, mapping=self.mapping)
            detections.add(detection)

        return detections
