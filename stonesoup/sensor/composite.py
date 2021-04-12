from stonesoup.base import Property
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.groundtruth import GroundTruthState, CompositeGroundTruthState
from typing import Set, Union, Sequence
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import TrueDetection, CompositeDetection
import numpy as np


class CompositeSensor(Sensor):

    sensors: Sequence[Sensor] = Property()
    mapping: Sequence = Property(default=None,
                                 doc="Mapping of which component states in the composite truth "
                                     "state is meausured.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping is None:
            self.mapping = np.arange(len(self.sensors))

    def measure(self, ground_truths: Set[CompositeGroundTruthState],
                noise: Sequence[Union[np.ndarray, bool]] = True,
                **kwargs) -> Set[TrueDetection]:

        if isinstance(noise, bool) or len(noise) == 1:
            noise = len(self.sensors) * [noise]

        detections = set()
        for truth in ground_truths:

            detection = CompositeDetection(groundtruth_path=truth, sensor=self)

            states = [truth.inner_states[i] for i in self.mapping]

            for state, sensor, sub_noise in zip(states, self.sensors, noise):
                inner_detection = sensor.measure(ground_truths={state}, noise=sub_noise).pop()  # returns a set
                detection.append(inner_detection)

            detections.add(detection)

        return detections
