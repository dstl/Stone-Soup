from typing import Set, Union

import numpy as np

from .sensor import Sensor
from ..base import Property
from ..types.array import StateVector
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState
from ..models.measurement.nonlinear import PasquilGaussianPlume
from ..types.numeric import Probability


class GasIntensitySensor(Sensor):

    noise: float = Property(default=0.6)

    missed_detection_probability: Probability = Property(
        default=0.1,
        doc="The probability that the detection has detection has been affected by turbulence."
    )

    sensing_threshold: float = Property(
        default=0.1,
        doc="Measurement threshold. Should be set high enough to minimise false detections."
    )

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = self.measurement_model

        detections = set()
        for truth in ground_truths:
            if (noise and np.random.rand > self.missed_detection_probability) or not noise:
                measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            else:
                measurement_vector = StateVector([[0.0]])

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)

            detections.add(detection)

        return detections

    @property
    def measurement_model(self):
        return PasquilGaussianPlume(noise=self.noise,
                                    translation_offset=self.position,
                                    missed_detection_probability=self.missed_detection_probability,
                                    sensing_threshold=self.sensing_threshold)