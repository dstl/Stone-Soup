from typing import Set, Union

import numpy as np

from .sensor import Sensor
from ..base import Property
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState
from ..models.measurement.nonlinear import IsotropicPlume
from ..types.numeric import Probability


class GasIntensitySensor(Sensor):

    min_noise: float = Property(
        default=1e-4,
        doc="The minimum noise added to sensor measurements"
    )

    standard_deviation_percentage: float = Property(
        default=0.5,
        doc="Standard deviation as a percentage of the concentration level"
    )

    missed_detection_probability: Probability = Property(
        default=0.1,
        doc="The probability that the detection has detection has been affected by turbulence "
            "and therefore not sensed the gas."
    )

    sensing_threshold: float = Property(
        default=1e-4,
        doc="Measurement threshold. Should be set high enough to minimise false detections."
    )

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = self.measurement_model

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)

            detections.add(detection)

        return detections

    @property
    def measurement_model(self):
        return IsotropicPlume(min_noise=self.min_noise,
                              standard_deviation_percentage=self.standard_deviation_percentage,
                              translation_offset=self.position,
                              missed_detection_probability=self.missed_detection_probability,
                              sensing_threshold=self.sensing_threshold)
