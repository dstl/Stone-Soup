from typing import Set, Union

import numpy as np

from .sensor import Sensor
from ..base import Property
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState
from ..models.measurement.gas import IsotropicPlume
from ..types.numeric import Probability


class GasIntensitySensor(Sensor):
    """A simple gas sensor that measures the concentration of gas at the
    location of the sensor. It implements the :class:`~.IsotropicPlume`
    model for calculating concentration.
    """

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
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `True`, in which
            case :meth:`~.Model.rvs` is used; if `False`, no noise will be added). If `False`,
            the :attr:`sensing_threshold` and :attr:`missed_detection_probability` are not
            considered.

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """

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
