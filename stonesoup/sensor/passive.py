from typing import Set, Union

import numpy as np

from ..base import Property
from ..models.measurement.nonlinear import CartesianToElevationBearing
from ..sensor.sensor import Sensor
from ..types.array import CovarianceMatrix
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState


class PassiveElevationBearing(Sensor):
    """A simple passive sensor that generates measurements of targets, using a
    :class:`~.CartesianToElevationBearing` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.CartesianToElevationBearing`\
            model")
    mapping: np.ndarray = Property(
        doc="Mapping between the targets state space and the sensors\
            measurement capability")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by\
            (and follow in format) the underlying \
            :class:`~.CartesianToElevationBearing` model")

    @property
    def measurement_model(self):
        return CartesianToElevationBearing(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = self.measurement_model

        if noise is True and len(ground_truths) > 1:
            # Pre-fetch noise values when more than one target
            noise_vectors_iter = iter(measurement_model.rvs(len(ground_truths), **kwargs))

        detections = set()
        for truth in ground_truths:
            if noise is True and len(ground_truths) > 1:
                noise_val = next(noise_vectors_iter)
            else:
                noise_val = noise

            measurement_vector = measurement_model.function(truth, noise=noise_val, **kwargs)

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections
