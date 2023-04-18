import numpy as np

from stonesoup.types.detection import Detection
from ..base import Property
from ..models.measurement.nonlinear import CartesianToElevationBearing
from ..sensor.sensor import SimpleSensor
from ..types.array import CovarianceMatrix
from ..types.groundtruth import GroundTruthState


class PassiveElevationBearing(SimpleSensor):
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

    def is_detectable(self, state: GroundTruthState) -> bool:
        return True

    def is_clutter_detectable(self, state: Detection) -> bool:
        return True
