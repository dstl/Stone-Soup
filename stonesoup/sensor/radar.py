# -*- coding: utf-8 -*-
import numpy as np

from .base import Sensor
from ..base import Property
from ..types.state import StateVector
from ..types.detection import Detection
from ..types.array import CovarianceMatrix
from ..models.measurement.nonlinear\
    import RangeBearingGaussianToCartesian


class SimpleRadar(Sensor):
    """A simple radar sensor that generates measurements of targets, using a
    :class:`~.RangeBearingGaussianToCartesian` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 2D Cartesian plane.

    TODO: Extend to nD state space
    """

    position = Property(StateVector,
                        doc="The radar position on a 2D Cartesian plane")
    noise_covar = Property(CovarianceMatrix,
                           doc="The sensor noise covariance matrix. This is utilised\
                                by (and follow in format) the underlying\
                                :class:`~.RangeBearingGaussianToCartesian`\
                                model")

    def __init__(self, position, noise_covar, *args, **kwargs):

        measurement_model = RangeBearingGaussianToCartesian(
            ndim_state=2, mapping=np.array([0, 1]),
            noise_covar=noise_covar,
            origin_offset=position)
        super().__init__(position, noise_covar, measurement_model,
                         *args, **kwargs)

    def set_position(self, position):
        self.position = position
        self.measurement_model.origin_offset = position

    def gen_measurement(self, ground_truth, **kwargs):
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truth : :class:`~.State`
            A ground-truth state

        Returns
        -------
        :class:`~.Detection`
            A measurement generated from the given state. The timestamp of the\
            measurement is set equal to that of the provided state.
        """

        measurement_vector = self.measurement_model.function(
            ground_truth.state_vector, **kwargs)

        return Detection(measurement_vector, timestamp=ground_truth.timestamp)
