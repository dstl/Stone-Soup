# -*- coding: utf-8 -*-
import copy
import numpy as np

from .base import Sensor
from ..base import Property
from ..types.state import State, StateVector
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
    orientation = Property(StateVector,
                           doc="A 3x1 array of angles (rad), specifying the\
                                radar orientation in terms of the\
                                clockwise rotation around each\
                                Cartesian axis in the order :math:`z,y,x`.\
                                The rotation angles are positive if the\
                                rotation is in the clockwise direction\
                                when viewed by an observer looking along the\
                                respective rotation axis, towards the origin")
    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.RangeBearingGaussianToCartesian`\
            model")
    mapping = Property(
        [np.array], doc="Mapping between the targets state space and the\
                        sensors measurement capability")
    noise_covar = Property(CovarianceMatrix,
                           doc="The sensor noise covariance matrix. This is \
                                utilised by (and follow in format) the \
                                underlying \
                                :class:`~.RangeBearingGaussianToCartesian`\
                                model")

    def __init__(self, position, orientation, ndim_state, mapping, noise_covar,
                 *args, **kwargs):
        measurement_model = RangeBearingGaussianToCartesian(
            ndim_state=ndim_state,
            mapping=mapping,
            noise_covar=noise_covar,
            translation_offset=position,
            rotation_offset=orientation)

        super().__init__(position, orientation, ndim_state, mapping,
                         noise_covar, measurement_model, *args, **kwargs)

    def set_position(self, position):
        self.position = position
        self.measurement_model.translation_offset = position

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.measurement_model.axis_roation = orientation

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

        model_copy = copy.copy(self.measurement_model)

        return Detection(measurement_vector,
                         measurement_model=model_copy,
                         timestamp=ground_truth.timestamp)
