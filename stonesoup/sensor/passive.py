# -*- coding: utf-8 -*-
import numpy as np

from typing import Set, Union

from ..base import Property
from ..models.measurement.nonlinear import CartesianToElevationBearing
from ..sensor.sensor import Sensor
from ..types.array import CovarianceMatrix
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthState


class PassiveElevationBearing(Sensor):
    """A simple passive sensor that generates measurements of targets, using a
    :class:`~.CartesianToElevationBearing` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.CartesianToElevationBearing`\
            model")
    mapping = Property(
        [np.array],
        doc="Mapping between the targets state space and the sensors\
            measurement capability")
    noise_covar = Property(
        CovarianceMatrix,
        doc="The sensor noise covariance matrix. This is utilised by\
            (and follow in format) the underlying \
            :class:`~.CartesianToElevationBearing` model")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[Detection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `True`, in which case :meth:`~.Model.rvs` is used
            if 'False', no noise will be added)

        Returns
        -------
        Set[:class:`~.Detection`]
            A measurement generated from the given state. The timestamp of the\
            measurement is set equal to that of the provided state.
        """

        measurement_model = CartesianToElevationBearing(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:

            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = Detection(measurement_vector,
                                  measurement_model=measurement_model,
                                  timestamp=truth.timestamp)
            detections.add(detection)

        return detections
