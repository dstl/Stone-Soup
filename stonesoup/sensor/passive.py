# -*- coding: utf-8 -*-
from typing import Set, Union

import numpy as np

from stonesoup.models.measurement.classification import BasicTimeInvariantObservervation
from ..base import Property
from ..models.measurement.nonlinear import CartesianToElevationBearing
from ..sensor.sensor import Sensor
from ..types.array import CovarianceMatrix, Matrix
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

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = CartesianToElevationBearing(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections


class BasicObserveSensor(Sensor):
    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in format) the "
            "underlying :class:`~.BasicTimeInvariantObservervation` model")
    mapping: np.ndarray = Property(
        doc="Mapping between the target's state space and the sensors measurement capability")
    reverse_emission: Matrix = Property(default=None, doc="K_{ij} = P(z_{i} | \phi_{j})")
    emission_matrix: Matrix = Property(doc="Matrix defining emissions from measurement classes. "
                                           ":math:`E_{ij} = P(\phi_{i} | z_{j})`, "
                                           "where :math:`z_{j}` is measurement class :math:`j` "
                                           "and :math:`\phi_{i}` is hidden class :math:`i`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if np.shape(self.emission_matrix)[0] != len(self.mapping):
            emission_i = f"{self.emission_matrix.shape[0]}"
            emission_j = f"{self.emission_matrix.shape[1]}"
            emission_shape = str(emission_i) + "x" + str(emission_j)
            raise ValueError(
                f"Emission matrix must be of shape NxM and mapping of length "
                f"N, where N, M denote measurement and state space dimensions "
                f"respectively. An emission matrix of shape {emission_shape} with mapping length "
                f"{len(self.mapping)} was given.")

        self.emission_matrix = self.measurement_model.emission_matrix
        if self.reverse_emission is None:
            self.reverse_emission = self.measurement_model.reverse_emission

    @property
    def measurement_model(self):
        return BasicTimeInvariantObservervation(ndim_state=self.ndim_state,
                                                reverse_emission=self.reverse_emission,
                                                emission_matrix=self.emission_matrix)

    def measure(self, ground_truths: Set[GroundTruthState], **kwargs) -> Set[TrueDetection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """

        detections = set()
        for truth in ground_truths:
            measurement_vector = self.measurement_model.function(truth, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=self.measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections
