# -*- coding: utf-8 -*-
from typing import Set, Sequence

import numpy as np

from ..models.measurement.categorical import CategoricalMeasurementModel
from ..base import Property
from ..sensor.sensor import Sensor
from ..types.array import CovarianceMatrix, Matrix
from ..types.detection import TrueDetection, TrueCategoricalDetection
from ..types.groundtruth import GroundTruthState, CategoricalGroundTruthState, GroundTruthPath


class CategoricalSensor(Sensor):
    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in format) the "
            "underlying :class:`~.CategoricalMeasurementModel`.")
    mapping: np.ndarray = Property(
        doc="Mapping between the target's state space and the sensors measurement capability")
    emission_matrix: Matrix = Property(doc="Emission matrix passed to the underlying categorical "
                                           "measurement model. Used for generating measurements.")
    emission_covariance: CovarianceMatrix = Property(doc="Emission covariance matrix passed to "
                                                         "the underlying categorical measurement "
                                                         "model.")
    num_categories: int = Property(default=None,
                                   doc="Number of possible categories in the measurement space. "
                                       "Used in generated categorical detections.")
    category_names: Sequence[str] = Property(default=None,
                                             doc="Measurement category names passed to underlying "
                                                 "categorical measurement model.")

    @property
    def measurement_model(self):
        return CategoricalMeasurementModel(ndim_state=self.ndim_state,
                                           mapping=self.mapping,
                                           emission_matrix=self.emission_matrix,
                                           emission_covariance=self.emission_covariance,
                                           category_names=self.category_names)

    def measure(self, ground_truths: Set[GroundTruthState], **kwargs) -> Set[TrueDetection]:
        """Generate a categorical measurement for a given categorical state.

        Parameters
        ----------
        ground_truths : Set[:class:`~.CategoricalGroundTruthState`]
            A set of :class:`~.CategoricalGroundTruthState`

        Returns
        -------
        Set[:class:`~.TrueCategoricalDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """

        detections = set()
        for truth in ground_truths:

            wrong_type = False
            if isinstance(truth, GroundTruthPath):
                if not isinstance(truth[-1], CategoricalGroundTruthState):
                    wrong_type = True
            elif not isinstance(truth, CategoricalGroundTruthState):
                wrong_type = True

            if wrong_type:
                raise ValueError("Categorical sensor can only observe categorical states")

            measurement_vector = self.measurement_model.function(truth, **kwargs)
            detection = TrueCategoricalDetection(state_vector=measurement_vector,
                                                 timestamp=truth.timestamp,
                                                 measurement_model=self.measurement_model,
                                                 groundtruth_path=truth,
                                                 num_categories=self.num_categories,
                                                 category_names=self.category_names)
            detections.add(detection)

        return detections
