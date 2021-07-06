# -*- coding: utf-8 -*-
import numpy as np

from .base import Initiator
from ..base import Property
from ..models.measurement.categorical import CategoricalMeasurementModel
from ..types.hypothesis import SingleHypothesis
from ..types.state import CategoricalState
from ..types.track import Track
from ..types.update import CategoricalStateUpdate


class SimpleCategoricalInitiator(Initiator):
    """Initiator that creates tracks in a categorical state space.

    The defined :attr:`measurement_model` or all detections' measurement models must be a
    :class:`CategoricalMeasurementModel` type as this class utilises the models' emission matrices
    to determine detections' state space equivalents.
    For state space indices where a detection provides no information, the :attr:`prior_state`
    value is used instead."""
    prior_state: CategoricalState = Property(doc="Prior state information")
    measurement_model: CategoricalMeasurementModel = Property(
        default=None, doc="Measurement model (should be categorical model)")

    def initiate(self, detections, **kwargs):
        tracks = set()

        for detection in detections:
            if detection.measurement_model is not None:
                measurement_model = detection.measurement_model
            else:
                measurement_model = self.measurement_model

            state_vector = measurement_model.emission_matrix @ detection.state_vector

            # Replace the default prior with the measurement's state-space components were possible
            prior_state_vector = self.prior_state.state_vector.copy()
            for index, element in zip(measurement_model.mapping, state_vector):
                prior_state_vector[index] = element

            # Normalise the prior
            prior_state_vector = prior_state_vector / np.sum(prior_state_vector)

            state = CategoricalStateUpdate(state_vector=prior_state_vector,
                                           hypothesis=SingleHypothesis(None, detection),
                                           timestamp=detection.timestamp,
                                           category_names=self.prior_state.category_names)

            tracks.add(Track([state]))
        return tracks
