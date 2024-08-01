from typing import Sequence

import numpy as np

from .base import Gater
from ..base import Property
from ..measures import Measure
from ..models.base import LinearModel, ReversibleModel
from ..types.detection import MissedDetection
from ..types.hypothesis import Hypothesis, SingleHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.state import State


class DistanceGater(Gater):
    """ Distance based gater

    Uses a measure to calculate the distance between a hypothesis' measurement prediction and the
    hypothised measurement, then removes any hypotheses whose calculated distance exceeds the
    specified gate threshold.
    """
    measure: Measure = Property(
        doc="Measure class used to calculate the distance between the measurement "
            "prediction and the hypothesised measurement.")
    gate_threshold: float = Property(
        doc="The gate threshold. Hypotheses whose calculated distance "
            "exceeds this threshold will be filtered out.")

    def hypothesise(self, track, detections, *args, **kwargs):

        hypotheses = self.hypothesiser.hypothesise(track, detections, *args, **kwargs)

        gated_hypotheses = [hypothesis for hypothesis in hypotheses
                            if (not hypothesis
                                or self.measure(hypothesis.measurement_prediction,
                                                hypothesis.measurement) < self.gate_threshold)]

        return MultipleHypothesis(sorted(gated_hypotheses, reverse=True))


class TrackingStateSpaceDistanceGater(Gater):
    """
    This Distance Gater measures in the track state space (not measurement state space).

    Each measurement is transformed into the state space of the track. In this state space a
    measure is used to calculate the distance between a hypothesis' prediction and the
    measurement. Any hypotheses whose calculated distance exceeds the specified gate threshold are
    removed.
    """

    measure: Measure = Property(
        doc="Measure class used to calculate the distance between the measurement (in state space)"
            " and the track prediction.")
    gate_threshold: float = Property(
        doc="The gate threshold. Measurements whose calculated distance "
            "exceeds this threshold will be filtered out.")
    allow_non_reversible_detections: bool = Property(
        default=True,
        doc="Should detections with non-reversible measurement models be allowed passed the gate.")

    def hypothesise(self, track, detections, *args, **kwargs) -> Sequence[Hypothesis]:

        hypotheses: Sequence[Hypothesis] = \
            self.hypothesiser.hypothesise(track, detections, *args, **kwargs)

        return [hypothesis for hypothesis in hypotheses
                if self.check_hypothesise(hypothesis)]

    def check_hypothesise(self, hypothesis: Hypothesis) -> bool:
        if isinstance(hypothesis, SingleHypothesis):
            return self.check_single_hypothesise(hypothesis)
        else:
            raise NotImplementedError(f"Cannot gate hypothesis with type: '{type(hypothesis)}'.")

    def check_single_hypothesise(self, hypothesis: SingleHypothesis) -> bool:

        detection = hypothesis.measurement

        if isinstance(detection, MissedDetection):
            return True

        if detection.measurement_model is not None:
            measurement_model = detection.measurement_model
        else:
            raise ValueError("No measurement model specified in detection.")

        if isinstance(measurement_model, LinearModel):
            model_matrix = measurement_model.matrix()
            inv_model_matrix = np.linalg.pinv(model_matrix)
            state_vector = inv_model_matrix @ detection.state_vector
        elif isinstance(measurement_model, ReversibleModel):
            try:
                state_vector = measurement_model.inverse_function(detection)
            except NotImplementedError:
                # Model failed to be reversed
                return self.allow_non_reversible_detections
        else:
            # Invalid measurement model used. Must be an instance of linear or reversible
            return self.allow_non_reversible_detections

        measurement_in_state_space = State(state_vector)

        track_prediction = hypothesis.prediction
        distance = self.measure(track_prediction, measurement_in_state_space)

        return distance < self.gate_threshold
