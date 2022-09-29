from ..base import Property
from ..measures import Measure
from ..types.multihypothesis import MultipleHypothesis
from .base import Gater


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
