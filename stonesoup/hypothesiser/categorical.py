# -*- coding: utf-8 -*-

from .base import Hypothesiser
from ..base import Property
from ..measures import ObservationAccuracy
from ..predictor.categorical import HMMPredictor
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..updater.categorical import HMMUpdater


class HMMHypothesiser(Hypothesiser):
    r"""Hypothesiser based on categorical distribution accuracy.

    This hypothesiser generates track predictions at detection times and scores each hypothesised
    prediction-detection pair according to the accuracy of the corresponding measurement
    prediction compared to the detection.
    """

    predictor: HMMPredictor = Property(doc="Predictor used to predict tracks to detection times")
    updater: HMMUpdater = Property(doc="Updater used to get measurement prediction")
    prob_detect: Probability = Property(default=Probability(0.99),
                                        doc="Target Detection Probability")
    prob_gate: Probability = Property(default=Probability(0.95),
                                      doc="Gate Probability - prob. gate contains true "
                                          "measurement if detected")

    def hypothesise(self, track, detections, timestamp):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a MultipleHypothesis object
        with N+1 detections (first detection is a 'MissedDetection'), each with an associated
        accuracy (of prediction emission to measurement), considered the probability of the
        hypothesis being true.

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on. Composed of :class:`~.CategoricalState` types.
        detections: :class:`set`
            A set of :class:`~.CategoricalDetection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement predictions. Note that if a
            given detection has a non empty timestamp, then prediction will be performed according
            to the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~.SingleProbabilityHypothesis` objects
        """

        hypotheses = list()

        prediction = self.predictor.predict(track, timestamp=timestamp)

        probability = Probability(1 - self.prob_detect * self.prob_gate)

        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            ))

        for detection in detections:
            prediction = self.predictor.predict(track, timestamp=detection.timestamp)

            measurement_prediction = self.updater.predict_measurement(
                predicted_state=prediction,
                measurement_model=detection.measurement_model,
                noise=False,
                measurement=detection
            )

            probability = self.measure(measurement_prediction, detection)
            probability = probability * self.prob_detect
            probability = Probability(probability, log_value=False)

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=False, total_weight=1)

    @property
    def measure(self):
        return ObservationAccuracy()
