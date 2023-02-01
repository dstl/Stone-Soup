from copy import copy
from typing import Union, Callable

import numpy as np
from scipy.stats import multivariate_normal as mn

from stonesoup.base import Property
from stonesoup.hypothesiser import Hypothesiser
from stonesoup.measures import SquaredMahalanobis
from stonesoup.predictor import Predictor
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State
from stonesoup.updater import Updater


class PDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    clutter_spatial_density: float = Property(
        default=None,
        doc="Spatial density of clutter - tied to probability of false detection. Default is None "
            "where the clutter spatial density is calculated based on assumption that "
            "all but one measurement within the validation region of the track are clutter.")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")
    prob_detect: Union[Probability, Callable[[State], Probability]] = Property(
        default=None,
        doc="Target Detection Probability")
    predict: bool = Property(default=True, doc="Perform prediction step")
    per_measurement: bool = Property(default=False, doc="Generate per measurement predictions")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.prob_detect is None:
            self.prob_detect = lambda x: Probability(0.85)

    def hypothesise(self, track, detections, timestamp, **kwargs):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track : Track
            The track object to hypothesise on
        detections : set of :class:`~.Detection`
            The available detections
        timestamp : datetime.datetime
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~.SingleProbabilityHypothesis` objects
        """

        hypotheses = list()

        if self.predict:
            # Common state & measurement prediction
            prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
        else:
            prediction = track.state
        # Missed detection hypothesis
        prob_detect = self.prob_detect(prediction)
        # Missed detection hypothesis
        probability = Probability(1 - prob_detect*self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
                ))

        # True detection hypotheses
        measurement_prediction = None
        for detection in detections:
            if self.predict and self.per_measurement:
                # Re-evaluate prediction
                prediction = self.predictor.predict(
                    track.state, timestamp=detection.timestamp)
                prob_detect = self.prob_detect(prediction)

            if self.per_measurement or measurement_prediction is None:
                # Compute measurement prediction and probability measure
                measurement_prediction = self.updater.predict_measurement(
                    prediction, detection.measurement_model, **kwargs)

            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            log_pdf = mn.logpdf(
                (detection.state_vector - measurement_prediction.state_vector).ravel(),
                cov=measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * prob_detect) / self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class IPDAHypothesiser(PDAHypothesiser):
    """ Integrated PDA Hypothesiser """

    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hypothesise(self, track, detections, timestamp, **kwargs):
        r"""Evaluate and return all track association hypotheses.
        """

        hypotheses = list()

        if self.predict:
            # Common state & measurement prediction
            prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
            # Compute predicted existence
            time_interval = timestamp - track.timestamp
            prob_survive = np.exp(-(1-self.prob_survive)*time_interval.total_seconds())
            track.exist_prob = prob_survive * track.exist_prob
        else:
            prediction = track.state
        # Missed detection hypothesis
        prob_detect = self.prob_detect(prediction)
        probability = Probability(1 - prob_detect * self.prob_gate * track.exist_prob)
        w = (1 - track.exist_prob) / ((1 - prob_detect * self.prob_gate) * track.exist_prob)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability,
                metadata={"w": w}
            ))

        # True detection hypotheses
        measurement_prediction = None
        for detection in detections:
            if self.predict and self.per_measurement:
                # Re-evaluate prediction
                prediction = self.predictor.predict(
                    track.state, timestamp=detection.timestamp)
                prob_detect = self.prob_detect(prediction)
            if self.per_measurement or measurement_prediction is None:
                # Compute measurement prediction and probability measure
                measurement_prediction = self.updater.predict_measurement(
                    prediction, detection.measurement_model, **kwargs)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            log_pdf = mn.logpdf(
                (detection.state_vector - measurement_prediction.state_vector).ravel(),
                cov=measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * prob_detect * track.exist_prob)/self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)