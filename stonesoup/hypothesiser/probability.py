from functools import lru_cache

from scipy.stats import multivariate_normal, chi2
from scipy.linalg import det
from scipy.special import gamma
import numpy as np

from .base import Hypothesiser
from ..base import Property
from ..measures import SquaredMahalanobis
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..predictor import Predictor
from ..updater import Updater


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
    prob_detect: Probability = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")
    include_all: bool = Property(
        default=False,
        doc="If `True`, hypotheses outside probability gates will be returned. This requires "
            "that the clutter spatial density is also provided, as it may not be possible to"
            "estimate this. Default `False`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.include_all and self.clutter_spatial_density is None:
            raise ValueError("Must provide clutter spatial density if including all hypotheses")

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
        http://ais.informatik.uni-freiburg.de/teaching/ws11/robotics2/pdfs/rob2-20-dataassociation.pdf

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
        validated_measurements = 0
        measure = SquaredMahalanobis(state_covar_inv_cache_size=None)

        # Common state & measurement prediction
        prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
        # Missed detection hypothesis
        probability = Probability(1 - self.prob_detect*self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
                ))

        # True detection hypotheses
        for detection in detections:
            # Re-evaluate prediction
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp, **kwargs)
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model, **kwargs)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            log_pdf = multivariate_normal.logpdf(
                (detection.state_vector - measurement_prediction.state_vector).ravel(),
                cov=measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)

            if measure(measurement_prediction, detection) \
                    <= self._gate_threshold(self.prob_gate, measurement_prediction.ndim):
                validated_measurements += 1
                valid_measurement = True
            else:
                # Will be gated out unless include_all is set
                valid_measurement = False

            if self.include_all or valid_measurement:
                probability = pdf * self.prob_detect
                if self.clutter_spatial_density is not None:
                    probability /= self.clutter_spatial_density

                # True detection hypothesis
                hypotheses.append(
                    SingleProbabilityHypothesis(
                        prediction,
                        detection,
                        probability,
                        measurement_prediction))

        if self.clutter_spatial_density is None:
            for hypothesis in hypotheses[1:]:  # Skip missed detection
                hypothesis.probability *= self._validation_region_volume(
                    self.prob_gate, hypothesis.measurement_prediction) / validated_measurements

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)

    @classmethod
    @lru_cache()
    def _validation_region_volume(cls, prob_gate, meas_pred):
        n = meas_pred.ndim
        gate_threshold = cls._gate_threshold(prob_gate, n)
        c_z = np.pi**(n/2) / gamma(n/2 + 1)
        return c_z * gate_threshold**(n/2) * np.sqrt(det(meas_pred.covar))

    @staticmethod
    @lru_cache()
    def _gate_threshold(prob_gate, n):
        return chi2.ppf(float(prob_gate), n)
