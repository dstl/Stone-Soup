from copy import copy
from typing import Union, Callable

import numpy as np
from scipy.stats import multivariate_normal as mn

from stonesoup.base import Property
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State


class IPDAHypothesiser(PDAHypothesiser):

    """ Integrated PDA Hypothesiser """

    prob_detect: Union[Probability, Callable[[State], Probability]] = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    predict: bool = Property(default=True, doc="Perform prediction step")
    per_measurement: bool = Property(default=False, doc="Generate per measurement predictions")

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
        for detection in detections:
            if self.predict and self.per_measurement:
                # Re-evaluate prediction
                prediction = self.predictor.predict(
                    track.state, timestamp=detection.timestamp)
                prob_detect = self.prob_detect(prediction)
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