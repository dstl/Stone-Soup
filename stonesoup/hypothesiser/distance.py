# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import mahalanobis

from .base import Hypothesiser
from ..base import Property
from ..types import DistanceHypothesis
from ..predictor import Predictor


class MahalanobisDistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on Mahalanobis Distance

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the mahalanobis distance.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")

    missed_distance = Property(
        int,
        default=4,
        doc="Distance in standard deviations at which a missed detection is"
            "considered more likely. Default is 4 standard deviations.")

    def hypothesise(self, track, detections, timestamp):

        hypotheses = list()

        for detection in detections:
            prediction, innovation, _ = self.predictor.predict(
                track, timestamp=detection.timestamp)
            distance = mahalanobis(detection.state_vector,
                                   innovation.state_vector,
                                   np.linalg.inv(innovation.covar))

            hypotheses.append(
                DistanceHypothesis(prediction, innovation, detection, distance))

        # Missed detection hypothesis with distance as 'missed_distance'
        prediction = self.predictor.predict_state(track,
                                                  timestamp=timestamp)
        hypotheses.append(
            DistanceHypothesis(prediction, None, None, self.missed_distance))

        return sorted(hypotheses, reverse=True)
