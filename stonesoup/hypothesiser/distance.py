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

    def hypothesise(self, track, detections, time):

        hypotheses = set()

        for detection in detections:
            prediction, innovation, _ = self.predictor.predict(track, time=detection.timestamp)
            distance = mahalanobis(detection, prediction.state_vector, np.linalg.inv(prediction.covar))

            hypotheses.add(DistanceHypothesis(prediction, innovation, detection, distance))

        prediction = self.predictor.predict_state(track, time=time)
        hypotheses.add(DistanceHypothesis(prediction, None, None, self.missed_distance))

        return hypotheses
