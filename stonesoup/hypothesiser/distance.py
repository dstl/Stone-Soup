# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import mahalanobis

from .base import Hypothesiser
from ..base import Property
from ..types import DistanceHypothesis
from ..predictor import Predictor
from ..updater import Updater


class MahalanobisDistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on Mahalanobis Distance

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the mahalanobis distance.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    missed_distance = Property(
        int,
        default=4,
        doc="Distance in standard deviations at which a missed detection is"
            "considered more likely. Default is 4 standard deviations.")

    def hypothesise(self, track, detections, timestamp):

        hypotheses = list()

        for detection in detections:
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp)
            measurement_prediction, _ = self.updater.get_measurement_prediction(
                prediction)
            distance = mahalanobis(detection.state_vector,
                                   measurement_prediction.state_vector,
                                   np.linalg.inv(measurement_prediction.covar))

            hypotheses.append(
                DistanceHypothesis(
                    prediction, measurement_prediction, detection, distance))

        # Missed detection hypothesis with distance as 'missed_distance'
        prediction = self.predictor.predict(track, timestamp=timestamp)
        hypotheses.append(
            DistanceHypothesis(prediction, None, None, self.missed_distance))

        return sorted(hypotheses, reverse=True)
