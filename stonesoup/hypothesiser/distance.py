# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..measures import Measure
from ..predictor import Predictor
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.detection import MissedDetection
from ..updater import Updater


class DistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`Measure` class.
    """

    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    measure = Property(
        Measure,
        doc="Measure class used to calculate the distance between two states.")
    missed_distance = Property(
        float,
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")

    def hypothesise(self, track, detections, timestamp):

        hypotheses = list()

        for detection in detections:
            prediction = self.predictor.predict(
                track.state, timestamp=detection.timestamp)
            measurement_prediction = self.updater.get_measurement_prediction(
                prediction, detection.measurement_model)
            distance = self.measure(measurement_prediction, detection)

            hypotheses.append(
                SingleDistanceHypothesis(
                    prediction, detection, distance, measurement_prediction))

        # Missed detection hypothesis with distance as 'missed_distance'
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        hypotheses.append(SingleDistanceHypothesis(
            prediction,
            MissedDetection(timestamp=timestamp),
            self.missed_distance))

        return sorted(hypotheses, reverse=True)
