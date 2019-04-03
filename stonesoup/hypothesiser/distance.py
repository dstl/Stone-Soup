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

        # Common state & measurement prediction
        prediction = self.predictor.predict(track.state, timestamp=timestamp)
        measurement_prediction = self.updater.get_measurement_prediction(
            prediction)

        # Missed detection hypothesis with distance as 'missed_distance'
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                self.missed_distance,
                measurement_prediction))

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction if detection timestamp does not match default
            if detection.timestamp != timestamp:
                state_prediction = self.predictor.predict(track.state, timestamp=detection.timestamp)
            else:
                state_prediction = prediction

            # Compute measurement prediction and distance measure
            measurement_prediction = self.updater.get_measurement_prediction(
                state_prediction, detection.measurement_model)
            distance = self.measure(measurement_prediction, detection)

            # True detection hypothesis
            hypotheses.append(
                SingleDistanceHypothesis(
                    state_prediction,
                    detection,
                    distance,
                    measurement_prediction))

        return sorted(hypotheses, reverse=True)
