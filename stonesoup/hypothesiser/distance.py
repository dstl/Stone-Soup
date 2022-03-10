# -*- coding: utf-8 -*-
from typing import Mapping

from .base import Hypothesiser
from ..base import Property
from ..measures import Measure
from ..models.measurement import MeasurementModel
from ..predictor import Predictor
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleDistanceHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..updater import Updater


class DistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a Measure

    Generate track predictions at detection times and score each hypothesised
    prediction-detection pair using the distance of the supplied
    :class:`~.Measure` class.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    measure: Measure = Property(
        doc="Measure class used to calculate the distance between two states.")
    missed_distance: float = Property(
        default=float('inf'),
        doc="Distance for a missed detection. Default is set to infinity")
    include_all: bool = Property(
        default=False,
        doc="If `True`, hypotheses beyond missed distance will be returned. Default `False`")

    def hypothesise(self, track, detections, timestamp, **kwargs):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a
        MultipleHypothesis object with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated distance measure..

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
            A container of :class:`~SingleDistanceHypothesis` objects

        """
        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
        # Missed detection hypothesis with distance as 'missed_distance'
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                self.missed_distance
                ))

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp, **kwargs)

            # Compute measurement prediction and distance measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model, **kwargs)
            distance = self.measure(measurement_prediction, detection)

            if self.include_all or distance < self.missed_distance:
                # True detection hypothesis
                hypotheses.append(
                    SingleDistanceHypothesis(
                        prediction,
                        detection,
                        distance,
                        measurement_prediction))

        return MultipleHypothesis(sorted(hypotheses, reverse=True))


class MultiDistanceHypothesiser(Hypothesiser):
    """Prediction Hypothesiser based on a set of Measures

    Similar to the :class:`~.DistanceHypothesiser` this class scores prediction-detection pairs
    using their distance in a measurement space. However, this class attempts to use the
    detection's measurement model to determine a suitable Measure and distance to use. These are
    provided by the user.
    To do so, the class utilises a list of :class:`~.DistanceHypothesiser` and corresponding
    measurement models. When a detection is received of particular model, the class will attempt
    to find the model's corresponding hypothesiser and use its output.
    """

    hypothesiser_dict: Mapping[type[MeasurementModel], DistanceHypothesiser] = Property(
        doc="Mapping of measurement model class - hypothesiser pairs. Each incoming detection's "
            "measurement model will be compared against this mapping."
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def hypothesise(self, track, detections, timestamp, **kwargs):
        """Hypothesise track-detections pairs using detections' corresponding hypothesisers from
        :attr:`hypothesiser_dict`. Raises error if an incoming detection as no measurement model,
        or a model with no corresponding hypothesiser. Resultant hypothesis collection will
        contain a null hypothesis from each hypothesiser."""

        # Keep track of detections attributed to each hypothesiser
        hypothesisers_detections = {
            hypothesiser: set() for hypothesiser in self.hypothesiser_dict.values()
        }

        # True detection hypotheses
        for detection in detections:

            detection_measurement_model = detection.measurement_model

            relevant_hypothesiser = self.hypothesiser_dict.get(type(detection_measurement_model))

            if relevant_hypothesiser is None:
                # detection either doesn't have measurement model
                # or its model is not accounted for in hypothesiser_dict
                print(f"{detection_measurement_model =}")
                if detection_measurement_model is None:
                    raise ValueError("Hypothesiser received detection with no measurement model")
                else:
                    raise ValueError(f"Hypothesiser received detection with measurement model of "
                                     f"type {type(detection_measurement_model)} which has no "
                                     f"corresponding hypothesiser")

            hypothesisers_detections[relevant_hypothesiser].add(detection)

        hypotheses = list()
        for hypothesiser, relevant_detections in hypothesisers_detections.items():
            multi_hypothesis = hypothesiser.hypothesise(track,
                                                        relevant_detections,
                                                        timestamp,
                                                        **kwargs)
            hypotheses.extend(multi_hypothesis.single_hypotheses)

        return MultipleHypothesis(sorted(hypotheses, reverse=True))
