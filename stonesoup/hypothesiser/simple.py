import datetime

from typing import Set

from stonesoup.base import Property
from stonesoup.hypothesiser import Hypothesiser
from stonesoup.predictor import Predictor
from stonesoup.types.detection import MissedDetection, Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.track import Track
from stonesoup.updater import Updater


class SimpleHypothesiser(Hypothesiser):
    """Simple Hypothesiser class

    Generate track predictions at detection times and create hypotheses for
    each detection, as well as a missed detection hypothesis.
    """
    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(
        default=None,
        doc="Updater used to get measurement prediction. Only required if "
            "`predict_measurement` is `True`. Default is `None`")
    check_timestamp: bool = Property(
        default=True,
        doc="Check that all detections have the same timestamp. Default is `True`")
    predict_measurement: bool = Property(
        default=False,
        doc="Predict measurement for each detection. Default is `True`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.predict_measurement and self.updater is None:
            raise ValueError("Updater must be provided if `predict_measurement` is `True`")

    def hypothesise(self, track: Track, detections: Set[Detection], timestamp: datetime.datetime,
                    **kwargs) -> MultipleHypothesis:
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a
        MultipleHypothesis object with N+1 detections (first detection is
        a 'MissedDetection').

        Parameters
        ----------
        track : Track
            The track object to hypothesise on
        detections : set of :class:`~.Detection`
            The available detections
        timestamp : datetime.datetime
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non-empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleHypothesis` objects

        """

        if self.check_timestamp:
            # Check to make sure all detections are obtained from the same time
            timestamps = set([detection.timestamp for detection in detections])
            if len(timestamps) > 1:
                raise ValueError("All detections must have the same timestamp")

        hypotheses = []

        # Common state prediction
        prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)

        # Missed detection hypothesis
        hypotheses.append(
            SingleHypothesis(prediction, MissedDetection(timestamp=timestamp))
        )

        # True detection hypotheses
        for detection in detections:

            # Re-evaluate prediction
            prediction = self.predictor.predict(track, timestamp=detection.timestamp, **kwargs)

            # Compute measurement prediction
            if self.predict_measurement:
                measurement_prediction = self.updater.predict_measurement(
                    prediction, timestamp=detection.timestamp, **kwargs)
            else:
                measurement_prediction = None

            hypotheses.append(
                SingleHypothesis(prediction, detection, measurement_prediction)
            )

        return MultipleHypothesis(hypotheses)
