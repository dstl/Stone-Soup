from typing import List

from stonesoup.base import Property
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import Hypothesis
from stonesoup.types.prediction import Prediction, MeasurementPrediction


class MultiHypothesis(Hypothesis):
    """A hypothesis based on multiple measurements. """
    prediction: Prediction = Property(doc="Predicted track state")
    measurements: List[Detection] = Property(doc="Detection used for hypothesis and updating")
    measurement_prediction: MeasurementPrediction = Property(
        default=None, doc="Optional track prediction in measurement space")