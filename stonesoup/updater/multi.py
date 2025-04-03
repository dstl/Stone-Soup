"""Updaters designed to wrap existing updaters that may offer a performance benefit
in certain circumstances when dealing with many sensors
"""
import warnings

import numpy as np

from stonesoup.base import Property
from stonesoup.models.measurement.nonlinear import CombinedReversibleGaussianMeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.updater import Updater


class CombineMeasurementUpdater(Updater):
    """Combine measurement updater

    This combines multiple measurements into a single measurement by combining their
    state vectors and models.

    All :class:`~.Detection` must have a measurement model associated with them,
    and have the same associated prediction/timestamp.

    In case where you have multiple measurements from many sensors at the
    same timestamp, this may provide a computation performance increase over
    sequential updates.
    For linear case, this should provide equivalent update to sequential updates.
    However, for a non-linear case will result in non-optimal update due to
    linearisation differences.
    """
    measurement_model = None
    updater: Updater = Property(doc="Base updater being wrapped")

    def predict_measurement(self, *args, **kwargs):
        return self.updater.predict_measurement(*args, **kwargs)

    def update(self, hypothesis: MultipleHypothesis, **kwargs):
        if len(hypothesis) == 1:  # No need to combine
            return self.updater.update(hypothesis[0])

        models = []
        state_vectors = []
        predictions = set()
        for hyp in hypothesis:
            state_vectors.append(hyp.measurement.state_vector)
            models.append(hyp.measurement.measurement_model)
            predictions.add(hyp.prediction)

        if len(predictions) > 1:
            warnings.warn("More than one prediction in combined update")
        prediction = predictions.pop()

        measurement = Detection(
            state_vector=np.vstack(state_vectors),
            timestamp=prediction.timestamp,
            measurement_model=CombinedReversibleGaussianMeasurementModel(models)
        )

        update = self.updater.update(SingleHypothesis(prediction, measurement))

        update.hypothesis = hypothesis

        return update
