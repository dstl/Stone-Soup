"""Updaters designed to wrap existing updaters that may offer a performance benefit
in certain circumstances when dealing with many sensors
"""
import copy
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np

from stonesoup.base import Property
from stonesoup.models.measurement.nonlinear import CombinedReversibleGaussianMeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.update import Update
from stonesoup.updater import Updater
from stonesoup.updater.kalman import KalmanUpdater


class ParallelUpdater(Updater):
    """Parallel Updater

    All :class:`~.Detection` must have the same associated prediction/timestamp.

    In case where you have multiple measurements from many sensors at the
    same timestamp, this may provide a computation performance increase over
    sequential updates with :attr:`multiprocessing` enabled.
    For linear case, this should provide equivalent update to sequential updates.
    However, for a non-linear case will result in non-optimal update due to
    linearisation differences.
    """
    measurement_model = None
    updater: KalmanUpdater = Property(doc="Base updater to be called in parallel")
    multiprocessing: bool = Property(default=False, doc="Use multiprocessing Pool for updates")

    def predict_measurement(self, *args, **kwargs):
        return self.updater.predict_measurement(*args, **kwargs)

    def update(self, hypothesis: MultipleHypothesis, **kwargs):
        if len(hypothesis) == 1:  # No need for parallelism
            return self.updater.update(hypothesis[0])

        predictions = {hyp.prediction for hyp in hypothesis}
        if len(predictions) > 1:
            warnings.warn("More than one prediction in parallel update")
        prediction = copy.copy(predictions.pop())
        prediction.covar = prediction.covar * len(hypothesis)

        # this can be computed in parallel
        new_hypotheses = (SingleHypothesis(prediction, hyp.measurement) for hyp in hypothesis)
        if self.multiprocessing:
            with Pool() as p:
                updates = p.map(partial(self.updater.update, **kwargs), new_hypotheses)
        else:
            updates = [self.updater.update(hyp, **kwargs) for hyp in new_hypotheses]

        # here results are collected and fused
        fused_covar = np.linalg.inv(
            np.sum(np.array([np.linalg.inv(upd.covar) for upd in updates]), axis=0))
        fused_state = fused_covar @ np.sum(
            np.array([np.linalg.inv(upd.covar) @ upd.state_vector for upd in updates]), axis=0)

        return Update.from_state(
            prediction,
            state_vector=fused_state,
            covar=fused_covar,
            hypothesis=hypothesis
        )


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
