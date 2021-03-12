# -*- coding: utf-8 -*-
from .base import Updater
from .kalman import KalmanUpdater, ExtendedKalmanUpdater
from ..models.base import LinearModel


class UpdaterWrapper(Updater):
    r"""
    """

    linear_updater = KalmanUpdater(measurement_model=None)
    non_linear_updater = ExtendedKalmanUpdater(measurement_model=None)

    @staticmethod
    def _pick_updater(self, measurement_model):
        if isinstance(measurement_model, LinearModel):
            return self.linear_updater
        else:
            return self.non_linear_updater

    def predict_measurement(self, predicted_state, measurement_model, **kwargs):
        updater = self._pick_updater(self, measurement_model)
        return updater.predict_measurement(predicted_state, measurement_model, **kwargs)

    def soft_predict_measurement(self, predicted_state, GM_measurement, measurement_model,
                                 **kwargs):
        updater = self._pick_updater(self, measurement_model)
        return updater.soft_predict_measurement(predicted_state, GM_measurement,
                                                measurement_model, **kwargs)

    def update(self, hypothesis, **kwargs):
        updater = self._pick_updater(self, hypothesis.measurement.measurement_model)
        return updater.update(hypothesis, **kwargs)

    def soft_update(self, hypothesis, **kwargs):
        updater = self._pick_updater(self, hypothesis.measurement.measurement_model)
        return updater.soft_update(hypothesis, **kwargs)
