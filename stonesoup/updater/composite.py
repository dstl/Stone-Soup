# -*- coding: utf-8 -*-
from typing import Sequence

from ..types.hypothesis import CompositeHypothesis
from .base import Updater
from ..base import Property
from ..types.prediction import CompositePrediction, CompositeMeasurementPrediction
from ..types.update import Update, CompositeUpdate


class CompositeUpdater(Updater):
    """A composition of multiple sub-updaters

    Updates a :class:`~.CompositeState` composed of a sequence of predictions using a
    :class:`~.CompositeDetection` composed of a sequence of measurements using a sequence of
    sub-updaters
    """
    sub_updaters: Sequence[Updater] = Property(doc="A sequence of sub-updaters")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.sub_updaters, Sequence):
            raise ValueError("sub-updaters must be defined as an ordered list")

        if any(not isinstance(sub_updater, Updater) for sub_updater in self.sub_updaters):
            raise ValueError("all sub-updaters must be an Updater type")

    @property
    def measurement_model(self):
        raise NotImplementedError("A composition of updaters have no defined measurement model")

    def predict_measurement(self, predicted_state, measurement_models, **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted composite state
        measurement_models : :class:`~.MeasurementModel`
            A sequence of measurement models to predict each corresponding sub-prediction with.
            This cannot be omitted
        **kwargs : various
            These are passed to each sub-updater's predict measurement method

        Returns
        -------
        : :class:`MeasurementPrediction`
            The composite measurement prediction
        """
        measurement_predictions = list()
        for sub_updater, sub_prediction, measurement_model in zip(self.sub_updaters,
                                                                  predicted_state,
                                                                  measurement_models):
            measurement_predictions.append(
                sub_updater.predict_measurement(sub_prediction, measurement_model, **kwargs)
            )
        return CompositeMeasurementPrediction(measurement_predictions)

    def update(self, hypothesis: CompositeHypothesis, **kwargs):

        sub_updates = []
        is_prediction = True

        if not isinstance(hypothesis, CompositeHypothesis):
            raise ValueError("CompositeUpdater can only be used with CompositeHypothesis types")
        if len(hypothesis.sub_hypotheses) != len(self.sub_updaters):
            raise ValueError("CompositeHypothesis must be composed of same number of"
                             " sub-hypotheses as sub-updaters")

        prediction = hypothesis.prediction
        measurement_models = [sub_hypothesis.measurement.measurement_model
                              for sub_hypothesis in hypothesis.sub_hypotheses]

        # generate measurement prediction
        if hypothesis.measurement_prediction is None:
            measurement_prediction = \
                self.predict_measurement(prediction, measurement_models, **kwargs)
            hypothesis.measurement_prediction = measurement_prediction

        for sub_updater, sub_hypothesis, sub_meas_pred in zip(self.sub_updaters,
                                                              hypothesis.sub_hypotheses,
                                                              measurement_prediction):
            if sub_hypothesis.measurement_prediction is None:
                sub_hypothesis.measurement_prediction = sub_meas_pred

            sub_update = sub_updater.update(sub_hypothesis, **kwargs)
            sub_updates.append(sub_update)

            if isinstance(sub_update, Update):
                is_prediction = False

        if is_prediction:
            return CompositePrediction(sub_states=sub_updates)

        return CompositeUpdate(sub_states=sub_updates, hypothesis=hypothesis)

    def __getitem__(self, index):
        return self.sub_updaters.__getitem__(index)

    def __setitem__(self, index, value):
        return self.sub_updaters.__setitem__(index, value)

    def __delitem__(self, index):
        return self.sub_updaters.__delitem__(index)

    def __iter__(self):
        return iter(self.sub_updaters)

    def __len__(self):
        return self.sub_updaters.__len__()

    def __contains__(self, item):
        return self.sub_updaters.__contains__(item)

    def insert(self, index, value):
        inserted_state = self.sub_updaters.insert(index, value)
        return inserted_state

    def append(self, value):
        """Add value at end of :attr:`updaters`.

        Parameters
        ----------
        value: :class:`~.Updater`
            An Updater to be added at the end of :attr:`sub_updaters`.
        """
        self.sub_updaters.append(value)
