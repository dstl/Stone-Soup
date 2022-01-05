# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from ..models.measurement.categorical import CategoricalMeasurementModel
from ..types.prediction import MeasurementPrediction
from ..types.state import CategoricalState
from ..types.update import Update
from ..updater import Updater


class HMMUpdater(Updater):
    r"""Models the update step of a hidden Markov model"""

    measurement_model: CategoricalMeasurementModel = Property(
        default=None,
        doc="An observation-based measurement model. Measurements are assumed to be as defined "
            "above. This model need not be defined if a measurement model is provided in the "
            "measurement. If no model specified on construction, or in the measurement, then an "
            "error will be thrown.")

    def _check_measurement_model(self, measurement_model):
        """Check that the measurement model passed actually exists. If not attach the one in the
        updater. If that one's not specified, return an error.

        Parameters
        ----------
        measurement_model : :class`~.MeasurementModel`
            A measurement model to be checked

        Returns
        -------
        : :class`~.MeasurementModel`
            The measurement model to be used
        """

        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        try:
            measurement_model.emission_matrix
        except AttributeError:
            raise ValueError("Measurement model must be categorical. I.E. it must have an "
                             "Emission matrix property for the HMMUpdater")

        return measurement_model

    def _get_emission_matrix(self, hypothesis):

        measurement_model = self._check_measurement_model(hypothesis.measurement.measurement_model)

        return measurement_model.emission_matrix

    def predict_measurement(self, predicted_state, measurement_model=None, category_names=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`X_{k|k-1}`.
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object is used.
        category_names : :class:`list`
            List of :class:`str` measurement category names.
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`.

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The measurement prediction, :math:`Y_{k|k-1}`
        """

        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        return MeasurementPrediction.from_state(predicted_state, pred_meas,
                                                category_names=category_names)

    def update(self, hypothesis, **kwargs):
        r"""The update method. Given a hypothesised association between a predicted state or
        predicted measurement and an actual measurement, calculate the posterior state.

        Bayes' rule: :math:`p(x_k|z_{1:k}) \propto p(z_k|x_k) p(x_k|z_{1:k-1})`. The likelihood is
        calculated as an Nx1 vector of :math:`p(z_k|x_{k|k-1})` for the :math:`z_k` actually
        observed.
        This is element-wise multiplied by the prior and normalised.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis may carry a
            predicted measurement, or a predicted state. In the latter case a predicted
            measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.CategoricalStateUpdate`
            The posterior categorical state
        """

        prediction = hypothesis.prediction

        if not isinstance(prediction, CategoricalState):
            raise ValueError("Prediction must be a categorical state type")

        # category names to be passed to measurement prediction
        measurement_category_names = hypothesis.measurement.category_names

        if hypothesis.measurement_prediction is None:
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                prediction, measurement_model=measurement_model,
                category_names=measurement_category_names, **kwargs)

        emission_matrix = self._get_emission_matrix(hypothesis)
        likelihood = emission_matrix @ hypothesis.measurement.state_vector

        # Bayes rule
        posterior = np.multiply(likelihood, hypothesis.prediction.state_vector)
        posterior = posterior / np.sum(posterior)

        return Update.from_state(hypothesis.prediction, posterior,
                                 timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
