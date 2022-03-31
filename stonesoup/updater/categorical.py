# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from ..models.measurement.categorical import MarkovianMeasurementModel
from ..types.prediction import MeasurementPrediction
from ..types.update import Update
from ..updater import Updater


class HMMUpdater(Updater):
    r"""Hidden Markov model updater"""

    measurement_model: MarkovianMeasurementModel = Property(
        default=None,
        doc="The measurement model used to predict measurement vectors. If no model is specified "
            "on construction, or in a measurement, then an error will be thrown.")

    def update(self, hypothesis, **kwargs):
        r"""The update method. Given a hypothesised association between a predicted state or
        predicted measurement and an actual measurement, calculate the posterior state.

        .. math::
            \alpha_t^i = E^{ki}(F\alpha_{t-1})^i

        Measurements are assumed to be discrete categories from a finite set of measurement
        categories :math:`Z = \{\zeta^n|n\in \mathbf{N}, n\le N\} (for some finite :math:`N`).
        A measurement should be equivalent to a basis vector :math:`e^k`, (the N-tuple with all
        components equal to 0, except the k-th (indices starting at 0), which is 1). This
        indicates that the measured category is :math:`\zeta^k`.

        The equation above can be simplified to:

        .. math::
            \alpha_t = E^Ty_t \circ F\alpha_{t-1}

        where :math:`\circ` denotes element-wise (Hadamard) product.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis may carry a
            predicted measurement, or a predicted state. In the latter case a predicted
            measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`.

        Returns
        -------
        : :class:`~.CategoricalStateUpdate`
            The posterior categorical state.
        """

        prediction = hypothesis.prediction
        measurement = hypothesis.measurement
        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(measurement_model)

        if hypothesis.measurement_prediction is None:
            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state=prediction,
                measurement_model=measurement_model,
                measurement=measurement,
                **kwargs
            )

        emission_matrix = measurement_model.emission_matrix
        likelihood = emission_matrix.T @ measurement.state_vector

        posterior = np.multiply(likelihood, hypothesis.prediction.state_vector)
        posterior = posterior / np.sum(posterior)

        return Update.from_state(hypothesis.prediction, posterior,
                                 timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)

    def _check_measurement_model(self, measurement_model):
        """Check that the measurement model passed actually exists. If not attach the one in the
        updater. If that one is not specified, raise an error.

        Parameters
        ----------
        measurement_model : :class`~.MeasurementModel`
            A measurement model to be checked.

        Returns
        -------
        : :class`~.MeasurementModel`
            The measurement model to be used.
        """

        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        if not isinstance(measurement_model, MarkovianMeasurementModel):
            raise ValueError(
                "HMMUpdater must be used in conjuction with HiddenMarkovianMeasurementModel types"
            )

        return measurement_model

    def predict_measurement(self, predicted_state, measurement_model, **kwargs):
        r"""Predict the measurement implied by the predicted state.

        Parameters
        ----------
        predicted_state : :class:`~.CategoricalState`
            The predicted state..
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object is used.
        measurement : :class:`~.CategoricalState`.
            The measurement.
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function`.

        Returns
        -------
        : :class:`~.CategoricalMeasurementPrediction`
            The measurement prediction.
        """

        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        return MeasurementPrediction.from_state(
            predicted_state,
            pred_meas,
            categories=measurement_model.measurement_categories
        )
