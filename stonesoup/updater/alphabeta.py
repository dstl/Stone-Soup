# -*- coding: utf-8 -*-


import numpy as np
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.prediction import StateMeasurementPrediction
from ..types.update import StateUpdate
from ..models.measurement.linear import LinearGaussian


class AlphaBetaUpdater(Updater):
    r"""A class which
    performs measurement update step as in the standard Alpha-Beta Filter.
    .. math::

        \hat{\mathbf{x}}_{k} = \hat{\mathbf{x}}_{k} \alpha \hat{\mathbf{x}}_{k}

        \hat{\mathbf{v}}_{k} = \hat{\mathbf{v}}_{k} \frac{\beta}{\Delta T}\hat{\mathbf{x}}_{k}

    :meth:`predict_measurement` returns a
    :class:`~.StateMeasurementPrediction`.

    """
    alpha = Property(float,
                     default=0.5,
                     doc="Alpha value, required to be 0 < alpha < 1.")
    beta = Property(float,
                    default=0.5,
                    doc="Beta value, required to be 0 < alpha < 1.")

    measurement_model = Property(
        LinearGaussian, default=None,
        doc="A linear Gaussian measurement model. This need not be defined if "
            "a measurement model is provided in the measurement. If no model "
            "specified on construction, or in the measurement, then error "
            "will be thrown.")

    def _check_measurement_model(self, measurement_model):
        """Check that the measurement model passed actually exists. If not
        attach the one in the updater. If that one's not specified, return an
        error.

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

        return measurement_model

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`StateMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state.state_vector,
                                               noise=0, **kwargs)

        return StateMeasurementPrediction(pred_meas, predicted_state.timestamp)

    def update(self, hypothesis, **kwargs):
        r"""The Alpha-Beta update method.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.StateUpdate`
            The posterior state with mean :math:`\mathbf{x}_{k|k}`

        """

        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater (which might still be
            # none)
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(
                measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

        # Get the predicted measurement mean
        residual = (hypothesis.measurement.state_vector - hypothesis.measurement_prediction.state_vector)

        posterior = np.zeros(predicted_state.state_vector.shape)

        posterior[::2] = predicted_state.state_vector[::2] + self.alpha*residual
        posterior[1::2] = predicted_state.state_vector[1::2] + self.beta*residual

        return StateUpdate(posterior,
                           hypothesis,
                           hypothesis.measurement.timestamp)
