# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from ..models.measurement.observation import BasicTimeInvariantObservationModel
from ..types.prediction import MeasurementPrediction
from ..types.update import Update
from ..updater import Updater


class ClassificationUpdater(Updater):
    r"""Models the update step of a forward algorithm as such:

    .. math::
        X_{k|k}^i &= P(\phi_k^i)\\
                  &= P(\phi_k^i| y_k^j)P(y_k^j)P(\phi_k^i| \phi_{k-1}^l)P(\phi_{k-1}^l)\\
                  &= E^{ij}Z_k^jF^{il}X_{k-1}^l\\
                  &= EZ_k*TX_{k-1}

    Where:

    * :math:`X_{k|k}^i` is the :math:`i`-th component of the posterior state vector, representing
      the probability :math:`P(\phi_k^i)` that the state is class :math:`i` at 'time'
      :math:`k`, with the possibility of the state being any of a finite, discrete set of classes
      :math:`\{\phi^i | i \in \mathbb{Z}_{>0}\}`
    * :math:`y_j` is the :math:`j`-th component of the measurement vector :math:`Z_k`, an
      observation of the state, assumed to be defining a multinomial distribution over a discrete,
      finite set of possible measurement classes :math:`\{y^j | j \in \mathbb{Z}_{>0}\}`
    * :math:`E` defines the time invariant emission matrix of the corresponding measurement model
    * :math:`F` is the stochastic matrix defining the time invariant class transition
      :math:`P(\phi_k^i| \phi_{k-1}^l)`
    * :math:`*` is the element-wise product of two vectors
    * :math:`k` is the 'time' of update, attained from the 'time' at which the measurement
      :math:`Z_k` has been received
    """

    measurement_model: BasicTimeInvariantObservationModel = Property(
        default=None,
        doc="An observation-based measurement model. Measurements are assumed to be as defined "
            "above. This model need not be defined if a measurement model is provided in the "
            "measurement. If no model specified on construction, or in the measurement, then an "
            "error will be thrown.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.measurement_model and not isinstance(self.measurement_model,
                                                     BasicTimeInvariantObservationModel):
            raise ValueError("ClassificationUpdater requires a time invariant observation-based "
                             "measurement model")

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

        return measurement_model

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.State`
            The predicted state :math:`X_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`MeasurementPrediction`
            The measurement prediction, :math:`Z_{k|k-1}`
        """

        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        return MeasurementPrediction.from_state(predicted_state, pred_meas)

    def update(self, hypothesis, **kwargs):
        r"""The update method. Given a hypothesised association between a predicted state or
        predicted measurement and an actual measurement, calculate the posterior state.

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
        : :class:`~.StateUpdate`
            The posterior state with mean :math:`X_{k|k}`
        """

        prediction = hypothesis.prediction

        try:
            E = hypothesis.measurement.measurement_model.emission_matrix
        except AttributeError:
            raise ValueError("All detections must be from class observer measurement models")

        EY = E @ hypothesis.measurement.state_vector

        prenormalise = np.multiply(prediction.state_vector, EY)

        normalise = prenormalise / np.sum(prenormalise)

        return Update.from_state(hypothesis.prediction, normalise,
                                 timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
