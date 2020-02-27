# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurement import MeasurementModel


class Updater(Base):
    r"""Updater base class

    An updater is used to update the predicted state, utilising a measurement
    and a :class:`~.MeasurementModel`.  The general observation model is

    .. math::

        \mathbf{z} = h(\mathbf{x}, \mathbf{\sigma})

    where :math:`\mathbf{x}` is the state, :math:`\mathbf{\sigma}`, the
    measurement noise and :math:`\mathbf{z}` the resulting measurement.

    """

    measurement_model = Property(MeasurementModel, doc="measurement model")

    @abstractmethod
    def predict_measurement(
            self, state_prediction, measurement_model=None, prior_timestamp=None,
            transition=None, always_resample=True, **kwargs):
        """Get measurement prediction from state prediction

        Parameters
        ----------
        state_prediction : :class:`~.StatePrediction`
            The state prediction
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.
            Should be used in cases where the measurement model is dependent
            on the received measurement. The default is `None`, in which case
            the updater will use the measurement model specified on
            initialisation
        prior_timestamp: :class: `~.datetime.datetime`
            the timestamp associated with the prior measurement state.
        transition: :np.array:
            a non block diagonal transition_matrix example:
                [[0.97 0.01 0.01 0.01]
                 [0.01 0.97 0.01 0.01]
                 [0.01 0.01 0.97 0.01]
                 [0.01 0.01 0.01 0.97]]
            which would represent using four models.
        always_resample: :Boolean:
            if True, then the particle filter will resample every time step.
            Otherwise, will only resample when 25% or less of the particles
            are deemed effective.
            Calculated by 1 / sum(particle.weight^2) for all particles

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The predicted measurement
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, hypothesis, predictor=None, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.
        predictor: :class:`~.Predictor`
            Predictor which holds the transition matrix, dynamic models and the
            mapping rules.
            Optional

        Returns
        -------
        : :class:`~.State`
            The state posterior
        """
        raise NotImplementedError
