# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurement import MeasurementModel


class Updater(Base):
    """Updater base class

    An updater is used to update the state, utilising a
    :class:`~.MeasurementModel`.
    """

    measurement_model = Property(MeasurementModel, doc="measurement model")

    @abstractmethod
    def predict_measurement(self, state_prediction, **kwargs):
        """Get measurement prediction from state prediction

        Parameters
        ----------
        state_prediction : :class:`~.StatePrediction`
            The state prediction

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The predicted measurement
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, hypothesis, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.State`
            The state posterior
        """
        raise NotImplementedError
