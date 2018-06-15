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
    def get_measurement_prediction(self, state_prediction, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        state_prediction : State
            The state prediction

        Returns
        -------
        State
            The state posterior
        CovarianceMatrix
            The calculated state-to-measurement cross covariance
        """
        raise NotImplemented

    @abstractmethod
    def update(self, prediction, measurement,
               measurement_prediction=None, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        prediction : State
            The state prediction
        measurement : Detection
            The measurement
        measurement_prediction : State
            The measurement prediction

        Returns
        -------
        State
            The state posterior
        """
        raise NotImplemented
