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
    def update(self, state_pred, meas_pred, meas, cross_covar=None, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        state_pred : State
            The state prediction
        meas_pred : State
            The measurement prediction
        meas : Detection
            The measurement
        cross_covar: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The state-to-measurement cross covariance (the default is None, in
            which case ``cross_covar`` will be computed internally)

        Returns
        -------
        State
            The state posterior
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed gain matrix
        """
        raise NotImplemented
