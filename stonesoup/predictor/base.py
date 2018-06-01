# -*- coding: utf-8 -*-
"""Base classes for Stone Soup Predictor interface"""
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..models.measurement import MeasurementModel
from ..models.control import ControlModel


class Predictor(Base):
    """Predictor base class

    A predictor is used to advance a state to another point in time,
    by utilising a specified :class:`~.TransitionModel`. A
    :class:`~.MeasurementModel` is also required to allow prediction of the
    measurement state. In addition a :class:`~.ControlModel` may be used to
    model an external influence to the state.
    """

    transition_model = Property(TransitionModel, doc="transition model")
    measurement_model = Property(
        MeasurementModel, doc="measurement model")
    control_model = Property(
        ControlModel, default=None, doc="control model")

    def predict(self, state, control_input=None, timestamp=None, **kwargs):
        """Predict state and measurement

        This returns results of  both :meth:`~.Predictor.predict_state` and
        :meth:`~.Predictor.predict_measurement`.

        Parameters
        ----------
        state : State
            State
        control_input : State
            State
        timestamp : datetime.datetime
            Time which to predict to which will be passed to transition model

        Returns
        -------
        State
            State prediction
        State
            Measurement prediction
        CovarianceMatrix
            Cross-covariance matrix
        """
        state_prediction = self.predict_state(
            state, control_input, timestamp, **kwargs)
        return (state_prediction, *self.predict_measurement(state_prediction))

    @abstractmethod
    def predict_state(self, state, control_input=None, timestamp=None,
                      **kwargs):
        """Predict state

        Parameters
        ----------
        state : State
            State
        control_input : State
            State
        timestamp : datetime.datetime
            Time which to predict to which will be passed to transition model.

        Returns
        -------
        State
            State prediction.

         """
        raise NotImplemented

    @abstractmethod
    def predict_measurement(self, state, **kwargs):
        """Predict measurement state

        Parameters
        ----------
        state : State
            Predicted state

        Returns
        -------
        State
            Predicted measurement state
        """
        raise NotImplemented
