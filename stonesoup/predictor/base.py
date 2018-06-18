# -*- coding: utf-8 -*-
"""Base classes for Stone Soup Predictor interface"""
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..models.control import ControlModel


class Predictor(Base):
    """Predictor base class

    A predictor is used to advance a state to another point in time, by
    utilising a specified :class:`~.TransitionModel`. In addition a
    :class:`~.ControlModel` may be used to model an external influence to the
    state.
    """

    transition_model = Property(TransitionModel, doc="transition model")
    control_model = Property(ControlModel, default=None, doc="control model")

    @abstractmethod
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Predict state

        Parameters
        ----------
        prior : :class:`~.State`
            State
        control_input : :class:`~.State`
            State
        timestamp : :class:`datetime.datetime`
            Time which to predict to which will be passed to transition model

        Returns
        -------
        : :class:`~.StatePrediction`
            State prediction
        """
        raise NotImplemented
