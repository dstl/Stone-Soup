# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.transitionmodel import TransitionModel
from ..models.controlmodel import ControlModel


class Predictor(Base):
    """Predictor base class"""

    transition_model = Property(TransitionModel, doc="transition model")
    control_model = Property(ControlModel, default=None, doc="control model")

    @abstractmethod
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        raise NotImplemented
