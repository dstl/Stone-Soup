# -*- coding: utf-8 -*-
from ..base import Base, Property
from ..models.transitionmodel import TransitionModel
from ..models.measurementmodel import MeasurementModel
from ..models.controlmodel import ControlModel


class Predictor(Base):
    """Predictor base class"""

    transition_model = Property(TransitionModel, doc="transition model")
    measurement_model = Property(MeasurementModel, doc="measurement model")
    control_model = Property(ControlModel, doc="control model")
