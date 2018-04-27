# -*- coding: utf-8 -*-
from .state import State, GaussianState


class Detection(State):
    """Detection type"""


class GaussianDetection(Detection, GaussianState):
    """GaussianDetection type"""
