# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import Type


class GroundTruth(Type):
    """Ground Truth type"""
    state = Property(np.ndarray)

    def __init__(self, state, *args, **kwargs):
        if not state.shape[1] == 1:
            raise ValueError(
                "state shape should be Nx1 dimensions: got {state.shape}")

        super().__init__(state, *args, **kwargs)
