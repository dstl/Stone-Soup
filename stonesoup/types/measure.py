# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import Type


class Measure(Type):
    """Measure base type

    A measure provides a means to assess the seperation between two
    :class:'~.State' objects X and Y.
    """
    threshold = Property(
        float,
        default=None,
        doc="Threshold to be applied to distance measure")
    mapping = Property(
        [np.array],
        doc="Mapping array which specifies which elements within the state \
             vectors are to be assessed as part of the measure"
    )

    def __get__(self, x, y):
        return NotImplemented


class Euclidean(Measure):
    """Euclidean distance measure

    This measure returns the euclidean distance between the
    :class:'~StateVector' information within the :class:'~.State' objects X and
     Y.
    """
    def __get__(self, x, y):

        # Calculate Euclidean distance between two state
        dist = np.linalg.norm(x.state_vector[self.mapping] -
                              y.state_vector[self.mapping])

        if (self.threshold is None) or (dist < self.threshold):
            return dist
        else:
            return None



