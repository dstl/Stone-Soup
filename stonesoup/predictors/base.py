from typing import Sequence, Type

from ..base import Base, Property
from ..models.transition.linear import LinearGaussianTransitionModel
from ..predictor import Predictor


class Predictors(Base):
    """Container for a set of predictors built from a sequence of transition models.

    The Predictors class takes a predictor type and a sequence of transition
    models, and instantiates one predictor per transition model. This is useful
    for filtering approaches that require a predictor instance for each model in
    a multiple model algorithms.
    """
    predictor_class: Type[Predictor] = Property(
        doc="The Predictor class to instantiate for each transition model.")
    transitions: Sequence[LinearGaussianTransitionModel] = Property(
        doc="Sequence of transition models used to create predictor instances.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predictors = []
        for transition in self.transitions:
            self._predictors.append(self.predictor_class(transition))

    @property
    def predictors(self):
        """List of predictor instances created from the configured models."""
        return self._predictors
