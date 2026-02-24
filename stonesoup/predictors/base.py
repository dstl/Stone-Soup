from typing import Sequence, Type

from ..base import Base, Property
from ..models.transition.linear import LinearGaussianTransitionModel
from ..predictor import Predictor


class Predictors(Base):
    predictor_class: Type[Predictor] = Property(doc="The Predictor class to be used.")
    transitions: Sequence[LinearGaussianTransitionModel] = Property(
        doc="The list of transition models to be used.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predictors = []
        for transition in self.transitions:
            self._predictors.append(self.predictor_class(transition))

    @property
    def predictors(self):
        return self._predictors
