from typing import Sequence

from ..base import Property
from .array import StateVector
from .base import Type


class Particle(Type):
    """
    Particle type

    A particle type which contains a state and weight
    """
    state_vector: StateVector = Property(doc="State vector")
    weight: float = Property(doc='Weight of particle')
    parent: 'Particle' = Property(default=None, doc='Parent particle')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.parent:
            self.parent.parent = None
        if self.state_vector is not None and not isinstance(self.state_vector, StateVector):
            self.state_vector = StateVector(self.state_vector)

    @property
    def ndim(self):
        return self.state_vector.shape[0]


class MultiModelParticle(Particle):
    """
    Particle type

    A MultiModelParticle type which contains a state, weight and the dynamic_model
    """
    dynamic_model: int = Property(doc='Assigned dynamic model')
    parent: 'MultiModelParticle' = Property(default=None, doc='Parent particle')


class RaoBlackwellisedParticle(Particle):
    """
    Particle type

    A RaoBlackwellisedParticle type which contains a state, weight, dynamic_model and
    associated model probabilities
    """
    model_probabilities: Sequence[float] = Property(
        doc="The dynamic probabilities of changing models")
    parent: 'RaoBlackwellisedParticle' = Property(default=None, doc='Parent particle')
