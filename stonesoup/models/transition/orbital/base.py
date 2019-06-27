from abc import abstractmethod

from ..base import TransitionModel


class OrbitalModel(TransitionModel):
    """Orbital Transition Model base class"""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass
