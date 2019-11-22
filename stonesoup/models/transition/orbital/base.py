from abc import abstractmethod

from ..base import TransitionModel


class OrbitalTransitionModel(TransitionModel):
    """Orbital Transition Model base class. This class will execute a
    transition model on an orbital element state vector. Input is an
    :class:~`OrbitalState`, and the various daughter classes will
    implement their chosen state transitions."""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass
