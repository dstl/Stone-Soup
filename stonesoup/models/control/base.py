from abc import abstractmethod
from typing import Sequence

from ..base import Model
from ...base import Property


class ControlModel(Model):
    """Control Model base class"""

    #ndim_state: int = Property(doc="Number of state dimensions")
    #mapping: Sequence[int] = Property(doc="Mapping between control and state dims")

    @property
    def ndim(self) -> int:
        return self.ndim_ctrl

    @property
    @abstractmethod
    def ndim_ctrl(self) -> int:
        """Number of control input dimensions"""
        pass
