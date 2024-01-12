from abc import abstractmethod

from ..base import Model


class ControlModel(Model):
    """Control Model base class"""

    @property
    def ndim(self) -> int:
        return self.ndim_ctrl

    @property
    @abstractmethod
    def ndim_ctrl(self) -> int:
        """Number of control input dimensions"""
        pass
