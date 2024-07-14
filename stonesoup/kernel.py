from abc import abstractmethod

import numpy as np

from .base import Base, Property
from .types.array import StateVectors
from .types.state import State


class Kernel(Base):
    """Kernel base type

    A Kernel provides a means to translate state space or measurement space into kernel space.
    """

    @abstractmethod
    def __call__(self, state1, state2=None):
        r"""
        Compute the kernel state of a pair of :class:`~.StateVectors` objects

        Parameters
        ----------
        state1 : :class:`~.StateVectors`
        state2 : :class:`~.StateVectors`

        Returns
        -------
        StateVectors
            kernel state of a pair of input :class:`~.StateVectors` objects

        """
        raise NotImplementedError

    @staticmethod
    def _get_state_vectors(state1, state2):
        if isinstance(state1, State):
            state_vector1 = state1.state_vector
        else:
            state_vector1 = state1
        if state2 is None:
            state_vector2 = state_vector1
        else:
            if isinstance(state2, State):
                state_vector2 = state2.state_vector
            else:
                state_vector2 = state2
        return state_vector1, state_vector2


class PolynomialKernel(Kernel):
    r"""Polynomial Kernel

    This kernel returns the polynomial kernel of order :math:`p` state from a pair of
    :class:`.StateVectors` objects.

    The polynomial kernel of state vectors :math:`\mathbf{x}` and :math:`\mathbf{x}^\prime` is
    defined as:

    .. math::
        \mathtt{k}(\mathbf{x}, \mathbf{x}^\prime) =
        \left(\alpha \left\langle \mathbf{x}, \mathbf{x}^\prime \right\rangle + c \right) ^ p
    """

    power: int = Property(doc="The polynomial power :math:`p`.")
    c: float = Property(default=1,
                        doc="Free parameter trading off the influence of higher-order versus "
                            "lower-order terms in the polynomial. Default is 1.")
    ialpha: float = Property(default=1e1, doc="Slope. Range is [1e0, 1e4]. Default is 1e1.")

    def __call__(self, state1, state2=None):
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)
        return (state_vector1.T @ state_vector2 / self.ialpha + self.c) ** self.power
    

class QuadraticKernel(Kernel):
    r"""Quadratic Kernel type

    This kernel returns the quadratic kernel state vector from a pair of
    :class:`~.StateVectors` state vectors.

    The Quadratic kernel of state vectors :math:`\mathbf{x}` and
    :math:`\mathbf{x}'` is defined as:

    .. math::
         \mathtt{k}\left(\mathbf{x}, \mathbf{x}'\right) =
         \left(\alpha \langle \mathbf{x}, \mathbf{x}' \rangle + c\right)^2
    """
    c: float = Property(
        default=1,
        doc="Free parameter trading off the influence of higher-order versus lower-order "
            "terms in the polynomial. Default is 1.")
    ialpha: float = Property(default=1e1, doc="Slope. Range is [1e0, 1e4].")

    def __call__(self, state1, state2=None):
        r"""Calculate the Quadratic Kernel transformation for a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.StateVectors`
        state2 : :class:`~.StateVectors`

        Returns
        -------
        StateVectors
            Transformed state vector in kernel space.
        """
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)
        return (state_vector1.T@state_vector2/self.ialpha + self.c) ** 2


class QuarticKernel(Kernel):
    r"""Quartic Kernel

    This kernel returns the quartic kernel state from a pair of
    :class:`~.StateVectors` objects.

    The Quartic kernel of state vectors :math:`\mathbf{x}` and
    :math:`\mathbf{x}'` is defined as:

    .. math::
         \mathtt{k}(\mathbf{x}, \mathbf{x}') =
         \left(\alpha \langle \mathbf{x}, \mathbf{x}' \rangle + c\right)^4
    """
    c: float = Property(
        default=1,
        doc="Free parameter trading off the influence of higher-order versus lower-order "
            "terms in the polynomial. Default is 1.")
    ialpha: float = Property(default=1e1, doc="Slope. Range is [1e0, 1e4].")

    def __call__(self, state1, state2=None):
        r"""Calculate the Quartic Kernel transformation for a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.StateVectors`
        state2 : :class:`~.StateVectors`

        Returns
        -------
        StateVectors
            Transformed state in kernel space.
        """
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)
        return (state_vector1.T@state_vector2/self.ialpha + self.c) ** 4


class GaussianKernel(Kernel):
    r"""Gaussian Kernel

    This kernel returns the Gaussian kernel state vector from a pair of
    :class:`~.StateVectors` objects.

    The Gaussian kernel of state vectors :math:`\mathbf{x}` and
    :math:`\mathbf{x}'` is defined as:

    .. math::
         \mathtt{k}(\mathbf{x}, \mathbf{x}') =
         \mathrm{exp}\left(-\frac{||\mathbf{x} - \mathbf{x}'||^{2}}{2\pi\sigma^2}\right)
    """
    variance: float = Property(
        default=1e1,
        doc=r"Denoted as :math:`\sigma^2` in the equation above. Determines the width of the "
            r"Gaussian kernel. Range is [1e0, 1e2].")

    def __call__(self, state1, state2=None):
        r"""Calculate the Gaussian Kernel transformation for a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.StateVectors`
        state2 : :class:`~.StateVectors`

        Returns
        -------
        StateVectors
            Transformed state vector in kernel space.
        """
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)
        diff_tilde_x = (state_vector1.T[:, :, None] - state_vector2.T[:, None, :]) ** 2
        diff_tilde_x_sum = np.sum(diff_tilde_x, axis=0)

        k_tilde_x = np.exp(-diff_tilde_x_sum/(2*self.variance)) / np.sqrt(2*np.pi*self.variance)

        return StateVectors(k_tilde_x)
