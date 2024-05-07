from abc import abstractmethod

import numpy as np
from stonesoup.types.array import StateVectors
from .base import Base, Property


class Kernel(Base):
    """Kernel base type

    A Kernel provides a means to translate state space or measurement space into kernel space.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, state1, state2):
        r"""
        Compute the kernel state of a pair of :class:`~.State` objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            kernel state of a pair of input :class:`~.State` objects

        """
        raise NotImplementedError


class QuadraticKernel(Kernel):
    r"""Quadratic Kernel type

    This kernel returns the quadratic kernel state from a pair of
    :class:`~.KernelParticleState` objects.

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
        state1 : :class:`~.KernelParticleState`
        state2 : :class:`~.KernelParticleState`

        Returns
        -------
        StateVectors
            Transformed state in kernel space.
        """
        if state2 is None:
            state2 = state1
        return (state1.state_vector.T @ state2.state_vector /
                self.ialpha + self.c) ** 2


class QuarticKernel(Kernel):
    r"""Quartic Kernel

    This kernel returns the quartic kernel state from a pair of
    :class:`~.KernelParticleState` objects.

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
        state1 : :class:`~.KernelParticleState`
        state2 : :class:`~.KernelParticleState`

        Returns
        -------
        StateVectors
            Transformed state in kernel space.
        """
        if state2 is None:
            state2 = state1
        return (state1.state_vector.T @ state2.state_vector /
                self.ialpha + self.c) ** 4


class GaussianKernel(Kernel):
    r"""Gaussian Kernel

    This kernel returns the Gaussian kernel state from a pair of
    :class:`~.KernelParticleState` objects.

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, state1, state2=None):
        r"""Calculate the Gaussian Kernel transformation for a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.KernelParticleState`
        state2 : :class:`~.KernelParticleState`

        Returns
        -------
        StateVectors
            Transformed state in kernel space.
        """
        if state2 is None:
            state2 = state1
        diff_tilde_x = (state1.state_vector.T[:, :, None] - state2.state_vector.T[:, None, :]) ** 2
        diff_tilde_x_sum = np.sum(diff_tilde_x, axis=0)

        k_tilde_x = np.exp(-diff_tilde_x_sum / (2 * self.variance)) / (
            np.sqrt(2 * np.pi * self.variance))

        return StateVectors(k_tilde_x)
