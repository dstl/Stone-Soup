from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from .base import Base, Property
from .types.array import StateVectors
from .types.state import State


class Kernel(Base):
    """Kernel base type

    A Kernel provides a means to translate state space or measurement space into kernel space.
    """

    @abstractmethod
    def __call__(self, state1, state2=None, **kwargs):
        r"""
        Compute the kernel state of a pair of :class:`~.State` objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        StateVectors
            kernel state of a pair of input :class:`~.State` objects

        """
        raise NotImplementedError

    def update_parameters(self, kwargs):
        for parameter in type(self).properties:
            if parameter in kwargs.keys():
                setattr(self, parameter, kwargs[parameter])

    @property
    def parameters(self):
        return {parameter: getattr(self, parameter) for parameter in type(self).properties}

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


class AdditiveKernel(Kernel):
    """Additive kernel

    Elementwise addition of corresponding kernel state values. Similar to an OR operation.
    """

    kernel_list: Sequence[Kernel] = Property(doc="List of kernels")

    def __call__(self, state1, state2=None, **kwargs):
        return np.sum([kernel(state1, state2, **kwargs) for kernel in self.kernel_list], axis=0)


class MultiplicativeKernel(Kernel):
    """Multiplicative kernel

    Elementwise multiplication of corresponding kernel state values. Similar to an AND operation.
    """

    kernel_list: Sequence[Kernel] = Property(doc="List of kernels")

    def __call__(self, state1, state2=None, **kwargs):
        return np.prod([kernel(state1, state2, **kwargs) for kernel in self.kernel_list], axis=0)


class PolynomialKernel(Kernel):
    r"""Polynomial Kernel

    This kernel returns the polynomial kernel of order :math:`p` state from a pair of
    :class:`.State` objects.

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

    def __call__(self, state1, state2=None, **kwargs):
        self.update_parameters(kwargs)
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)
        return (state_vector1.T @ state_vector2 / self.ialpha + self.c) ** self.power


class LinearKernel(PolynomialKernel):
    r"""Linear Kernel

   This kernel returns the linear kernel state vector from a pair of :class:`~.State`
   objects.

   The linear kernel of state vectors :math:`\mathbf{x}` and :math:`\mathbf{x}^\prime` is
   defined as:

   .. math::
        \mathtt{k}\left(\mathbf{x}, \mathbf{x}^\prime\right) =
        \mathbf{x}^T\mathbf{x}^\prime

   The linear kernel can capture the first-order moments of a distribution, such as the mean
   and covariance.
   """

    @property
    def power(self):
        r"""The linear polynomial power, :math:`p=1`"""
        return 1

    @property
    def c(self):
        return 0

    @property
    def ialpha(self):
        return 1


class QuadraticKernel(PolynomialKernel):
    r"""Quadratic Kernel type

    This kernel returns the quadratic kernel state vector from a pair of :class:`~.State`
    objects.

    The quadratic kernel of state vectors :math:`\mathbf{x}` and :math:`\mathbf{x}^\prime` is
    defined as:

    .. math::
         \mathtt{k}\left(\mathbf{x}, \mathbf{x}^\prime\right) =
         \left(\alpha \langle \mathbf{x}, \mathbf{x}^\prime \rangle + c\right)^2

    The quadratic kernel can capture the second-order moments of a distribution, such as the
    covariance and correlations between pairs of variables.
    The quadratic kernel is appropriate when the data is nonlinear but still relatively simple.
    """
    @property
    def power(self):
        r"""The quadratic polynomial power, :math:`p=2`"""
        return 2


class QuarticKernel(PolynomialKernel):
    r"""Quartic Kernel

    This kernel returns the quartic kernel state from a pair of :class:`~.State` objects.

    The quartic kernel of state vectors :math:`\mathbf{x}` and :math:`\mathbf{x}^\prime` is defined
    as:

    .. math::
         \mathtt{k}(\mathbf{x}, \mathbf{x}^\prime) =
         \left(\alpha \langle \mathbf{x}, \mathbf{x}^\prime \rangle + c\right)^4

    The quartic kernel can capture higher-order moments beyond the mean and covariance, such as
    skewness and kurtosis.
    THe quartic kernel can be used when the data is highly nonlinear and complex.
    """
    @property
    def power(self):
        r"""The quartic polynomial power, :math:`p=4`"""
        return 4


class GaussianKernel(Kernel):
    r"""Gaussian Kernel

    This kernel returns the Gaussian kernel state vector from a pair of
    :class:`~.State` objects.

    The Gaussian kernel of state vectors :math:`\mathbf{x}` and
    :math:`\mathbf{x}^\prime` is defined as:

    .. math::
         \mathtt{k}(\mathbf{x}, \mathbf{x}^\prime) =
         \mathrm{exp}\left(-\frac{||\mathbf{x} - \mathbf{x}^\prime||^{2}}{2\pi\sigma^2}\right)
    """
    variance: float = Property(
        default=1e1,
        doc=r"Denoted as :math:`\sigma^2` in the equation above. Determines the width of the "
            r"Gaussian kernel. Range is [1e0, 1e2].")

    def __call__(self, state1, state2=None, **kwargs):
        r"""Calculate the Gaussian Kernel transformation for a pair of :class:`~.State` objects.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        StateVectors
            Transformed state vector in kernel space.
        """
        self.update_parameters(kwargs)
        state_vector1, state_vector2 = self._get_state_vectors(state1, state2)

        diff_tilde_x = (state_vector1[:, :, None] - state_vector2[:, None, :]) ** 2
        diff_tilde_x_sum = np.sum(diff_tilde_x, axis=0)

        k_tilde_x = np.exp(-diff_tilde_x_sum/(2*self.variance)) / np.sqrt(2*np.pi*self.variance)

        return StateVectors(k_tilde_x)


class _StateKernel(Kernel):
    kernel: Kernel = Property(doc="Base Kernel class")
    mapping: list = Property(default=None, doc="List of mappings of the components to be used in "
                                               "the kernel from the state vector.")

    @property
    def parameters(self):
        return self.kernel.parameters

    def update_parameters(self, kwargs):
        return self.kernel.update_parameters(kwargs)

    def _get_states(self, state1, state2):
        if state2 is None:
            state2 = state1
        if self.mapping is None:
            self.mapping = list(range(state1[0].state_vector.shape[0]))
        state1 = np.hstack([state.state_vector[self.mapping] for state in state1])
        state2 = np.hstack([state.state_vector[self.mapping] for state in state2])
        return state1, state2

    def __call__(self, state1, state2=None, **kwargs):
        r"""
        Compute the kernel state of a pair of objects containing States

        Parameters
        ----------
        state1 : :class:`~.Track`
        state2 : :class:`~.Track`

        Returns
        -------
        StateVectors
            kernel state of a pair of input :class:`~.State` objects
        """
        state1, state2 = self._get_states(state1, state2)
        return self.kernel.__call__(state1, state2, **kwargs)


class TrackKernel(_StateKernel):
    def __call__(self, state1, state2=None, **kwargs):
        r"""
        Compute the kernel state of a pair of :class:`~.Track` objects

        Parameters
        ----------
        state1 : :class:`~.Track`
        state2 : :class:`~.Track`

        Returns
        -------
        StateVectors
            kernel state of a pair of input :class:`~.State` objects
        """
        state1, state2 = self._get_states(state1, state2)
        return self.kernel.__call__(state1, state2, **kwargs)


class MeasurementKernel(_StateKernel):
    def __call__(self, state1, state2=None, **kwargs):
        r"""
        Compute the kernel state of a pair of lists of measurements as :class:`~.List` objects

        Parameters
        ----------
        state1 : :class:`~.List`
        state2 : :class:`~.List`

        Returns
        -------
        StateVectors
            kernel state of a pair of input :class:`~.State` objects
        """
        state1, state2 = self._get_states(state1, state2)
        return self.kernel.__call__(state1, state2, **kwargs)
