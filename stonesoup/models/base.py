from abc import abstractmethod
from typing import TYPE_CHECKING, Union, Optional
from datetime import timedelta
import numpy as np
from scipy.stats import multivariate_normal

from .base_driver import LevyDriver, Latents
from ..base import Base, Property
from ..functions import jacobian as compute_jac
from ..types.array import StateVector, StateVectors, CovarianceMatrix, CovarianceMatrices
from ..types.numeric import Probability
from ..types.state import State
from scipy.integrate import quad_vec
from scipy.linalg import expm

if TYPE_CHECKING:
    from ..types.detection import Detection


class Model(Base):
    """Model type

    Base/Abstract class for all models."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of model"""
        raise NotImplementedError

    @abstractmethod
    def function(self, state: State, noise: Union[bool, np.ndarray] = False,
                 **kwargs) -> Union[StateVector, StateVectors]:
        """Model function :math:`f_k(x(k),w(k))`

        Parameters
        ----------
        state: State
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is used)

        Returns
        -------
        : :class:`StateVector` or :class:`StateVectors`
            The StateVector(s) with the model function evaluated.
        """
        raise NotImplementedError

    def jacobian(self, state, **kwargs):
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state : :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """

        return compute_jac(self.function, state, **kwargs)

    @abstractmethod
    def rvs(self, num_samples: int = 1, **kwargs) -> Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model.


        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        : :class:`~.Probability` or :class:`~.numpy.ndarray` of :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """
        raise NotImplementedError

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        :  float or :class:`~.numpy.ndarray`
            The log likelihood of ``state1``, given ``state2``
        """
        return np.log(self.pdf(state1, state2, **kwargs))


class LinearModel(Model):
    """LinearModel class

    Base/Abstract class for all linear models"""

    @abstractmethod
    def matrix(self, **kwargs) -> np.ndarray:
        """Model matrix"""
        raise NotImplementedError

    def function(self, state: State, noise: Union[bool, np.ndarray] = False,
                 **kwargs) -> Union[StateVector, StateVectors]:
        """Model linear function :math:`f_k(x(k),w(k)) = F_k(x_k) + w_k`

        Parameters
        ----------
        state: State
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        : :class:`StateVector` or :class:`StateVectors`
            The StateVector(s) with the model function evaluated.
        """
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs) @ state.state_vector + noise

    def jacobian(self, state: State, **kwargs) -> np.ndarray:
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state : :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """
        return self.matrix(**kwargs)


class ReversibleModel(Model):
    """Non-linear model containing sufficient co-ordinate
    information such that the linear co-ordinate conversions
    can be calculated from the non-linear counterparts.

    Contains an inverse function which computes the reverse
    of the relevant linear-to-non-linear function"""

    @abstractmethod
    def inverse_function(self, detection: 'Detection', **kwargs) -> StateVector:
        """Takes in the result of the function and
        computes the inverse function, returning the initial
        input of the function.

        Parameters
        ----------
        detection: :class:`~.Detection`
            Input state (non-linear format)

        Returns
        -------
        StateVector
            The linear co-ordinates
        """
        raise NotImplementedError


class TimeVariantModel(Model):
    """TimeVariantModel class

    Base/Abstract class for all time-variant models"""


class TimeInvariantModel(Model):
    """TimeInvariantModel class

    Base/Abstract class for all time-invariant models"""


class GaussianModel(Model):
    """GaussianModel class

    Base/Abstract class for all Gaussian models"""
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(self.seed) if self.seed is not None else None

    def rvs(self, num_samples: int = 1, random_state=None, **kwargs) ->\
            Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal{N}(0,Q)

        where :math:`v_t =` ``noise`` and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")

        random_state = random_state if random_state is not None else self.random_state

        noise = multivariate_normal.rvs(
            np.zeros(self.ndim), covar, num_samples, random_state=random_state)

        noise = np.atleast_2d(noise)

        if self.ndim > 1:
            noise = noise.T  # numpy.rvs method squeezes 1-dimensional matrices to integers

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        : :class:`~.Probability` or :class:`~.numpy.ndarray` of :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        :  float or :class:`~.numpy.ndarray`
            The log likelihood of ``state1``, given ``state2``
        """
        covar = self.covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")

        # Calculate difference before to handle custom types (mean defaults to zero)
        # This is required as log pdf coverts arrays to floats
        likelihood = np.atleast_1d(
            multivariate_normal.logpdf((state1.state_vector - self.function(state2, **kwargs)).T,
                                       cov=covar))

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    @abstractmethod
    def covar(self, **kwargs) -> CovarianceMatrix:
        """Model covariance"""


class LevyModel(Model):
    """
    Class to be derived from for Levy models.
    For now, we consider only conditionally Gaussian ones
    """
    driver: LevyDriver = Property(doc="Levy process noise driver")
    mu_W: Optional[float] = Property(default=None, doc="Condtional Gaussian mean")
    sigma_W2: Optional[float] = Property(default=None, doc="Conditional Gaussian variance")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def _integrand(self, dt: float, jtimes: np.ndarray) -> np.ndarray:
        pass

    def _integrate(self, func: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res, err = quad_vec(func, a=a, b=b)
        return res
    
    def mean(
        self, latents: Latents, time_interval: timedelta, **kwargs
    ) -> StateVector | StateVectors:
        """Model mean"""
        dt = time_interval.total_seconds()
        integrand_f = self._integrand
        func = lambda dt: integrand_f(dt, jtimes=np.zeros((1, 1)))[0, 0, :] # currying
        integral_f = lambda dt: self._integrate(func, a=0, b=dt)
        return self.driver.mean(
            latents=latents,
            dt=dt,
            e_ft_func=integral_f,
            ft_func=integrand_f,
            mu_W=self.mu_W,
        )

    def covar(self, latents: Latents, time_interval: timedelta, **kwargs) -> CovarianceMatrix | CovarianceMatrices:
        """Model covariance"""
        dt = time_interval.total_seconds()
        integrand_f = self._integrand
        func = lambda dt: integrand_f(dt, jtimes=np.zeros((1, 1)))[0, 0, :]
        integral_f = lambda dt: self._integrate(func, a=0, b=dt)
        return self.driver.covar(
            latents=latents,
            dt=dt,
            e_ft_func=integral_f,
            ft_func=integrand_f,
            mu_W=self.mu_W,
            sigma_W2=self.sigma_W2,
        )

    def sample_latents(self, time_interval: timedelta, num_samples: int, random_state: Optional[np.random.RandomState]=None) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        jsizes, jtimes = self.driver.sample_latents(dt=dt, num_samples=num_samples, random_state=random_state)
        latents.add(driver=self.driver, jsizes=jsizes, jtimes=jtimes)
        return latents
    
    def rvs(
        self,
        latents: Optional[Latents] = None,
        num_samples: int = 1,
        random_state: Optional[np.random.RandomState] = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        noise = 0
        if not latents:
            latents = self.sample_latents(num_samples=1, random_state=random_state, **kwargs)

        mean = self.mean(latents=latents, **kwargs)
        if mean is None or None in mean:
            raise ValueError("Cannot generate rvs from None-type mean")
        assert isinstance(mean, StateVector)

        covar = self.covar(latents=latents, **kwargs)
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")
        assert isinstance(covar, CovarianceMatrix)
        
        noise += self.driver.rvs(
            mean=mean, covar=covar, random_state=random_state, num_samples=num_samples, **kwargs
        )
        return noise

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        return NotImplementedError

    # def pdf(self, state1: State, state2: State, **kwargs) -> Probability | np.ndarray:
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from scipy.stats import norm

    #     # Parameters
    #     x_min = -10.0
    #     x_max = 10.0
    #     N = 2**8

    #     # Generate k and w
    #     k = np.arange(N)
    #     w = (0.5 - N / 2 + k) * (2 * np.pi / (x_max - x_min))

    #     # Characteristic function
    #     cffun = lambda w: np.exp(-0.5 * w**2)

    #     alpha=1
    #     nu=2

    #     cffun = lambda w: ( 1 -  (1j * w) / alpha) ** (-nu)

    #     cf = cffun(w[int(N/2):])
    #     cf = np.concatenate([np.conj(cf[::-1]), cf])

    #     # Compute dx, C, and D
    #     dx = (x_max - x_min) / N

    #     C = (-1+0j) ** ((1 - 1 / N) * (x_min / dx + k)) / (x_max - x_min)

    #     D = (-1+0j) ** (-2 * (x_min / (x_max - x_min)) * k)

    #     # Compute the PDF
    #     pdf = np.real(C * np.fft.fft(D * cf))

    #     # Compute the CDF
    #     cdf = np.cumsum(pdf * dx)

    #     # Generate x values
    #     x = x_min + k * dx

    #     # Plot the PDF
    #     plt.figure()
    #     plt.plot(x, pdf, label='inverse CF')
    #     plt.plot(x, norm.pdf(x), label='scipy')
    #     plt.title('PDF')
    #     plt.grid()
    #     plt.legend()
    #     plt.show()

    #     # Plot the CDF
    #     # plt.figure()
    #     # plt.plot(x, cdf)
    #     # plt.title('CDF')
    #     # plt.grid()
    #     # plt.show()
