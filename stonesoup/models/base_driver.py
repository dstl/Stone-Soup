from abc import abstractmethod
from enum import Enum
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np

from stonesoup.base import Base, Property
from stonesoup.types.array import (
    CovarianceMatrices,
    CovarianceMatrix,
    StateVector,
    StateVectors,
)


class NoiseCase(Enum):
    """Different methods of approximating the residuals for
    the truncated series representation of the associated
    Levy integrals
    """

    TRUNCATED = 0
    GAUSSIAN_APPROX = 1
    PARTIAL_GAUSSIAN_APPROX = 2


class Driver(Base):
    """Base class for all driver classes use to drive transition models."""

    pass


class LevyDriver(Driver):
    """Driver type

    Base/Abstract class for all stochastic Levy noise driving processes
    used to drive :class:`~.LevyModel` instances.
    """

    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_state = np.random.default_rng(self.seed)

    @abstractmethod
    def characteristic_func():
        """Characteristic function for associated Levy distribution"""


class ConditionallyGaussianDriver(LevyDriver):
    """Conditional Gaussian Levy noise driver.

    Noise samples are generated according to the Levy State-Space Model, which are Gaussian
    when conditioned on latent variables in the form of jump times and sizes. The
    latent variables may be non-Gaussian.
    """

    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")
    mu_W: float = Property(default=0.0, doc="Default conditional Gaussian mean")
    sigma_W2: float = Property(default=1.0, doc="Default conditional Gaussian variance")
    noise_case: NoiseCase = Property(
        default=NoiseCase.GAUSSIAN_APPROX,
        doc="Cases for compensating residuals from series truncation",
    )

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        """Thinning probabilities for accept-reject sampling of latent variables."""
        # Accept all
        return np.ones_like(jsizes)  # (n_jumps, n_samples)

    def _accept_reject(
        self, jsizes: np.ndarray, random_state: Optional[Generator]
    ) -> np.ndarray:
        """Accept reject sampling to thin out sampled latents (jumps).

        Args:
            jsizes (np.ndarray): Jump sizes to apply accept-reject sampling
            random_state (Generator, optional): Random state to use. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        probabilities = self._thinning_probabilities(jsizes)
        u = random_state.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u <= probabilities, jsizes, 0)
        return jsizes

    def sample_latents(
        self, dt: float, num_samples: int, random_state: Optional[Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the non-linear and possible non-Gaussian latent variables.

        Args:
            dt (float): _description_
            num_samples (int): Number of different jump sequences to sample. Each jump sequence
                               consist of a multiple jumps where the number of jumps depends
                               on the truncation parameter `self.c`
            random_state (Optional[Generator], optional): Random state to use. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A Tuple consisting of the jump sizes and jump times.
        """
        if random_state is None:
            random_state = self.random_state
        # Sample latents pairs
        # num_samples = 1 # TODO: constraned num_samples to 1 always.
        epochs = random_state.exponential(
            scale=1 / dt, size=(int(self.c * dt), num_samples)
        )
        epochs = epochs.cumsum(axis=0)
        # Accept reject sampling
        jsizes = self._hfunc(epochs=epochs)
        jsizes = self._accept_reject(jsizes=jsizes, random_state=random_state)
        # Generate jump times
        jtimes = random_state.uniform(low=0.0, high=dt, size=jsizes.shape)
        return jsizes, jtimes

    @abstractmethod
    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        """H function to be used an direct or indirect evaluation of the inverse upper tail
        probability of the Levy density. For indirect approaches, accept reject sampling is
        as an additional step is needed.
        """

    @abstractmethod
    def _centering(self, e_ft: np.ndarray, truncation: float) -> StateVector:
        """Compensation term for skewed Levy density.

        Args:
            e_ft (np.ndarray): Expectation of Levy stochastic integral over a unit time axis.
            truncation (float): Truncation parameter or no. expected
                                Possion jumps per unit time.

        Returns:
            StateVector: Vectorised form of compensation term.
        """

    @abstractmethod
    def _jump_power(self, jszies: np.ndarray) -> np.ndarray:
        """Raises the latent jump sizes to the desired power .

        Args:
            jszies (np.ndarray): Latent jump sizes to raise.

        Returns:
            np.ndarray: Latent jump sizes raised to the desired power.
        """

    @abstractmethod
    def _first_moment(self, truncation: float) -> float:
        """Computes first moment of the underlying subordinator process up to
        an upper limit defined by h(c), whereby h is the H function and c represents
        the truncation parameter.

        Args:
            truncation (float): Truncation parameter which defines the upper limit
                                of the associated integral.


        Returns:
            float: First moment of subordinator process up to limit h(c).
        """

    @abstractmethod
    def _second_moment(self, truncation: float) -> float:
        """Computes second moment of the underlying subordinator process up to
        an upper limit defined by h(c), whereby h is the H function and c represents
        the truncation parameter.

        Args:
            truncation (float): Truncation parameter which defines the upper limit
                                of the associated integral.


        Returns:
            float: Second moment of subordinator process up to limit h(c).
        """

    @abstractmethod
    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        """Calculates the covariance of Gaussian approximate residuals. Residuals
        arises from the truncated series representation of the Levy stotchastic
        integral.

        Args:
            e_ft (np.ndarray): Levy stochastic integral over a unit time axia.
            truncation (float): Truncation parameter.
            mu_W (float): Gaussian mean of Levy density when conditioned over the
                          latent variables (jumps).
            sigma_W2 (float): Gaussian variance of Levy density when conditioned
                              over the latent variables (jumps).

        Returns:
            CovarianceMatrix: Covariance matrix of the Gaussian approximated
                              residuals.
        """

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
        """Calculates the mean of Gaussian approximate residuals. Residuals
        arises from the truncated series representation of the Levy stotchastic
        integral.

        Args:
            e_ft (np.ndarray): Levy stochastic integral over a unit time axia.
            truncation (float): Truncation parameter.
            mu_W (float): Gaussian mean of Levy density when conditioned over the
                          latent variables (jumps).

        Returns:
            StateVector: Mean vector of the Gaussian approximated residuals.
        """
        if self.noise_case == NoiseCase.TRUNCATED:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
        elif (
            self.noise_case == NoiseCase.GAUSSIAN_APPROX
            or self.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX
        ):
            r_mean = e_ft * mu_W  # (m, 1)
        else:
            raise AttributeError("invalid noise case")
        return self._first_moment(truncation=truncation) * r_mean  # (m, 1)

    def mean(
        self,
        jsizes: np.array,
        jtimes: np.array,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Computes mean vectors. The number of mean vectors is dependent on the
        number of samples in the jump sizes/times. Each jump sequence results in
        an unique mean vector.

        Args:
            jsizes (np.array): Latents corresponding to jump sizes.
            jtimes (np.array): Latents corresponding to jump times.
            ft_func (Callable[..., np.ndarray]): The function f consisting of the
                state transtion matrix multiplied by the control matrix h as denoted
                by Godstill et. al. (2020).
            e_ft_func (Callable[..., np.ndarray]): The expectation of ft_func.
            dt (float): The time interval.
            mu_W (Optional[float], optional): The conditionally Gaussian mean vector.
                Defaults to None and the default mu_W specified during initialisation
                is used.

        Returns:
            Union[StateVector, StateVectors]: The resulting mean vectors.
        """
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        assert jsizes.shape == jtimes.shape
        num_samples = jsizes.shape[1]
        truncation = self.c * dt
        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        series = np.sum(jsizes[..., None, None] * ft, axis=0)  # (n_samples, m, 1)
        m = series * mu_W

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_mean = self._residual_mean(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[
            None, ...
        ]
        centering = (
            dt * self._centering(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[None, ...]
        )
        mean = m - centering + residual_mean
        if num_samples == 1:
            return mean[0].view(StateVector)
        else:
            return mean.view(StateVectors)

    def covar(
        self,
        jsizes: np.array,
        jtimes: np.array,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        sigma_W2: Optional[float] = None,
        **kwargs
    ) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Computes covariance matrices. The number of covariance matrices is dependent
        on the number of samples in the jump sizes/times. Each jump sequence results
        in an unique covariance matrix.

        Args:
            jsizes (np.array): Latents corresponding to jump sizes.
            jtimes (np.array): Latents corresponding to jump times.
            ft_func (Callable[..., np.ndarray]): The function f consisting of the
                state transtion matrix multiplied by the control matrix h as denoted
                by Godstill et. al. (2020).
            e_ft_func (Callable[..., np.ndarray]): The expectation of ft_func.
            dt (float): The time interval.
            mu_W (Optional[float], optional): The conditionally Gaussian mean.
                Defaults to None and the default mu_W specified during initialisation
                is used.
            sigma_W2 (Optional[float], optional): The conditionally Gaussian variance.
                Defaults to None and the default sigma_W2 specified during initialisation
                is used.

        Returns:
            Union[CovarianceMatrix, CovarianceMatrices]: The resulting covariance matrices.
        """
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        sigma_W2 = (
            np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)
        )

        assert jsizes.shape == jtimes.shape
        num_samples = jsizes.shape[1]
        jsizes = self._jump_power(jsizes)  # (n_jumps, n_samples)
        truncation = self._hfunc(self.c * dt)

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        ft2 = np.einsum("ijkl, ijml -> ijkm", ft, ft)  # (n_jumps, n_samples, m, m)
        series = np.sum(jsizes[..., None, None] * ft2, axis=0)  # (n_samples, m, m)
        s = sigma_W2 * series

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_cov = self._residual_covar(
            e_ft=e_ft, mu_W=mu_W, sigma_W2=sigma_W2, truncation=truncation
        )
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)  # (m, m)
        else:
            return covar.view(CovarianceMatrices)  # (n_samples, m, m)

    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrices,
        random_state: Optional[np.random.RandomState] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Computes the driving noise term given the mean and covariance matrix specified.


        Args:
            mean (StateVector): The mean vector.
            covar (CovarianceMatrices): The covariance matrix.
            random_state (Optional[np.random.RandomState], optional): RNG to use. Defaults to None.
            num_samples (int, optional): Number of driving noise samples. Defaults to 1.

        Returns:
            Union[StateVector, StateVectors]: Driving noise samples.
        """
        assert isinstance(mean, StateVector)
        assert isinstance(covar, CovarianceMatrix)
        if random_state is None:
            random_state = self.random_state
        noise = random_state.multivariate_normal(mean.flatten(), covar, size=num_samples)
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)


class NormalSigmaMeanDriver(ConditionallyGaussianDriver):
    """Implements the class of Normal Sigma Mean (NSM) Levy models."""

    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes**2

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if self.noise_case == NoiseCase.TRUNCATED:
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif self.noise_case == NoiseCase.GAUSSIAN_APPROX:
            r_cov = (
                e_ft
                @ e_ft.T
                * self._second_moment(truncation=truncation)
                * (mu_W**2 + sigma_W2)
            )
        elif self.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX:
            r_cov = e_ft @ e_ft.T * self._second_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)


class NormalVarianceMeanDriver(ConditionallyGaussianDriver):
    """Implements the Normal Variance Mean (NVM) Levy models."""

    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes

    def _centering(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> StateVector:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if self.noise_case == NoiseCase.TRUNCATED:
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif self.noise_case == NoiseCase.GAUSSIAN_APPROX:
            r_cov = (
                e_ft
                @ e_ft.T
                * (
                    self._second_moment(truncation=truncation) * mu_W**2
                    + self._first_moment(truncation=truncation) * sigma_W2
                )
            )
        elif self.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX:
            r_cov = e_ft @ e_ft.T * self._first_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)
