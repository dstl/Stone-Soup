from stonesoup.base import Property
from stonesoup.models.transition.linear import (
    LinearGaussianTransitionModel,
    TimeVariantModel,
)
import numpy as np
from scipy.linalg import solve
from scipy.stats import multivariate_normal
from functools import lru_cache
from stonesoup.types.state import State
from typing import Union
from stonesoup.types.array import CovarianceMatrix


class MarkovianGP(LinearGaussianTransitionModel, TimeVariantModel):
    r"""Discrete model implementing a zero-mean Gaussian process (GP).

    We apply the Markovian approximation :math:`P(x_t \mid x_{1:t-1}) \approx P(x_t \mid \mathbf{x}_{t-1})`,
    defining the GP over a sliding window of states, defined by the state vector [].
    
    Denoting the time vector containing the timestamps corresponding to those states as [], 
    the GP over those states forms a multivariate Gaussian distribution:

        .. math::
            p(\mathbf{x}_{t:t-L+1}) \sim \mathcal{N}(\mathbf{0}, \mathbf{K})

    For prediction, the model computes the conditional Gaussian distribution
    for the current state :math:`x_t` given the previous states
    :math:`\mathbf{x}_{t-1:t-L+1}` with time vector :math:`\mathbf{t}_{L-1}`:

        .. math::
            p(x_t | \mathbf{x}_{t-1:t-L+1}) \sim \mathcal{N}(\mu_t, \sigma^2_t)

    where:

        .. math::
            \mu_t = \mathbf{k}_{t, \mathbf{t}_{L-1}}
                    \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1}
                    \mathbf{x}_{\mathbf{t}_{L-1}}

            \sigma^2_t = k_{t,t} - \mathbf{k}_{t, \mathbf{t}_{L-1}}
                        \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1}
                        \mathbf{k}_{t, \mathbf{t}_{L-1}}^\top

    The state transition equation is:

        .. math::
            \mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + w_t,
            \quad w_t \sim \mathcal{N}(0, \mathbf{Q}),

    where:

        .. math::
            \mathbf{x}_t & = & \begin{bmatrix}
                        x_t \\
                        x_{t-1} \\
                        \vdots \\
                        x_{t-L+1}
                    \end{bmatrix}

        .. math::
            \mathbf{F} =
            \begin{bmatrix}
            \mathbf{k}_{t, \mathbf{t}_{L-1}}^\top
            \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1} \\
            \mathbf{I}_{L-1} \quad \mathbf{0}_{L-1, 1}
            \end{bmatrix}

        .. math::
            \mathbf{Q} =
            \begin{bmatrix}
            \sigma_t^2 & \mathbf{0}_{1, L-1} \\
            \mathbf{0}_{L-1, 1} & \mathbf{0}_{L-1, L-1}
            \end{bmatrix}

    The model assumes a constant time interval between observations. To
    construct the covariance matrices, a time vector is created based on the
    current prediction timestep (:attr:`pred_time`) and the specified
    :attr:`time_interval`. The time vector spans backward over the sliding
    window with a total length of :attr:`window_size`.

    Pad the state vector with zeros if the prediction time is smaller than time_interval * window_size

    By default, the GP has the Squared Exponential (SE) covariance function (kernel).
    The :py:meth:`kernel` can be overridden to implement different kernels.

    Specify hyperparameters for the SE kernel through :py:attr:`kernel_params`.
    For SE-based models, the hyperparameters are length_scale and kernel_variance.

    """

    window_size: int = Property(doc="Size of the state vector :math:`d`")
    jitter: float = Property(
        doc="Small diagonal regularisation term added to covariance matrix for numerical stability",
        default=1e-6,
    )
    kernel_params: dict = Property(
        doc=(
            "Dictionary containing keyword arguments for the kernel function.\n\n"
            "By default, the model uses a squared exponential (SE) kernel:\n"
            r"    k(t_1, t_2) = \sigma^2 \exp\left(-\frac{1}{2}\left(\frac{t_1 - t_2}{\ell}\right)^2\right)\n\n"
            "Required keys:\n"
            "    - 'length_scale': float or array-like of shape (num_dims,), the kernel length-scale(s)\n"
            "    - 'kernel_variance': float or array-like of shape (num_dims,), the variance term(s)\n\n"
            "If array-like, parameters are applied per spatial dimension.\n"
            "Custom kernels can be implemented by overriding `kernel()`."
        )
    )

    @property
    def requires_track(self):
        return True

    @property
    def ndim_state(self):
        return self.window_size

    def kernel(self, t1, t2, **kwargs) -> np.ndarray:
        """SE Covariance function (kernel) of the Gaussian Process.

        Computes the covariance matrix between two time vectors using a
        kernel function.
        Override this method to implement different kernels.

        Parameters
        ----------
        t1 : array-like, shape (n_samples_1,)

        t2 : array-like, shape (n_samples_2,)

        Returns
        -------
        np.ndarray, shape (n_samples_1, n_samples_2)
            The covariance matrix between the input arrays `t1` and `t2`.
            Each entry (i, j) represents the covariance between `t1[i]` and `t2[j]`.
        """
        l = self.kernel_params["length_scale"]
        var = self.kernel_params["kernel_variance"]
        t1 = t1.reshape(-1, 1)
        t2 = t2.reshape(1, -1)
        return var * np.exp(-0.5 * ((t1 - t2) / l) ** 2)

    def matrix(self, track, time_interval, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """
        d = self.window_size
        t = self._get_time_vector(track, time_interval)

        gp_weights, _ = self._gp_pred_wrapper(t)
        Fmat = np.eye(d, k=-1)
        Fmat[0, : len(gp_weights)] = gp_weights.T

        return Fmat

    def covar(self, track, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        track : :class:`~.Track`
            The track containing the states to obtain the time vector
        time_interval : :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        d = self.ndim_state
        t = self._get_time_vector(track, time_interval)

        _, noise_var = self._gp_pred_wrapper(t)
        covar = np.zeros((d, d))
        covar[0, 0] = 1
        return CovarianceMatrix(covar * noise_var)

    def _get_time_vector(self, track, time_interval):
        return self._get_time_vector_markov1(track, time_interval)

    def _get_time_vector_markov1(self, track, time_interval):
        """
        Generates a time vector containing the time elapsed (in seconds)
        from the start time of the track.

        The time vector includes:
            - The prediction time: (last state timestamp + time_interval)
            - The timestamps of the last `window_size` states in the track.

        Parameters:
            track: The track containing the states.
            time_interval: The time interval for prediction.

        Returns:
            time_vector (ndarray): A 2D array of elapsed times (d+1 x 1).

        Markov approx 2 assumes a constatn time interval.
        Window spans absolute time time-interval*window_size
        """
        d = min(self.window_size, len(track.states))
        start_time = track.states[0].timestamp

        prediction_time = track.states[-1].timestamp + time_interval
        time_vector = np.array([(prediction_time - start_time).total_seconds()])
        for i in range(0, d):
            state_time = track.states[-1 - i].timestamp
            time_vector = np.append(
                time_vector, (state_time - start_time).total_seconds()
            )
        return time_vector.reshape(-1, 1)

    def _gp_pred_wrapper(self, t):
        """Prepare inputs and call cached GP prediction helper function."""
        t = tuple(np.round(np.atleast_1d(t).flatten(), 10))
        return self._gp_pred(t)

    @lru_cache
    def _gp_pred(self, t1):
        """Cached GP prediction: compute weights and noise variance."""
        t1 = np.array(t1)
        C = self.kernel(t1, t1)
        prior_var = C[0, 0]
        C = C + np.eye(np.shape(C)[0]) * self.jitter
        gp_weights = solve(C[1:, 1:], C[1:, 0])
        noise_var = prior_var - C[0, 1:] @ gp_weights
        return gp_weights, noise_var

    def rvs(self, num_samples: int = 1, random_state=None, **kwargs):
        """Model noise/sample generation function.

        Generates noise samples from the model. For Markovian GP models,
        only the first dimension (new prediction) has noise; the remaining
        dimensions (history states) are deterministic and have zero noise.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to be generated (default is 1)
        random_state : numpy.random.RandomState, optional
            Random state for reproducible sampling

        Returns
        -------
        noise : StateVector or StateVectors
            Noise samples from the model's distribution
        """
        from ...types.array import StateVector, StateVectors

        track = kwargs.get("track")
        time_interval = kwargs.get("time_interval")
        t = self._get_time_vector(track, time_interval)
        _, noise_var = self._gp_pred_wrapper(t)

        random_state = random_state if random_state is not None else self.random_state

        # Only the first dimension has noise; rest are deterministic (zero variance)
        noise = np.zeros((self.ndim_state, num_samples))
        if random_state is not None:
            noise[0, :] = random_state.normal(0, np.sqrt(noise_var), num_samples)
        else:
            noise[0, :] = np.random.normal(0, np.sqrt(noise_var), num_samples)

        if num_samples == 1:
            return noise.reshape(-1, 1).view(StateVector)
        return noise.view(StateVectors)

    def logpdf(
        self, state1: State, state2: State, **kwargs
    ) -> Union[float, np.ndarray]:
        r"""Evaluate the log-likelihood under a Gaussian model.

        This computes the log probability density of ``state1`` given ``state2``
        under a Gaussian distribution with covariance :attr:`covar`.

        Unlike the default SciPy behaviour, this implementation sets
        ``allow_singular=True``. This is necessary because some models
        (e.g. Gaussian Process–based models) naturally produce singular
        or near-singular covariance matrices. In such cases, the log-likelihood
        is still well-defined using a pseudo-inverse and pseudo-determinant.

        Parameters
        ----------
        state1 : State
        state2 : State

        Returns
        -------
        float or np.ndarray
            The log likelihood of ``state1`` given ``state2``.
        """
        covar = self.covar(**kwargs)

        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")

        diff = (state1.state_vector - self.function(state2, **kwargs)).T

        likelihood = np.atleast_1d(
            multivariate_normal.logpdf(diff, cov=covar, allow_singular=True)
        )

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood
