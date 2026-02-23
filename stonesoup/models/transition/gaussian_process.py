import numpy as np
from datetime import datetime, timedelta
from scipy.linalg import inv
from typing import Union, List

from stonesoup.base import Property
from stonesoup.models.base import TimeVariantModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.types.track import Track
from stonesoup.types.array import StateVector, StateVectors


class SimpleGaussianProcess(LinearGaussianTransitionModel, TimeVariantModel):
    """ A simple Gaussian Process transition model. """

    num_lags: int = Property(doc='Number of lags in the state (aka dimensionality of state)')
    sigma: float = Property(doc='Process noise')
    start_time: datetime = Property(doc='The start time, necessary for computing dts')

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """
        return self.num_lags

    def function(self, track: Track, noise: Union[bool, np.ndarray] = False,
                 **kwargs) -> Union[StateVector, StateVectors]:
        """Model function :math:`f_k(x(k-l:k),w(k))` where `l` is the :py:attr:`~num_lags`

        Parameters
        ----------
        track: Track
            An input track
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `False`, in which
            case no noise will be added if 'True', the output of :meth:`~.Model.rvs` is used)

        Returns
        -------
        : :class:`StateVector` or :class:`StateVectors`
            The StateVector(s) with the model function evaluated.
        """
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(**kwargs)
            else:
                noise = 0

        return self.matrix(track, **kwargs) @ track.state_vector + noise

    def matrix(self, track: Track, time_interval: timedelta, **kwargs):
        """ Compute the model transition matrix

        Parameters
        ----------
        track: :class:`~.Track`
            An input track
        time_interval: :class:`datetime.timedelta`
            The time interval

        Returns
        -------
        : :class:`numpy.ndarray` of shape (:py:attr:`~num_lags`, :py:attr:`~num_lags`)
        """

        # Get timestamps from states
        timestamps = self._get_timestamps(track, time_interval)

        # Calculate kernel matrix
        k = self._calc_kernel_matrix(timestamps)

        # Make notation analogous to Kalman Filter
        p_xy = np.atleast_2d(k[0, 1:])
        p_yy = np.atleast_2d(k[1:, 1:])
        inv_p_yy = inv(p_yy) if len(p_yy) else p_yy

        # Calculate A
        n = len(timestamps)
        row_1 = np.concatenate((p_xy @ inv_p_yy, np.zeros((1, self.num_lags - n + 1))), axis=1)
        row_2 = np.concatenate((np.eye(self.num_lags - 1), np.zeros((self.num_lags - 1, 1))),
                               axis=1)
        A = np.concatenate((row_1, row_2))
        return A

    def covar(self, track, time_interval, **kwargs):
        """ Compute the model covariance matrix

        Parameters
        ----------
        track: :class:`~.Track`
            The track object
        time_interval: :class:`datetime.timedelta`
            The time interval

        Returns
        -------
        : :class:`numpy.ndarray` of shape (:py:attr:`~num_lags`, :py:attr:`~num_lags`)
        """

        # Get timestamps from states
        timestamps = self._get_timestamps(track, time_interval)

        # Calculate kernel matrix
        k = self._calc_kernel_matrix(timestamps)

        # Make notation analogous to Kalman Filter
        p_xx = np.atleast_2d(k[0, 0])
        p_xy = np.atleast_2d(k[0, 1:])
        p_yy = np.atleast_2d(k[1:, 1:])
        inv_p_yy = inv(p_yy) if len(p_yy) else p_yy

        # Calculate Q
        Q = np.zeros((self.num_lags, self.num_lags))
        Q[0, 0] = p_xx - p_xy @ inv_p_yy @ p_xy.T
        return Q

    def _get_timestamps(self, track, time_interval):
        timestamps = []
        start_index = len(track.states) - self.num_lags
        end_index = len(track.states)
        if start_index < 0:
            start_index = 0
        for i in range(start_index, end_index):
            timestamps.append(track.states[i].timestamp - self.start_time)
        last_timestamp = self.start_time
        if end_index:
            last_timestamp = track.states[-1].timestamp
        timestamps.append(last_timestamp - self.start_time + time_interval)
        timestamps.reverse()
        return timestamps

    def _calc_kernel_matrix(self, timestamps: List[timedelta]):
        """ Computes and returns the kernel matrix with form:

        .. math::
            K = \begin{bmatrix}
            k(t_1,t_2) & \cdots & k(t_1,t_k) \\
            \vdots     & \ddots & \vdots \\
            k(t_k,t_1) & \cdots & k(t_k,t_k) \\
            \end{bmatrix}

        where :math:`K(t,t')` is the chosen kernel function.

        Parameters
        ----------
        timestamps: list of :class:`datetime.timedelta`

        Returns
        -------
        : :class:`numpy.ndarray` of shape (`num_lags`, `num_lags`)
            The computed kernel matrix
        """
        n = len(timestamps)
        k = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k[i, j] = self._kernel(timestamps[i], timestamps[j], self.sigma)
        return k

    def _kernel(self, t1, t2, sigma):
        dt = t2 - t1
        return np.exp(-dt.total_seconds() ** 2 / sigma ** 2)