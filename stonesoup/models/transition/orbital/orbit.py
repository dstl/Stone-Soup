# -*- coding: utf-8 -*-

import numpy as np
from numpy import matlib
from scipy.stats import norm
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
from ....orbital_functions import lagrange_coefficients_from_universal_anomaly

from ....base import Property
from ....types.orbitalstate import OrbitalState, TLEOrbitalState
from ....types.array import CovarianceMatrix
from ...base import LinearModel, NonLinearModel
from .base import OrbitalTransitionModel

from sgp4.api import jday, Satrec


class SimpleMeanMotionTransitionModel(OrbitalTransitionModel, LinearModel):
    r"""This simple transition model uses the mean motion to update the
    mean anomaly and then use that to construct a new orbital state
    vector via the TLE parameterisation. Statistics assume a
    Gauss-distributed mean anomaly with all other quantities fixed.

    The transition proceeds as,

        .. math::

            M_{t_1} = M_{t_0} + n(t_1 - t_0), \; (\mathrm{modulo} \,
            2\pi)


    for the interval :math:`t_0 \rightarrow t_1`, where :math:`n` is
    the mean motion, computed as:

        .. math::

           n = \sqrt{ \frac{\mu}{a^3} }

    which is in units of :math:`\mathrm{rad} \, s^{-1}`. The state
    vector is then recreated from the TLE parameterisation of the
    orbital state vector:

        .. math::

            X = [i, \Omega, e, \omega, M_0, n]^{T} \\

    where :math:`i` the inclination (radian), :math:`\Omega` is the
    longitude of the ascending node (radian), :math:`e` is the orbital
    eccentricity (unitless), :math:`\omega` the argument of perigee
    (radian), :math:`M_0` the mean anomaly (radian) and :math:`n` the
    mean motion (radian/[time])

    For sampling, the :attr:`transition_noise` parameter,
    :math:`\epsilon`, should be used to draw from
    :math:`\mathcal{N}(M_{t_1},\epsilon)`

    TODO: test the efficiency of this method

    """

    transition_noise = Property(
        float, default=0.0, doc=r"Transition noise :math:`\epsilon`")

    def matrix(self):
        pass

    def ndim_state(self):
        """The transition operates on the 6-dimensional orbital state vector

        Returns
        -------
        : int
            6
        """
        return 6

    def function(self, orbital_state, noise=0,
                 time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : float, optional
            The nominal standard deviation function parameter. This isn't
            passed to the transition function though, so can be left out.
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state, (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Note
        ----
            This merely passes the parameters to the :attr:`.transition()`
            function. The noise parameter has no effect, is included merely
            for compatibility with parent classes.  Units of mean motion must
            be :math:`\mathrm{rad} \, s^{-1}`

        """
        self.transition(orbital_state, time_interval)

    def transition(self, orbital_state,
                   time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state: :class:`~.OrbitalState`
            The prior orbital state
        time_interval: :attr:`datetime.timedelta`, optional
            The time interval over which to calculate the new state
            (default: 0 seconds)

        Returns
        -------
        : :class:`~.StateVector`
            The orbital state vector returned by the transition
            function

        Note
        ----
        Units of mean motion must be :math:`\mathrm{rad} \, s^{-1}`

        Warning
        -------
        Doesn't do the addition of noise. Use the sampling
        (:attr:`rvs()`) function instead

        """
        mean_anomaly = orbital_state.mean_anomaly
        mean_motion = orbital_state.mean_motion
        tle_state = orbital_state.two_line_element

        # TODO: Enforce the units of mean_motion are rad/s

        new_mean_anomaly = np.remainder(
            mean_anomaly + mean_motion * time_interval.total_seconds(),
            2 * np.pi)
        new_tle_state = np.insert(np.delete(tle_state, 4, 0), 4,
                                  new_mean_anomaly, axis=0)

        return OrbitalState(new_tle_state, coordinates='TLE',
                            timestamp=orbital_state.timestamp + time_interval,
                            grav_parameter=orbital_state.grav_parameter). \
            cartesian_state_vector

    def rvs(self, num_samples=1, orbital_state=None,
            time_interval=timedelta(seconds=0)):
        r"""Generate samples from the transition function. Assume that
        the noise is additive and Gauss-distributed in mean anomaly
        only. So,

            .. math::

                M_{t_1} = M_{t_0} + n(t_1 - t_0) + \zeta,
                (\mathrm{modulo} \, 2\pi)

                \zeta \sim \mathcal{N}(0, \epsilon)

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:`~.OrbitalState`, optional
            The orbital state class, (default is None, in which case the state
            is created with a mean anomaly of 0 and a date of 1st Jan 1970)
        time_interval : :class:`~.datetime.timedelta`, optional
            The time over which the transition occurs

        Returns
        -------
        : numpy.array, (of dimension 6 x num_samples)
            N random samples of the state vector drawn from a normal
            distribution defined by the transited mean anomaly,
            :math:`M_{t_1}` and the standard deviation :math:`\epsilon`

        Note
        ----
        Units of mean motion must be in :math:`\mathrm{rad} s^{-1}`

        """
        # Generate the samples
        mean_anomalies = np.random.normal(0, self.transition_noise,
                                          num_samples)

        # Use the passed state, or generate a 0-state, as the mean state
        if orbital_state is None:
            meanstate = OrbitalState(np.zeros((6, 1)), coordinates="TLE",
                                     timestamp=datetime(1970, 1, 1))
        else:
            meanstate = orbital_state

        meantlestate = meanstate.two_line_element

        out = np.zeros((6, 0))
        for mean_anomaly in mean_anomalies:
            currenttlestatev = np.remainder(meantlestate +
                                            np.array([[0], [0], [0], [0],
                                                      [mean_anomaly], [0]]),
                                            2*np.pi)
            currentstate = TLEOrbitalState(currenttlestatev,
                                           timestamp=orbital_state.timestamp,
                                           grav_parameter=meanstate.
                                           grav_parameter)
            out = np.append(out, currentstate.cartesian_state_vector, axis=1)

        return out  # to return an array of dimension 6xNDim

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`t_0 + \delta t` for a
        given test orbital state. Assumes constancy in all terms except
        the mean anomaly which is Gauss distributed according to
        :math:`\epsilon`.

        Parameters
        ----------
        test_state: :class:`~.OrbitalState`
            The orbital state vector to test
        orbital_state: :class:`~.OrbitalState`
            The prior orbital state class
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default
            is 0)

        Returns
        -------
        : float
            Value of the pdf at :attr:`test_state` and :math:`t + \delta t`

        Note
        ----
        Units of mean motion must be in :math:`\mathrm{rad} s^{-1}`

        """
        # First transit the prior to the current
        trans_state = OrbitalState(self.transition(orbital_state,
                                                   time_interval=time_interval
                                                   ),
                                   timestamp=orbital_state.timestamp +
                                   time_interval,
                                   grav_parameter=orbital_state.grav_parameter)

        return norm.pdf(test_state.mean_anomaly,
                        loc=trans_state.mean_anomaly,
                        scale=self.transition_noise)


class CartesianTransitionModel(OrbitalTransitionModel, NonLinearModel):
    """This class invokes a transition model in Cartesian coordinates
    assuming a Keplerian orbit. A calculation of the universal anomaly
    is used which relies on an approximate method (much like the
    eccentric anomaly) but repeated calls to a constructor (as in
    mean-anomaly-based method are avoided.

    Follows algorithm 3.4 in [1].

    References
    ----------
    1. Curtis, H.D 2010, Orbital Mechanics for Engineering Students (3rd
    Ed.), Elsevier Publishing

    """
    noise = Property(
        CovarianceMatrix, default=CovarianceMatrix(np.zeros((6, 6))),
        doc=r"Transition noise covariance :math:`\Sigma` per unit time "
            r"interval (assumed seconds)")

    _uanom_precision = Property(
        float, default=1e-8, doc="The stopping point in the progression of "
                                 "ever smaller f/f' ratio in the the Newton's "
                                 "method calculation of universal anomaly."
    )

    @property
    def ndim_state(self):
        """Dimension of the state vector is 6

        Returns
        -------
        : int
            The dimension of the state vector, i.e. 6

        """
        return 6

    def covar(self, time_interval=timedelta(seconds=1)):
        r"""Return the transition covariance matrix

        Parameters
        ----------
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 1
            second)

        Returns
        -------
        : CovarianceMatrix
            The transition covariance matrix

        """
        return CovarianceMatrix(self.noise * time_interval.total_seconds())

    def function(self, orbital_state, noise=0,
                 time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the transition function

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : CovarianceMatrix, optional
            The nominal covariance matrix. This isn't passed to the transition
            function though, so can be left out.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Note
        ----
        This merely passes the parameters to the :attr:`.transition()`
        function. The noise parameter has no effect, is included merely
        for compatibility with parent classes.
        """
        return self.transition(orbital_state, time_interval)

    def transition(self, orbital_state, time_interval=timedelta(seconds=0)):
        r"""The transition proceeds as algorithm 3.4 in [1]

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default
            is 0 seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Warning
        -------
        If noisy samples from the transition function are required, use
        the :attr:`rvs` method.
        """
        # Get the position and velocity vectors
        bold_r_0 = orbital_state.cartesian_state_vector[0:3]
        bold_v_0 = orbital_state.cartesian_state_vector[3:6]

        # Get the Lagrange coefficients via the universal anomaly
        f, g, f_dot, g_dot = lagrange_coefficients_from_universal_anomaly(
            orbital_state.cartesian_state_vector, time_interval,
            grav_parameter=orbital_state.grav_parameter,
            precision=self._uanom_precision)

        # Get the position vector
        bold_r = f * bold_r_0 + g * bold_v_0
        # The velocity vector
        bold_v = f_dot * bold_r_0 + g_dot * bold_v_0

        # And put them together
        return np.concatenate((bold_r, bold_v), axis=0)

    def rvs(self, num_samples=1, orbital_state=None,
            time_interval=timedelta(seconds=0)):
        r"""Sample from the transited state. Do this in a fairly simple-minded
        way by way of additive white noise in Cartesian coordinates.

        .. math::

            \mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{\zeta},
            \mathbf{\zeta} \sim \mathcal{N}(\mathbf{0}, \Sigma)

        where

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:`~.OrbitalState`, optional
            The orbital state class (default is None, in which case a
            Gauss-distributed samples are generated at Cartesian
            :math:`[0,0,0,0,0,0]^T`)
        time_interval : :class:`~.datetime.timedelta`, optional
            The time over which the transition occurs, (default is 0)

        Returns
        -------
        : numpy.array, dimension (6, num_samples)
            num_samples random samples of the state vector drawn from
            a normal distribution defined by the transited mean
            anomaly, :math:`M_{t_1}` and the covariances
            :math:`\Sigma`.

        """
        samples = multivariate_normal.rvs(mean=np.zeros(6), cov=self.covar(
            time_interval=time_interval), size=num_samples)

        # multivariate_normal.rvs() does stupid in the case of a single sample
        # so we have to put this back
        if num_samples == 1:
            samples = np.array([samples])

        if orbital_state is None:
            return samples.T
        else:
            new_cstate_vector = self.transition(
                orbital_state, time_interval=time_interval)
            return matlib.repmat(new_cstate_vector, 1, num_samples) + \
                samples.T

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`\delta t` for a given test
        orbital state. Assumes multi-variate normal distribution in Cartesian
        position and velocity coordinates.

        Parameters
        ----------
        test_state: :class:`~.OrbitalState`
            The orbital state vector to test.
        orbital_state: :class:`~.OrbitalState`
            The 'mean' orbital state class. This undergoes the state transition
            before comparison with the test state.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`
            The time interval over which to test the new state

        Returns
        -------
        : float
            The value of the pdf at 'test_state'


        """

        return multivariate_normal.pdf(test_state.cartesian_state_vector.
                                       ravel(),
                                       mean=self.transition(orbital_state,
                                                            time_interval).
                                       ravel(),
                                       cov=self.covar(
                                           time_interval=time_interval))
        # ravel appears to be necessary here.


class SGP4TransitionModel(OrbitalTransitionModel, NonLinearModel):
    """This class wraps https://pypi.org/project/sgp4/

    References
    ----------
    1. https://pypi.org/project/sgp4/

    """
    noise = Property(
        CovarianceMatrix, default=CovarianceMatrix(np.zeros((6, 6))),
        doc=r"Transition noise covariance :math:`\Sigma` per unit time "
            r"interval (assumed seconds)")

    @property
    def ndim_state(self):
        """Dimension of the state vector is 6

        Returns
        -------
        : int
            The dimension of the state vector, i.e. 6

        """
        return 6

    def covar(self, time_interval=timedelta(seconds=1)):
        r"""Return the transition covariance matrix

        Parameters
        ----------
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 1
            second)

        Returns
        -------
        : CovarianceMatrix
            The transition covariance matrix

        """
        return CovarianceMatrix(self.noise * time_interval.total_seconds())

    def function(self, orbital_state, noise=0,
                 time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the transition function

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : CovarianceMatrix, optional
            The nominal covariance matrix. This isn't passed to the transition
            function though, so can be left out.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Note
        ----
        This merely passes the parameters to the :attr:`.transition()`
        function. The noise parameter has no effect, is included merely
        for compatibility with parent classes.
        """
        return self.transition(orbital_state, time_interval)

    def transition(self, orbital_state, time_interval=timedelta(seconds=0)):
        r"""

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default
            is 0 seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Warning
        -------
        If noisy samples from the transition function are required, use
        the :attr:`rvs` method.
        """

        # Evaluated at initial timestamp
        tle_ext = Satrec.twoline2rv(orbital_state.metadata['line_1'],
                                    orbital_state.metadata['line_2'])

        # Predict over time interval
        tt = orbital_state.timestamp + time_interval

        jd, fr = jday(tt.year, tt.month, tt.day,
                      tt.hour, tt.minute, tt.second)
        # WARNING: These units returned as km and km/s
        e, bold_r, bold_v = tle_ext.sgp4(jd, fr)

        # Update the metadata?

        # And put them together
        return np.concatenate((bold_r, bold_v), axis=0)

    def rvs(self, num_samples=1, orbital_state=None,
            time_interval=timedelta(seconds=0)):
        r"""Sample from the transited state. Do this in a fairly simple-minded
        way by way of additive white noise in Cartesian coordinates.

        .. math::

            \mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{\zeta},
            \mathbf{\zeta} \sim \mathcal{N}(\mathbf{0}, \Sigma)

        where

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:`~.OrbitalState`, optional
            The orbital state class (default is None, in which case a
            Gauss-distributed samples are generated at Cartesian
            :math:`[0,0,0,0,0,0]^T`)
        time_interval : :class:`~.datetime.timedelta`, optional
            The time over which the transition occurs, (default is 0)

        Returns
        -------
        : numpy.array, dimension (6, num_samples)
            num_samples random samples of the state vector drawn from
            a normal distribution defined by the transited mean
            anomaly, :math:`M_{t_1}` and the covariances
            :math:`\Sigma`.

        """
        samples = multivariate_normal.rvs(mean=np.zeros(6), cov=self.covar(
            time_interval=time_interval), size=num_samples)

        # multivariate_normal.rvs() does stupid in the case of a single sample
        # so we have to put this back
        if num_samples == 1:
            samples = np.array([samples])

        if orbital_state is None:
            return samples.T
        else:
            new_cstate_vector = self.transition(
                orbital_state, time_interval=time_interval)
            return matlib.repmat(new_cstate_vector, 1, num_samples) + \
                samples.T

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`\delta t` for a given test
        orbital state. Assumes multi-variate normal distribution in Cartesian
        position and velocity coordinates.

        Parameters
        ----------
        test_state: :class:`~.OrbitalState`
            The orbital state vector to test.
        orbital_state: :class:`~.OrbitalState`
            The 'mean' orbital state class. This undergoes the state transition
            before comparison with the test state.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`
            The time interval over which to test the new state

        Returns
        -------
        : float
            The value of the pdf at 'test_state'


        """

        return multivariate_normal.pdf(test_state.cartesian_state_vector.
                                       ravel(),
                                       mean=self.transition(orbital_state,
                                                            time_interval).
                                       ravel(),
                                       cov=self.covar(
                                           time_interval=time_interval))
        # ravel appears to be necessary here.
