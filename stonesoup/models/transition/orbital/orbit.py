# -*- coding: utf-8 -*-

import copy
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta

from ....base import Property
from ....types.orbitalstate import OrbitalState, TLEOrbitalState
from ....types.array import CovarianceMatrix
#from ....astro_functions import calculate_itr_eccentric_anomaly
from ...base import LinearModel, NonLinearModel
from .base import OrbitalTransitionModel


class SimpleMeanMotionTransitionModel(OrbitalTransitionModel, LinearModel):

    r"""This simple transition model uses the mean motion to update the
    mean anomaly and then use that to construct a new orbital state
    vector via the TLE parameterisation. Statistics assume a
    Gauss-distributed mean anomaly with all other quantities fixed.

    The transition proceeds as,

        .. math::

            M_{t_1} = M_{t_0} + n(t_1 - t_0), (\mathrm(mod) \, 2\pi)


    for the interval :attr:`OrbitalState.timestamp` :math:`t_1 > t_0`
    where :math:`n` is the mean motion, computed as:

        .. math::

           n = \sqrt{ \frac{\mu}{a^3} }

    which is in units of :math:`\mathbf{\mathrm{rad} \, s^{-1}}`. The
    state vector is then recreated from the TLE parameterisation of
    the orbital state vector:

        .. math::

            X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T} \\

        where :math:`i` the inclination (rad),
        :math:`\Omega` is the longitude of the ascending node (rad),
        :math:`e` is the orbital eccentricity (unitless),
        :math:`\omega` the argument of perigee (rad),
        :math:'M_0' the mean anomaly (rad)
        :math:`n` the mean motion (rad/unit time)

    For sampling, the parameter :math:`\epsilon` should be used to draw
    from:
        .. math::

            \mathcal{N}(M_{t_1},\epsilon)

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

    def rvs(self, num_samples=1, orbital_state=None,
            time_interval=timedelta(seconds=0)):
        r"""Generate samples from the transition function. Assume that the
        noise is additive and Gauss-distributed in mean anomaly only. So,

        .. math::

            M_{t_1} = M_{t_0} + n(t_1 - t_0) + \epsilon, (\mathrm(mod) \, 2\pi)

            \epsilon ~ \mathcal{N}(0, \sigma_{M_0})

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:~`OrbitalState`, optional
            The orbital state class, (default is None, in which case the state
            is created with a mean anomaly of 0)
        time_interval : :class:~`datetime.timedelta`, optional
            The time over which the transition occurs

        Returns
        -------
        : numpy.array, (of dimension 6 x num_samples)
            N random samples of the state vector drawn from a normal
            distribution defined by the transited mean anomaly,
            :math:`M_{t_1}` and the standard deviation :math:`\epsilon`.

        Note
        ----
        Units of mean motion must be in :math:`\mathrm{rad} s^{-1}`

        """
        if orbital_state is None:
            orbital_state = OrbitalState(np.zeros((6,1)), coordinates="TLE",
                                         timestamp=datetime(1970, 1, 1))

        mean_anomalies = np.random.normal(0, self.transition_noise,
                                          num_samples)

        # Iterate to create a list. May be a better way.
        out = np.array((0, 6)) # empty 6-d array
        for a_mean_anomaly in mean_anomalies:
            # Noise is additive so we can do this first...
            new_mean_anomaly = self.transition(orbital_state,
                                               time_interval).mean_anomaly + \
                               a_mean_anomaly
            new_tle_state = np.insert(np.delete(orbital_state.two_line_element,
                                                4, 0), 4, new_mean_anomaly,
                                      axis=0)

            outstate = OrbitalState(new_tle_state,  coordinates="TLE",
                                    timestamp=orbital_state.timestamp +
                                              time_interval,
                                    grav_parameter=
                                    orbital_state.grav_parameter)

            out = np.append(out, outstate.cartesian_state_vector)

        return out.T # to return an array of dimension 6xNDim

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`\Delta t` for a given test
        orbital state. Assumes constancy in all terms except the mean anomaly
        which is Gauss distributed according to :math:`\epsilon`.

        Parameters
        ----------
        test_state: :class:~`OrbitalState`
            The orbital state vector to test
        orbital_state: :class:~`OrbitalState`
            The prior orbital state class
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0)

        Returns
        -------
        : float

        Note
        ----
        Units of mean motion must be in :math:`\mathrm{rad} s^{-1}`

        """
        return norm.pdf(test_state.mean_anomaly,
                        loc=self.transition(orbital_state,
                                            time_interval).mean_anomaly,
                        scale=self.transition_noise)

    def function(self, orbital_state, noise=0,
                 time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state : :class:~`OrbitalState`
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

        Notes
        -----
            This merely passes the parameters to the :attr:`.transition()`
            function. The noise parameter has no effect, is included merely
            for compatibility with parent classes.  Units of mean motion must
            be :math:`\mathbf{rad} \, s^{-1}`

        """
        self.transition(orbital_state, time_interval)

    def transition(self, orbital_state,
                   time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state: :class:~`OrbitalState`
            The prior orbital state
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to calculate the new state (default is
            0 seconds)

        Returns
        -------
        : :class:~`StateVector`
            The orbital state vector returned by the transition function

        Note
        ----
        Units of mean motion must be :math:`\mathbf{rad} \, s^{-1}`

        Warning
        -------
        Doesn't do the addition of noise. Use the sampling (rvs) function
        instead.

        """
        mean_anomaly = orbital_state.mean_anomaly
        mean_motion = orbital_state.mean_motion
        tle_state = orbital_state.two_line_element

        # TODO: Ensure that the units of mean_motion are rad/s

        new_mean_anomaly = np.remainder(
            mean_anomaly + mean_motion*time_interval.total_seconds(),
            2*np.pi)
        new_tle_state = np.insert(np.delete(tle_state, 4, 0), 4,
                                  new_mean_anomaly, axis=0)

        return OrbitalState(new_tle_state, coordinates='TLE',
                            timestamp=orbital_state.timestamp + time_interval,
                            grav_parameter=orbital_state.grav_parameter).\
            cartesian_state_vector


class CartesianTransitionModel(OrbitalTransitionModel, NonLinearModel):
    """This class invokes a transition model in Cartesian coordinates
    assuming a Keplerian orbit. A calculation of the ''universal
    anomaly'' is used which relies on an approximate method (much like
    the eccentric anomaly) but repeated calls to a constructor (as in
    mean-anomaly-based method are avoided.

    Follows algorithm 3.4 in [1].

    Reference
    ---------
    Curtis, H.D 2010, Orbital Mechanics for Engineering Students (3rd
    Ed.), Elsevier Publishing

    """
    noise = Property(
        CovarianceMatrix, default=CovarianceMatrix(np.zeros([6, 6])),
        doc=r"Transition noise :math:`\epsilon`")

    @property
    def ndim_state(self):
        """Dimension of the state vector is 6

        Returns
        -------
        : int
            The dimension of the state vector, i.e. 6

        """
        return 6

    def function(self, orbital_state, noise=0,
                 time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the transition function

        Parameters
        ----------
        orbital_state : :class:~`OrbitalState`
            The prior orbital state
        noise : CovarianceMatrix, optional
            The nominal covariance matrix. This isn't passed to the transition
            function though, so can be left out.
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Notes
        -----
            This merely passes the parameters to the :attr:`.transition()`
            function. The noise parameter has no effect, is included merely
            for compatibility with parent classes.  Units of mean motion must
            be :math:`\mathbf{rad} \, s^{-1}`
        """
        return self.transition(orbital_state, time_interval)

    def transition(self, orbital_state, time_interval=timedelta(seconds=0)):
        r"""The transition proceeds as algorithm 3.4 in [1]

        Parameters
        ----------
        orbital_state : :class:~`OrbitalState`
            The prior orbital state
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Warning
        -------
        If noisy samples from the transition function are required, use the rvs
        method. Units of mean motion must be :math:`\mathbf{rad} \, s^{-1}`

        """
        # Reused variables
        root_mu = np.sqrt(orbital_state.grav_parameter)
        inv_sma = 1. / orbital_state.semimajor_axis

        # Some helpful functions
        def calculate_universal_anomaly(mag_r_0, v_rad_0, delta_t,
                                        tolerance=1e-8):
            """Algorithm 3.3 in [1]"""

            # Initial estimate of Chi
            chi_i = root_mu * np.abs(inv_sma) * delta_t.total_seconds()
            ratio = 1

            # Do Newton's method
            while np.abs(ratio) > tolerance:
                z_i = inv_sma * chi_i**2
                f_chi_i = mag_r_0 * v_rad_0/root_mu * chi_i**2 * stumpf_c(z_i)\
                          + (1 - inv_sma * mag_r_0) * chi_i**3 * stumpf_s(z_i) \
                          + mag_r_0 * chi_i - root_mu * delta_t.total_seconds()
                fp_chi_i = mag_r_0 * v_rad_0/root_mu * chi_i * \
                           (1 - inv_sma * chi_i**2 * stumpf_s(z_i)) + \
                           (1 - inv_sma * mag_r_0) * chi_i**2 * stumpf_c(z_i) \
                           + mag_r_0
                ratio = f_chi_i/fp_chi_i
                chi_i = chi_i - ratio

            return chi_i

        def stumpf_s(z):
            """The Stumpf S function"""
            if z > 0:
                sqz = np.sqrt(z)
                return (sqz - np.sin(sqz))/sqz**3
            elif z < 0:
                sqz = np.sqrt(-z)
                return (np.sinh(sqz) - sqz)/sqz**3
            elif z == 0:
                return 1/6
            else:
                raise ValueError("Shouldn't get to this point")

        def stumpf_c(z):
            """The Stumpf C function"""
            if z > 0:
                sqz = np.sqrt(z)
                return (1 - np.cos(sqz))/sqz**2
            elif z < 0:
                sqz = np.sqrt(-z)
                return (np.cosh(sqz) - 1)/sqz**2
            elif z == 0:
                return 1/2
            else:
                raise ValueError("Shouldn't get to this point")

        # Calculate the magnitude of the position and velocity vectors
        bold_r_0 = orbital_state.cartesian_state_vector[0:3]
        bold_v_0 = orbital_state.cartesian_state_vector[3:6]

        r_0 = np.sqrt(np.dot(bold_r_0.T, bold_r_0).item())
        v_0 = np.sqrt(np.dot(bold_v_0.T, bold_v_0).item()) # Don't need this as we get this from the orbital state

        # Find the radial component of the velocity by projecting v_0 onto
        # direction of r_0
        v_r_0 = np.dot(bold_r_0.T, bold_v_0).item()/r_0

        # Get the universal anomaly
        u_anom = calculate_universal_anomaly(r_0, v_r_0, time_interval,
                                             tolerance=
                                             orbital_state._eanom_precision)

        # For convenience
        z = inv_sma * u_anom**2

        # Get the Lagrange coefficients
        f = 1 - u_anom**2 / r_0 * stumpf_c(z)
        g = time_interval.total_seconds() - 1/root_mu * u_anom**3 * stumpf_s(z)

        # Get the position vector and magnitude of that vector
        bold_r = f * bold_r_0 + g * bold_v_0
        r = np.sqrt(np.dot(bold_r.T, bold_r).item())

        # and the Lagrange (time) derivatives
        f_dot = root_mu/(r * r_0) * (inv_sma * u_anom**3 * stumpf_s(z) -
                                     u_anom)
        g_dot = 1 - (u_anom**2 / r) * stumpf_c(z)

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
            \mathbf{\zeta} ~ \mathcal{N}(\mathbf{0}, \Sigma_s)

        where

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:~`OrbitalState`, optional
            The orbital state class (default is None, in which case a
            0-position, 0-velocity vector is created)
        time_interval : :class:~`datetime.timedelta`, optional
            The time over which the transition occurs, (default is 0)

        Returns
        -------
        : numpy.array, dimension (6, num_samples)
            N random samples of the state vector drawn from a normal
            distribution defined by the transited mean anomaly,
            :math:`M_{t_1}` and the standard deviation :math:`\epsilon`.

        """
        if orbital_state is None:
            orbital_state = OrbitalState(np.zeros((6, 1)),
                                         coordinates="Cartesian",
                                         timestamp=datetime(1970, 1, 1))

        samples = multivariate_normal.rvs(mean=np.zeros(np.shape(
            self.noise)[0]), cov=self.noise, size=num_samples)

        # multivariate_normal.rvs() does stupid in the case of a single sample
        # so we have to put this back
        if num_samples == 1:
            samples = np.array([samples])

        out = np.array((0, 6))
        for sample in samples:
            new_cstate_vector = self.transition(
                orbital_state, time_interval).cartesian_state_vector + \
                                np.array([sample]).T

            out = np.append(out, new_cstate_vector, axis=0)

        return out.T

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`\Delta t` for a given test
        orbital state. Assumes multi-variate normal distribution in Cartesian
        position and velocity coordinates.

        Parameters
        ----------
        test_state: :class:~`OrbitalState`
            The orbital state vector to test.
        orbital_state: :class:~`OrbitalState`
            The 'mean' orbital state class. This undergoes the state transition
            before comparison with the test state.
        time_interval: :math:`dt` :attr:`datetime.timedelta`
            The time interval over which to test the new state

        Returns
        -------
        : float

        Note
        ----
        Units of mean motion must be in :math:`\mathrm{rad} s^{-1}`

        """

        return multivariate_normal.pdf(test_state.cartesian_state_vector.T,
                                       mean=self.transition(orbital_state,
                                                            time_interval).
                                       cartesian_state_vector.ravel(),
                                       cov=self.noise)
                                       # ravel appears to be necessary here.


