# -*- coding: utf-8 -*-

from datetime import timedelta

import numpy as np

from sgp4.api import jday, Satrec

from ....base import Property
from ....types.array import StateVector, Matrix
from ....types.angle import Inclination, EclipticLongitude
from .base import OrbitalGaussianTransitionModel
from ...base import LinearModel, NonLinearModel
from ....functions.orbital import lagrange_coefficients_from_universal_anomaly


class CartesianKeplerianTransitionModel(OrbitalGaussianTransitionModel, NonLinearModel):
    """This class invokes a transition model in Cartesian coordinates assuming a Keplerian orbit.
    A calculation of the universal anomaly is used which relies on an approximate method (much
    like the eccentric anomaly) but repeated calls to a constructor are avoided.

    Follows algorithm 3.4 in [1].

    Process noise is additive and sampled from a multivariate-normal distribution in Cartesian
    coordinates.

    References
    ----------
    1. Curtis, H.D 2010, Orbital Mechanics for Engineering Students (3rd Ed.), Elsevier Publishing

    """
    _uanom_precision = Property(
        float, default=1e-8, doc="The stopping point in the progression of "
                                 "ever smaller f/f' ratio in the the Newton's "
                                 "method calculation of universal anomaly."
    )

    @property
    def ndim_state(self):
        """Dimension of the state vector is 6"""
        return 6

    def transition(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the function which executes the transition

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        Note
        ----
        This merely passes the parameters to the :meth:`.function()` function.
        """
        return self.function(orbital_state, noise=noise, time_interval=time_interval)

    def function(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""The transition proceeds as algorithm 3.4 in [1]

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default
            is 0 seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function
        """
        noise = self._noiseinrightform(noise, time_interval=time_interval)

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
        return StateVector(np.concatenate((bold_r, bold_v), axis=0)) + noise


class TLEKeplerianTransitionModel(OrbitalGaussianTransitionModel, LinearModel):
    r"""This transition model uses the mean motion to update the mean anomaly in simple Keplerian
    fashion. It returns a state vector in TLE format. The function is linear (hence it's a linear
    model) and the uncertainty is multivariate normal in the elements of the TLE. It therefore
    differs from the :class:`~.CartesianKeplerianTransitionModel` only in the addition of noise.
    It should be clear therefore that the :attr:`process_noise` covariance matrix is in units of
    the TLE elements.

    The transition proceeds as,

        .. math::

            M_{t_1} = M_{t_0} + n(t_1 - t_0), \; (\mathrm{modulo} \, 2\pi)


    for the interval :math:`t_0 \rightarrow t_1`, where :math:`n` is the mean motion, computed as:

        .. math::

           n = \sqrt{ \frac{\mu}{a^3} }

    which is in units of :math:`\mathrm{rad} \, s^{-1}`. The state vector is then recreated from
    the TLE parameterisation of the orbital state vector:

        .. math::

            X = [i, \Omega, e, \omega, M_0, n]^{T} \\

    where :math:`i` the inclination (radian), :math:`\Omega` is the longitude of the ascending
    node (radian), :math:`e` is the orbital eccentricity (unitless), :math:`\omega` the argument
    of perigee (radian), :math:`M_0` the mean anomaly (radian) and :math:`n` the mean motion
    (radian/[time])

    """

    def matrix(self, time_interval, **kwargs):
        """Return the transition matrix
        """
        matrix_out = Matrix(np.diag(np.ones(6)))
        matrix_out[4, 5] = time_interval.total_seconds()
        return matrix_out

    @property
    def ndim_state(self):
        """The dimension of the state vector, 6"""
        return 6

    def transition(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
        time_interval: :math:`dt` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state, (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function. It is in 'TLE' form:
            :math:`X = [i, \Omega, e, \omega, M_0, n]^{T}`

        Note
        ----
            This merely passes the parameters to the :attr:`.transition()`
            function.  Units of mean motion must be :math:`\mathrm{rad} \, s^{-1}`

        """
        return self.function(orbital_state, noise=noise, time_interval=time_interval)

    def function(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""Execute the transition function

        Parameters
        ----------
        orbital_state: :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
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

        """
        noise = self._noiseinrightform(noise, time_interval=time_interval)

        out_statev = self.matrix(time_interval) @ orbital_state.two_line_element + noise

        # preserve type - must be a better way
        return StateVector([Inclination(out_statev[0]), EclipticLongitude(out_statev[1]),
                            out_statev[2], EclipticLongitude(out_statev[3]),
                            EclipticLongitude(out_statev[4]), out_statev[5]])


class SGP4TransitionModel(OrbitalGaussianTransitionModel, NonLinearModel):
    """This class wraps https://pypi.org/project/sgp4/

    Note that the transition works slightly differently from other versions. The TLE needs to be
    included in the metadata, and it is not updated. The content of the orbital state vector need
    not be consistent with the TLE and is ignored in this transition model. The state vector
    returned is in the TEME Cartesian frame in km, km s^{-1}.

    This presents an issue for the rvs function. As inherited, that function adds noise to the
    transitioned state - so needs to be in TEME Cartesian coordinates.

    References
    ----------
    1. https://pypi.org/project/sgp4/

    """

    @property
    def ndim_state(self):
        """Dimension of the state vector is 6

        Returns
        -------
        : int
            The dimension of the state vector, i.e. 6

        """
        return 6

    def _advance_by_metadata(self, orbitalstate, time_interval):
        # Evaluated at initial timestamp
        tle_as_dict = orbitalstate.tle_dict
        tle_ext = Satrec.twoline2rv(tle_as_dict['line_1'], tle_as_dict['line_2'])

        # scale factor in [length]/km - will multiply the outputs by this factor because the
        # output of sgp4 is in km.
        scale_fac = np.cbrt(orbitalstate.grav_parameter/3.986004418e5)

        # Predict over time interval
        tt = orbitalstate.timestamp + time_interval

        jd, fr = jday(tt.year, tt.month, tt.day,
                      tt.hour, tt.minute, tt.second)

        # WARNING: r and v returned as km and km/s so mut be multiplied by scale_fac
        e, bold_r, bold_v = tle_ext.sgp4(jd, fr)

        return e, tuple(br*scale_fac for br in bold_r), tuple(bv*scale_fac for bv in bold_v)

    def transition(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the function function

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 0
            seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function

        """
        return self.function(orbital_state, noise=noise, time_interval=time_interval)

    def function(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""

        Parameters
        ----------
        orbital_state : :class:`~.OrbitalState`
            The prior orbital state
        noise : bool or StateVector, optional
            A boolean parameter or a state vector. If the latter this represents an additive noise
            (or bias) term added to the transited state. If True the noise vector is sampled via
            the :meth:`rvs()` function. If False, noise is not added.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default
            is 0 seconds)

        Returns
        -------
        : StateVector
            The orbital state vector returned by the transition function. In Cartesian coordinates
            in km, km s^{-1}

        """
        noise = self._noiseinrightform(noise, time_interval=time_interval)

        e, bold_r, bold_v = self._advance_by_metadata(orbital_state, time_interval)

        # And put them together
        return StateVector(np.concatenate((bold_r, bold_v), axis=0)) + noise
