# -*- coding: utf-8 -*-

import numpy as np
from datetime import timedelta
from ....orbital_functions import lagrange_coefficients_from_universal_anomaly

from ....base import Property
from ....types.array import StateVector
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

    TODO: test the efficiency of this method

    """

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

    def function(self, orbital_state, noise=False,
                 time_interval=timedelta(seconds=0)):
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
        self.transition(orbital_state, noise=noise, time_interval=time_interval)

    def transition(self, orbital_state, noise=False,
                   time_interval=timedelta(seconds=0)):
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

        # Question: would it be quicker to do an array sum?

        mean_anomaly = orbital_state.mean_anomaly
        mean_motion = orbital_state.mean_motion
        tle_state = orbital_state.two_line_element

        # TODO: Enforce the units of mean_motion are rad/s

        new_mean_anomaly = np.remainder(
            mean_anomaly + mean_motion * time_interval.total_seconds(),
            2 * np.pi)
        return StateVector(np.insert(np.delete(tle_state, 4, 0), 4, new_mean_anomaly, axis=0)) \
            + noise


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

    def function(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the transition function

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
        This merely passes the parameters to the :attr:`.transition()`
        function. The noise parameter has no effect, is included merely
        for compatibility with parent classes.
        """
        return self.transition(orbital_state, noise=noise, time_interval=time_interval)

    def transition(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
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


class SGP4TransitionModel(OrbitalTransitionModel, NonLinearModel):
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
        tle_ext = Satrec.twoline2rv(orbitalstate.metadata['line_1'],
                                    orbitalstate.metadata['line_2'])

        # Predict over time interval
        tt = orbitalstate.timestamp + time_interval

        jd, fr = jday(tt.year, tt.month, tt.day,
                      tt.hour, tt.minute, tt.second)

        # WARNING: These units returned as km and km/s
        e, bold_r, bold_v = tle_ext.sgp4(jd, fr)

        return e, bold_r, bold_v

    def function(self, orbital_state, noise=False,
                 time_interval=timedelta(seconds=0)):
        r"""Just passes parameters to the transition function

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
        return self.transition(orbital_state, noise=noise, time_interval=time_interval)

    def transition(self, orbital_state, noise=False, time_interval=timedelta(seconds=0)):
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
