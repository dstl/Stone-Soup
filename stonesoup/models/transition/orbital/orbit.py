# -*- coding: utf-8 -*-

from datetime import timedelta

import numpy as np

from ....base import Property
from ....types.array import StateVector
from .base import OrbitalGaussianTransitionModel
from ...base import NonLinearModel
from ....functions.orbital import lagrange_coefficients_from_universal_anomaly


class CartesianKeplerianTransitionModel(OrbitalGaussianTransitionModel, NonLinearModel):
    """This class invokes a transition model in Cartesian coordinates assuming a Keplerian orbit.
    A calculation of the universal anomaly is used which relies on an approximate method (much
    like the eccentric anomaly) but repeated calls to a constructor are avoided.

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
