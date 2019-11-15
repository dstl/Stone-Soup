# -*- coding: utf-8 -*-
r"""Test the various constructions of the orbital state vector. Take a known
orbital state vector and check the various parameterisations.

Example 4.3 from Curtis. Take the orbital state vector as input and check the
various output parameterisations. The input state vector is:

    .. math::

        \mathbf{r} = [-6045 \, -3490 \, 2500] \mathrm{km}

        \mathbf{v} = [-3.457 \, 6.618 \, 2.553] \mathrm{km s^{-1}}

Selected outputs should be:

    magnitude of the specific orbital angular momentum, :math:`h = 58,310
    \mathrm{km^2 s^{-1}}`

For the Keplerian elements
    semi-major axis, :math:`8788 \mathrm{km}`

    eccentricity, :math:`0.1712`

    inclination, :math:`2.674 \mathrm{rad}`

    longitude of ascending node, :math:`4.456 \mathrm{rad}`

    argument of periapsis, :math:`0.3503 \mathrm{rad}`

    true anomaly, :math:`0.4965 \mathrm{rad}`

TLE stuff
    eccentric anomaly, :math:`0.4202 \mathrm{rad}`

    mean anomaly, :math:`0.3504 \mathrm{rad}`

    period, :math:`8201 \mathrm{s}`

    mean motion, :math:`0.0007662 \mathrm{rad} \, \mathrm{s}^{-1}`

Equinoctial
     horizontal component of eccentricity, :math:`h = -0.1704`

     vertical component of eccentricity, :math:`k = 0.01605`

     horizontal component of the inclination, :math:`p =

     vertical component of the inclination :math:`q`

      mean longitude



"""
import numpy as np
import pytest
from ..orbitalstate import OrbitalState
from ..orbitalstate import KeplerianOrbitalState, TLEOrbitalState, \
    EquinoctialOrbitalState

# Orbital state vector in km and km/s
orb_st_vec = np.array([[-6045], [-3490], [2500], [-3.457], [6.618], [2.533]])
cartesian_s = OrbitalState(orb_st_vec, coordinates='Cartesian')
# ensure that the Gravitational parameter is in km^3 s^-2
cartesian_s.grav_parameter = cartesian_s.grav_parameter/1e9

# The Keplarian elements should be (to 4sf)
out_kep = np.array([[0.1712], [8788], [2.674], [4.456], [0.3503], [0.4965]])
keplerian_s = OrbitalState(out_kep, coordinates='Keplerian',
                           grav_parameter=cartesian_s.grav_parameter)

# The TLE should be (to 4sf)
out_tle = np.array([[2.674], [4.456], [0.1712], [0.3503], [0.3504], [0.0007662]])

# Equinoctial elements are (again, 4sf)
out_equ = np.array([[8788], [-0.1704], [0.01605], [-4.062], [-1.065], [5.157]])


# The next three tests ensure that the initialisations in different forms
# yield the same results
def test_kep_cart():
    # Test that Keplerian initialisation yields same state vector
    # Firstly just flipping back and forth
    keplerian_sn = OrbitalState(cartesian_s.keplerian_elements,
                                coordinates='Keplerian',
                                grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(cartesian_s.state_vector, keplerian_sn.state_vector,
                       rtol=1e-4))

    # independent initialisation
    assert(np.allclose(keplerian_s.state_vector, orb_st_vec, rtol=2e-3))


def test_tle_cart():
    tle_sn = OrbitalState(cartesian_s.two_line_element, coordinates='TLE',
                          grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(cartesian_s.two_line_element, tle_sn.two_line_element,
                       rtol=1e-3))


def test_equ_cart():
    equ_sn = OrbitalState(cartesian_s.equinoctial_elements,
                          coordinates='Equinoctial',
                          grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(cartesian_s.equinoctial_elements,
                       equ_sn.equinoctial_elements, rtol=1e-3))


# Now we need to test that the output is actually correct.
# Test Cartesian input and Keplerian output on known equivalents
def test_cart_kep():
    # Simple assertion
    assert(np.all(cartesian_s.state_vector == orb_st_vec))
    # Check Keplerian elements come out right
    assert(np.allclose(cartesian_s.keplerian_elements, out_kep, rtol=1e-3))


# The test TLE output
def test_cart_tle():
    assert(np.allclose(cartesian_s.two_line_element, out_tle, rtol=1e-3))


# Test some specific quantities
# Next tests are to ensure that the daughter classes still work
def test_keplerian_init():
    """Keplerian elements"""

    k = KeplerianOrbitalState(out_kep, grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(k.state_vector, orb_st_vec, rtol=1e-2))


def test_tle_init():
    """Init TLE derived class"""

    tle = TLEOrbitalState(out_tle, grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(tle.state_vector, orb_st_vec, rtol=1e-2))


def test_equ_init():
    """Init Equinoctial derived class"""

    equ = EquinoctialOrbitalState(out_equ, grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(equ.state_vector, orb_st_vec, rtol=1e-2))
