r"""Test the various constructions of the orbital state vector. Take a known orbital state vector
and check the various parameterisations.

Example 4.3 from Curtis. Take the orbital state vector as input and check the various output
parameterisations. The input state vector is:

    .. math::

        \mathbf{r} = [-6045 \, -3490 \, 2500] \mathrm{km}

        \dot{\mathbf{r}} = [-3.457 \, 6.618 \, 2.553] \mathrm{km s^{-1}}

Selected outputs should be:

    magnitude of the specific orbital angular momentum, :math:`h = 58,310 \mathrm{km^2 s^{-1}}`

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
from datetime import datetime

from ...types.array import StateVector
from ..orbitalstate import OrbitalState

# Time
dtime = datetime.now()

# Orbital state vector in km and km/s
orb_st_vec = StateVector([-6045, -3490, 2500, -3.457, 6.618, 2.533])
cartesian_s = OrbitalState(orb_st_vec, coordinates='Cartesian')
# ensure that the Gravitational parameter is in km^3 s^-2
cartesian_s.grav_parameter = cartesian_s.grav_parameter/1e9

# The Keplarian elements should be (to 4sf)
out_kep = StateVector([0.1712, 8788, 2.674, 4.456, 0.3503, 0.4965])
keplerian_s = OrbitalState(out_kep, coordinates='Keplerian',
                           grav_parameter=cartesian_s.grav_parameter, timestamp=dtime)

# The TLE should be (to 4sf)
out_tle = StateVector([2.674, 4.456, 0.1712, 0.3503, 0.3504, 0.0007662])

# Equinoctial elements are (again, 4sf)
out_equ = StateVector([8788, -0.1704, 0.01605, -4.062, -1.065, 5.157])


def test_incorrect_initialisation():
    """Run a bunch of tests to show that initialisations with the wrong parameters will fail.
    """

    bad_stvec = orb_st_vec[0:4]
    with pytest.raises(ValueError):
        OrbitalState(bad_stvec)

    with pytest.raises(ValueError):
        OrbitalState(orb_st_vec, coordinates='Nonsense')

    with pytest.raises(TypeError):
        OrbitalState(None, metadata=None, coordinates='TLE')

    # Push the relevant quantities outside of their limits one at a time
    bad_out_kep = np.copy(out_kep)
    bad_out_kep[0] = 1.2
    with pytest.raises(ValueError):
        OrbitalState(bad_out_kep, coordinates='keplerian')
    bad_out_tle = np.copy(out_tle)
    bad_out_tle[2] = 1.2
    with pytest.raises(ValueError):
        OrbitalState(bad_out_tle, coordinates='TLE')
    bad_out_equ = np.copy(out_equ)
    bad_out_equ[2] = -1.5
    with pytest.raises(ValueError):
        OrbitalState(bad_out_equ, coordinates='Equinoctial')
    bad_out_equ[1] = -1.5
    with pytest.raises(ValueError):
        OrbitalState(bad_out_equ, coordinates='Equinoctial')


# The next three tests ensure that the initialisations in different forms
# yield the same results
def test_kep_cart():

    # Test that Keplerian initialisation yields same state vector
    # Firstly just flipping back and forth
    keplerian_sn = OrbitalState(cartesian_s.keplerian_elements,
                                coordinates='keplerian',
                                grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(cartesian_s.state_vector, keplerian_sn.cartesian_state_vector,
                       rtol=1e-4))

    # independent initialisation
    assert(np.allclose(keplerian_s.state_vector, orb_st_vec, rtol=2e-3))

    # Test timestamp
    assert keplerian_s.epoch == dtime


def test_tle_cart():
    tle_sn = OrbitalState(cartesian_s.two_line_element, coordinates='TLE',
                          grav_parameter=cartesian_s.grav_parameter)
    # Note that we need to convert to floats to do the comparison because np.allclose invokes the
    # np.isfinite() function which throws an error on Angle types
    assert(np.allclose(np.float64(cartesian_s.two_line_element),
                       np.float64(tle_sn.two_line_element), rtol=1e-3))


def test_equ_cart():
    equ_sn = OrbitalState(cartesian_s.equinoctial_elements, coordinates='equinoctial',
                          grav_parameter=cartesian_s.grav_parameter)
    assert(np.allclose(np.float64(cartesian_s.equinoctial_elements),
                       np.float64(equ_sn.equinoctial_elements), rtol=1e-3))


# Now we need to test that the output is actually correct.
# Test Cartesian input and Keplerian output on known equivalents
def test_cart_kep():
    # Simple assertion
    assert(np.all(cartesian_s.state_vector == orb_st_vec))
    # Check Keplerian elements come out right
    assert(np.allclose(np.float64(cartesian_s.keplerian_elements), out_kep, rtol=1e-3))


# The test TLE output
def test_cart_tle():
    assert(np.allclose(np.float64(cartesian_s.two_line_element), out_tle, rtol=1e-3))


# Test some specific quantities
def test_tle_via_metadata():
    """Initiate the orbitstate from a TLE (like you'd get from SpaceTrack).
    The TLE is an test TLE from copied verbatim with the following Cartesian
    state"""

    outstate = StateVector([-3.75652102e+06, 5.62622198e+06, 4.88985712e+05, -4.20560647e+03,
                            -2.29106828e+03, -5.98628657e+03])

    lin1 = "1 25544U 98067A   18182.57105324 +.00001714 +00000-0 +33281-4 0  9991"
    lin2 = "2 25544 051.6426 307.0095 0003698 252.8831 281.8833 15.53996196120757"

    tle_metadata = {'line_1': lin1, 'line_2': lin2}
    tle_state = OrbitalState(None, coordinates='TLE', metadata=tle_metadata)

    assert (np.allclose(tle_state.state_vector, outstate, rtol=1e-4))
