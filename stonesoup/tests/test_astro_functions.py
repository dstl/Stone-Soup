import numpy as np
from stonesoup.astro_functions import convert_orbit

from stonesoup.types.orbitalstate import KeplerianOrbitState


def test_tle_orbit_conversion():
    orbital_state_vector = np.array([[	0.0005579],
                                    [7000],
                                    [51.6398*(np.pi/180)],
                                    [261*(np.pi/180)],
                                    [15.7*(np.pi/180)],
                                    [90*(np.pi/180)]])

    kep_orbit = KeplerianOrbitState(orbital_state_vector, metadata={})
    tle_orbit = convert_orbit(kep_orbit, 'tle')
    new_kep_orbit = convert_orbit(tle_orbit, 'keplerian')
    assert np.allclose(kep_orbit.state_vector, new_kep_orbit.state_vector)


def test_equi_orbit_conversion():
    orbital_state_vector = np.array([[	0.0005579],
                                    [7000],
                                    [51.6398*(np.pi/180)],
                                    [300*(np.pi/180)],
                                    [15.7*(np.pi/180)],
                                    [90*(np.pi/180)]])
    print(orbital_state_vector)
    kep_orbit = KeplerianOrbitState(orbital_state_vector, metadata={})
    equi_orbit = convert_orbit(kep_orbit, 'equinoctial')
    new_kep_orbit = convert_orbit(equi_orbit, 'keplerian')

    print(new_kep_orbit.state_vector)
    assert np.allclose(kep_orbit.state_vector, new_kep_orbit.state_vector, atol=1e-2)


if __name__ == '__main__':
    test_tle_orbit_conversion()
    test_equi_orbit_conversion()
