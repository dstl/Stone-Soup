# -*- coding: utf-8 -*-
import numpy as np

from ..orbitalstate import KeplerianOrbitalState, TLEOrbitalState, EquinoctialOrbitalState
import pytest


def test_keplerian_orbit():
    with pytest.raises(TypeError):
        KeplerianOrbitalState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        kep_orbit = KeplerianOrbitalState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[0.7],
                                    [10000],
                                    [90*(np.pi/180)],
                                    [60*(np.pi/180)],
                                    [14.1*(np.pi/180)],
                                    [325*(np.pi/180)]])

    kep_orbit = KeplerianOrbitalState(orbital_state_vector, metadata={})
    kep_orbit = kep_orbit


def test_tle_orbit():
    with pytest.raises(TypeError):
        TLEOrbitalState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-0.4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        tle_orbit = TLEOrbitalState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[51.6416*(np.pi/180)],
                                    [90*(np.pi/180)],
                                    [0.7],
                                    [60*(np.pi/180)],
                                    [325*(np.pi/180)],
                                    [0.157*(np.pi/180)]])
    tle_orbit = TLEOrbitalState(orbital_state_vector, metadata={})
    tle_orbit = tle_orbit


def test_equinoctial_orbit():
    with pytest.raises(TypeError):
        EquinoctialOrbitalState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        equi_orbit = EquinoctialOrbitalState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[10000],
                                    [0.5],
                                    [0.5],
                                    [0.5],
                                    [0.5],
                                    [325*(np.pi/180)]])
    equi_orbit = EquinoctialOrbitalState(orbital_state_vector, metadata={})
    equi_orbit = equi_orbit


if __name__ == '__main__':
    test_keplerian_orbit()
    test_tle_orbit()
    test_equinoctial_orbit()
