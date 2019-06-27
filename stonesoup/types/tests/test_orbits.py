# -*- coding: utf-8 -*-
import numpy as np

from ..orbitalstate import KeplerianOrbitState, TLEOrbitState, EquinoctialOrbitState
import pytest


def test_keplerian_orbit():
    with pytest.raises(TypeError):
        KeplerianOrbitState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        kep_orbit = KeplerianOrbitState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[0.7],
                                    [10000],
                                    [90*(np.pi/180)],
                                    [60*(np.pi/180)],
                                    [14.1*(np.pi/180)],
                                    [325*(np.pi/180)]])

    kep_orbit = KeplerianOrbitState(orbital_state_vector, metadata={})
    kep_orbit = kep_orbit


def test_tle_orbit():
    with pytest.raises(TypeError):
        TLEOrbitState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        tle_orbit = TLEOrbitState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[0.7],
                                    [51.6416*(np.pi/180)],
                                    [90*(np.pi/180)],
                                    [60*(np.pi/180)],
                                    [15.7*(np.pi/180)],
                                    [325*(np.pi/180)]])

    tle_orbit = TLEOrbitState(orbital_state_vector, metadata={})
    tle_orbit = tle_orbit


def test_equinoctial_orbit():
    with pytest.raises(TypeError):
        EquinoctialOrbitState()
    with pytest.raises(ValueError):
        orbital_state_vector = np.array([[-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4],
                                        [-4]])
        equi_orbit = EquinoctialOrbitState(orbital_state_vector, metadata={})
    orbital_state_vector = np.array([[10000],
                                    [0.5],
                                    [0.5],
                                    [0.5],
                                    [0.5],
                                    [325*(np.pi/180)]])
    equi_orbit = EquinoctialOrbitState(orbital_state_vector, metadata={})
    equi_orbit = equi_orbit


if __name__ == '__main__':
    test_keplerian_orbit()
    test_tle_orbit()
    test_equinoctial_orbit()
