# coding: utf-8

from ..orbit import CartesianTransitionModel
from .....types.array import CovarianceMatrix
from .....types.orbitalstate import OrbitalState, GaussianOrbitalState

def test_cartkeptrans():

        ini_cart = StateVector([[7000], [-12124], [0], [2.6679], [4.621], [0]])
        # Adjust gravitational parameter because we're in km, rather than m
        gtstate = OrbitalState(ini_cart, timestamp=start_time, coordinates="Cartesian", grav_parameter=398600)

        propagator = CartesianTransitionModel(
        process_noise=CovarianceMatrix(np.diag([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])))