# -*- coding: utf-8 -*-
from math import trunc, ceil, floor

from pytest import approx, xfail
from numpy import deg2rad
import numpy as np

from ..angle import Bearing, Elevation, Latitude, Longitude
from ...functions import (mod_elevation, mod_bearing)


def pytest_generate_tests(metafunc):
    # Called on each function in a class
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append(([x[1] for x in items]))
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


scenario1 = ('bearing', {'class_': Bearing, 'func': mod_bearing})
scenario2 = ('elevation', {'class_': Elevation, 'func': mod_elevation})
scenario3 = ('longitude', {'class_': Longitude, 'func': mod_bearing})
scenario4 = ('latitude', {'class_': Latitude, 'func': mod_elevation})


class TestAngle:
    scenarios = [scenario1, scenario2, scenario3, scenario4]
    # In the following functions:
    # class_ - this is the class type we use for casting e.g. Bearing
    # func - this is the modulo function

    def test_bearing_init(self, class_, func):
        b1 = class_(3.14)
        b2 = class_(deg2rad(700))
        assert approx(b1) == func(3.14)
        assert approx(b2) == func(deg2rad(-20))
        assert b1 == b1
        assert b1 == b1
        assert b1 != b2

    def test_multiply(self, class_, func):
        b1 = class_(3.14)
        b2 = class_(2)
        bmul = float(b1)*float(b2)
        bmul2 = 2*float(b1)
        assert b1*b2 == approx(bmul)
        assert b2*b1 == approx(bmul)
        assert 2*b1 == approx(bmul2)
        assert b1*2 == approx(bmul2)

    def test_addition(self, class_, func):
        b1 = class_(3.14)
        b2 = class_(2)
        b_sum = float(b1) + float(b2)
        assert approx(func(b_sum)) == b1+b2
        assert approx(func(b_sum)) == float(b1)+b2
        assert approx(func(b_sum)) == b1+float(b2)
        assert approx(func(b_sum)) == b2+b1

    def test_subtraction(self, class_, func):
        b1 = class_(3.14)
        b2 = class_(2)
        b_diff = float(b1) - float(b2)
        assert approx(func(b_diff)) == b1-b2
        assert approx(func(b_diff)) == float(b1)-b2
        assert approx(func(b_diff)) == b1-float(b2)
        assert approx(-func(b_diff)) == b2-b1

    def test_division(self, class_, func):
        b1 = class_(3.14)
        b2 = class_(2)
        b_div = float(b1)/float(b2)
        assert approx(b_div) == b1/b2
        assert approx(float(b1)/2) == b1/2
        assert approx(b_div) == float(b1)/b2

    def test_comaparision(self, class_, func):
        b1 = class_(deg2rad(30))
        b2 = class_(deg2rad(10))
        assert b1 == b1
        assert b1 > b2
        assert b1 >= b1
        assert b1 <= b1
        assert b2 < b1

    def test_trig(self, class_, func):
        b1 = class_(deg2rad(22.0))
        b2 = float(b1)

        assert np.sin(b1) == np.sin(b2)
        assert np.cos(b1) == np.cos(b2)
        assert np.tan(b1) == np.tan(b2)
        assert np.sinh(b1) == np.sinh(b2)
        assert np.cosh(b1) == np.cosh(b2)
        assert np.tanh(b1) == np.tanh(b2)
        assert np.rad2deg(b1) == np.rad2deg(b2)

    def test_misc(self, class_, func):
        b1 = class_(deg2rad(22.0))
        b2 = float(b1)

        assert ceil(b1) == ceil(b2)
        assert floor(b1) == floor(b2)
        assert trunc(b1) == trunc(b2)
        assert round(b1, 2) == round(b2, 2)
        assert b1 % 0.1 == b2 % 0.1
        assert +b1 == +b2
        assert approx(-b1) == -b2

    def test_degrees(self, class_, func):
        b1 = class_(np.pi/4)  # pi/4 radians = 45 degrees
        assert b1.degrees == 45.0

    def test_average(self, class_, func):
        val = np.pi/4
        b1 = class_(val) - 0.1
        b2 = class_(val) + 0.1
        assert class_.average([b1, b2]) == approx(val)

        if func is mod_bearing:
            val = -np.pi
            b1 = class_(val) - 0.1
            b2 = class_(val) + 0.1
            assert class_.average([b1, b2]) == approx(val)
        else:
            raise xfail("Can't handle average when wrapping over ±pi")
