# -*- coding: utf-8 -*-

from pytest import approx
from numpy import deg2rad
from ...functions import mod_elevation, mod_bearing

from ..angle import Bearing, Elevation


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


scenario1 = ('bearing', {'Class': Bearing, 'func': mod_bearing})
scenario2 = ('elevation', {'Class': Elevation, 'func': mod_elevation})


class TestAngle:
    scenarios = [scenario1, scenario2]
    # In the following functions:
    # Class - this is the class type we use for casting e.g. Bearing
    # func - this is the modulo function

    def test_bearing_init(self, Class, func):
        b1 = Class(3.14)
        b2 = Class(deg2rad(700))
        assert approx(b1) == func(3.14)
        assert approx(b2) == func(deg2rad(-20))
        assert b1 == b1
        assert b1 == b1
        assert b1 != b2

    def test_multiply(self, Class, func):
        b1 = Class(3.14)
        b2 = Class(2)
        bmul = float(b1)*float(b2)
        bmul2 = 2*float(b1)
        assert b1*b2 == approx(bmul)
        assert b2*b1 == approx(bmul)
        assert 2*b1 == approx(bmul2)
        assert b1*2 == approx(bmul2)

    def test_addition(self, Class, func):
        b1 = Class(3.14)
        b2 = Class(2)
        b_sum = float(b1) + float(b2)
        assert approx(func(b_sum)) == b1+b2
        assert approx(func(b_sum)) == float(b1)+b2
        assert approx(func(b_sum)) == b1+float(b2)
        assert approx(func(b_sum)) == b2+b1

    def test_subtraction(self, Class, func):
        b1 = Class(3.14)
        b2 = Class(2)
        b_diff = float(b1) - float(b2)
        assert approx(func(b_diff)) == b1-b2
        assert approx(func(b_diff)) == float(b1)-b2
        assert approx(func(b_diff)) == b1-float(b2)
        assert approx(-func(b_diff)) == b2-b1

    def test_division(self, Class, func):
        b1 = Class(3.14)
        b2 = Class(2)
        b_div = float(b1)/float(b2)
        assert approx(b_div) == b1/b2
        assert approx(float(b1)/2) == b1/2
        assert approx(b_div) == float(b1)/b2

    def test_comaparision(self, Class, func):
        b1 = Class(deg2rad(30))
        b2 = Class(deg2rad(10))
        assert b1 == b1
        assert b1 > b2
        assert b1 >= b1
        assert b1 <= b1
        assert b2 < b1
