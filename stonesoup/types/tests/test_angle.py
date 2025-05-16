from math import trunc, ceil, floor

from pytest import approx, fixture, xfail
from numpy import deg2rad
import numpy as np

from stonesoup.types.array import StateVector
from ..angle import Bearing, Elevation, Latitude, Longitude
from ...functions import (mod_elevation, mod_bearing)


@fixture(params=[Bearing, Elevation, Longitude, Latitude])
def class_(request):
    return request.param


@fixture()
def func(class_):
    return mod_bearing if issubclass(class_, Bearing) else mod_elevation


def test_bearing_init(class_, func):
    b1 = class_(3.14)
    b2 = class_(deg2rad(700))
    assert approx(b1) == func(3.14)
    assert approx(b2) == func(deg2rad(-20))
    assert b1 == b1
    assert b2 == b2
    assert b1 != b2


def test_multiply(class_):
    b1 = class_(3.14)
    b2 = class_(2)
    bmul = float(b1)*float(b2)
    bmul2 = 2*float(b1)
    assert b1*b2 == approx(bmul)
    assert b2*b1 == approx(bmul)
    assert 2*b1 == approx(bmul2)
    assert b1*2 == approx(bmul2)


def test_addition(class_, func):
    b1 = class_(3.14)
    b2 = class_(2)
    b_sum = float(b1) + float(b2)
    assert approx(func(b_sum)) == b1+b2
    assert approx(func(b_sum)) == float(b1)+b2
    assert approx(func(b_sum)) == b1+float(b2)
    assert approx(func(b_sum)) == b2+b1


def test_subtraction(class_, func):
    b1 = class_(3.14)
    b2 = class_(2)
    b_diff = float(b1) - float(b2)
    assert approx(func(b_diff)) == b1-b2
    assert approx(func(b_diff)) == float(b1)-b2
    assert approx(func(b_diff)) == b1-float(b2)
    assert approx(-func(b_diff)) == b2-b1


def test_division(class_):
    b1 = class_(3.14)
    b2 = class_(2)
    b_div = float(b1)/float(b2)
    assert approx(b_div) == b1/b2
    assert approx(float(b1)/2) == b1/2
    assert approx(b_div) == float(b1)/b2


def test_comparison(class_):
    b1 = class_(deg2rad(30))
    b2 = class_(deg2rad(10))
    assert b1 == b1
    assert b1 > b2
    assert b1 >= b1
    assert b1 <= b1
    assert b2 < b1


def test_trig(class_):
    b1 = class_(deg2rad(22.0))
    b2 = float(b1)

    assert np.sin(b1) == np.sin(b2)
    assert np.cos(b1) == np.cos(b2)
    assert np.tan(b1) == np.tan(b2)
    assert np.sinh(b1) == np.sinh(b2)
    assert np.cosh(b1) == np.cosh(b2)
    assert np.tanh(b1) == np.tanh(b2)
    assert np.rad2deg(b1) == np.rad2deg(b2)


def test_misc(class_):
    b1 = class_(deg2rad(22.0))
    b2 = float(b1)

    assert ceil(b1) == ceil(b2)
    assert floor(b1) == floor(b2)
    assert trunc(b1) == trunc(b2)
    assert round(b1, 2) == round(b2, 2)
    assert b1 % 0.1 == b2 % 0.1
    assert +b1 == +b2
    assert approx(-b1) == -b2


def test_degrees(class_):
    b1 = class_(np.pi/4)  # pi/4 radians = 45 degrees
    assert b1.degrees == 45.0


def test_average(class_, func):
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


def test_wrapping_equality_and_abs(class_):
    val = class_(np.pi)
    if class_ in (Bearing, Longitude):
        wrapped_val = -np.pi
    elif class_ in (Elevation, Latitude):
        wrapped_val = 0
    else:
        raise NotImplementedError
    assert val == class_(np.pi)
    assert abs(val) >= 0
    assert abs(val) == abs(class_(np.pi))
    assert val == class_(wrapped_val)
    assert abs(val) == abs(class_(wrapped_val))

    # Must use a bearing in a StateVector below: the StateVector class overrides ufuncs,
    # such that isclose works. Raw Angle classes don't (and can't sensibly) override that
    # behaviour. isclose fails if abs does not return positive values, which is why the test is
    # here
    sv = StateVector([val])
    wrapped_sv = StateVector([wrapped_val])
    assert np.isclose(sv, sv)
    assert np.isclose(sv, wrapped_sv)

    assert np.isclose(abs(sv), abs(sv))
    assert np.isclose(abs(sv), abs(wrapped_sv))


def test_hash(class_):
    _ = {class_(x) for x in range(6)}
