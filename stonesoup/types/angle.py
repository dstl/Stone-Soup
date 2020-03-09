# -*- coding: utf-8 -*-
import numpy as np
from numbers import Real
from numpy import float64
from math import trunc, ceil, floor
from ..functions import mod_bearing, mod_elevation, mod_longitude, mod_latitude


class Angle(Real):
    """Angle class.

    Angle handles modulo arithmetic for adding and subtracting angles
    """
    @staticmethod
    def mod_angle(value):
        return value

    def __init__(self, value):
        self._value = float64(self.mod_angle(value))

    def __add__(self, other):
        out = self._value + float64(other)
        return self.__class__(self.mod_angle(out))

    def __radd__(self, other):
        return self.__class__.__add__(self, other)

    def __sub__(self, other):
        out = self._value - float64(other)
        return self.__class__(self.mod_angle(out))

    def __rsub__(self, other):
        return self.__class__.__add__(-self, other)

    def __float__(self):
        return float(self._value)

    def __mul__(self, other):
        return self._value * float64(other)

    def __rmul__(self, other):
        return self._value * float64(other)

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, float64(self))

    def __neg__(self):
        return self.__class__(-self._value)

    def __truediv__(self, other):
        return self._value / float64(other)

    def __rtruediv__(self, other):
        return float64(other) / self._value

    def __eq__(self, other):
        return self._value == float64(other)

    def __ne__(self, other):
        return self._value != float64(other)

    def __abs__(self):
        return self.__class__(abs(self._value))

    def __le__(self, other):
        return self._value <= float64(other)

    def __lt__(self, other):
        return self._value < float64(other)

    def __ge__(self, other):
        return self._value >= float64(other)

    def __gt__(self, other):
        return self._value > float64(other)

    def __floor__(self):
        return floor(self._value)

    def __ceil__(self):
        return ceil(self._value)

    def __floordiv__(self, other):
        return self._value // other

    def __mod__(self, other):
        return self._value % other

    def __pos__(self):
        return self.__class__(+self._value)

    def __pow__(self, value):
        return pow(self._value, value)

    def __rfloordiv__(self, other):
        return other // self._value

    def __rmod__(self, other):
        return other % self._value

    def __round__(self, ndigits=None):
        return float64(round(self._value, ndigits=ndigits))

    def __rpow__(self, base):
        return NotImplemented

    def __trunc__(self):
        return trunc(self._value)

    def cos(self):
        return np.cos(self._value)

    def sin(self):
        return np.sin(self._value)

    def tan(self):
        return np.tan(self._value)

    def cosh(self):
        return np.cosh(self._value)

    def sinh(self):
        return np.sinh(self._value)

    def tanh(self):
        return np.tanh(self._value)

    def rad2deg(self):
        return np.rad2deg(self._value)


class Bearing(Angle):
    """Bearing angle class.

    Bearing handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Bearing.
    Multiplication or division produces a float object rather than Bearing.
    """
    @staticmethod
    def mod_angle(value):
        return mod_bearing(value)


class Elevation(Angle):
    """Elevation angle class.

    Elevation handles modulo arithmetic for adding and subtracting elevation
    angles. The return type for addition and subtraction is Elevation.
    Multiplication or division produces a float object rather than Elevation.
    """
    @staticmethod
    def mod_angle(value):
        return mod_elevation(value)


class Longitude(Angle):
    """Longitude angle class.

    Longitude handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Longitude.
    Multiplication or division produces a float object rather than Longitude.
    """
    @staticmethod
    def mod_angle(value):
        return mod_longitude(value)


class Latitude(Angle):
    """Latitude angle class.

    Latitude handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Latitude.
    Multiplication or division produces a float object rather than Latitude.
    """
    @staticmethod
    def mod_angle(value):
        return mod_latitude(value)
