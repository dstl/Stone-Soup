# -*- coding: utf-8 -*-
from numbers import Real
from numpy import float64
from math import trunc, ceil, floor

import numpy as np

from ..functions import mod_bearing, mod_elevation
from ..functions.orbital import mod_inclination, mod_elongitude


class Angle(Real):
    """Angle class.

    Angle handles modulo arithmetic for adding and subtracting angles
    """
    @staticmethod
    def mod_angle(value):
        return value

    @property
    def degrees(self):
        return self.rad2deg()

    def __init__(self, value):
        self._value = float64(self.mod_angle(value))

    def __add__(self, other):
        out = self._value + other
        return self.__class__(self.mod_angle(out))

    def __radd__(self, other):
        return self.__class__.__add__(self, other)

    def __sub__(self, other):
        out = self._value - other
        return self.__class__(self.mod_angle(out))

    def __rsub__(self, other):
        return self.__class__.__add__(-self, other)

    def __float__(self):
        return float(self._value)

    def __mul__(self, other):
        return self._value * other

    def __rmul__(self, other):
        return self._value * other

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, float64(self))

    def __neg__(self):
        return self.__class__(-self._value)

    def __truediv__(self, other):
        return self._value / other

    def __rtruediv__(self, other):
        return other / self._value

    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __abs__(self):
        return self.__class__(abs(self._value))

    def __le__(self, other):
        return self._value <= other

    def __lt__(self, other):
        return self._value < other

    def __ge__(self, other):
        return self._value >= other

    def __gt__(self, other):
        return self._value > other

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

    @classmethod
    def average(cls, angles, weights=None):
        """Calculated the circular mean for sequence of angles

        Parameters
        ----------
        angles : sequence of :class:`~.Angle`
            Angles which to calculate the mean of.
        weights : sequence of float, optional
            Weights to calculate weighted mean. Default `None`, where no weights applied.

        Returns
        -------
        : :class:`Angle`
            Circular mean of angles
        """
        if weights is None:
            weight_sum = 1
            weights = 1
        else:
            weight_sum = np.sum(weights)

        result = np.arctan2(
            float(np.sum(np.sin(angles) * weights) / weight_sum),
            float(np.sum(np.cos(angles) * weights) / weight_sum))

        return cls(result)


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


class Longitude(Bearing):
    """Longitude angle class.

    Longitude handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Longitude.
    Multiplication or division produces a float object rather than Longitude.
    """


class Latitude(Elevation):
    """Latitude angle class.

    Latitude handles modulo arithmetic for adding and subtracting angles. \
    The return type for addition and subtraction is Latitude.
    Multiplication or division produces a float object rather than Latitude.
    """


class Inclination(Angle):
    """(Orbital) Inclination angle class.

    Inclination handles modulo arithmetic for adding and subtracting angles.
    The return type for addition and subtraction is Inclination.
    Multiplication or division produces a float object rather than Inclination.
    """
    @staticmethod
    def mod_angle(value):
        return mod_inclination(value)


class EclipticLongitude(Angle):
    """(Orbital) Ecliptic Longitude angle class.

    Ecliptic Longitude handles modulo arithmetic for adding and subtracting angles.
    The return type for addition and subtraction is Ecliptic Longitude.
    Multiplication or division produces a float object rather than Ecliptic Longitude.
    """
    @staticmethod
    def mod_angle(value):
        return mod_elongitude(value)
