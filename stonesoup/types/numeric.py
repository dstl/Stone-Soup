# -*- coding: utf-8 -*-
from math import log, log1p, exp, trunc, ceil, floor
from numbers import Real, Integral


class Probability(Real):
    """Probability class.

    Similar to a float, but value stored as natural log value internally.
    All operations are attempted with log values where possible, and failing
    that a float will be returned instead.
    """

    def __init__(self, value, *, log_value=False):
        if log_value:
            self._log_value = value
        else:
            try:
                self._log_value = self._log(value)
            except ValueError:
                raise ValueError("value must be greater than 0")

    @property
    def log_value(self):
        return self._log_value

    @staticmethod
    def _log(other):
        if isinstance(other, Probability):
            return other.log_value
        elif other == 0:
            return float("-inf")
        else:
            return log(other)

    def __hash__(self):
        value = float(self)
        if value == 0 and self.log_value != float("-inf"):  # Too close to zero
            # Add string so doesn't have same hash as the log value itself.
            return hash(('log', self._log_value))
        else:
            return hash(value)

    def __eq__(self, other):
        if other < 0:
            return False
        return self.log_value == self._log(other)

    def __le__(self, other):
        if other < 0:
            return False
        return self.log_value <= self._log(other)

    def __lt__(self, other):
        if other < 0:
            return False
        return self.log_value < self._log(other)

    def __ge__(self, other):
        if other < 0:
            return True
        return self.log_value >= self._log(other)

    def __gt__(self, other):
        if other < 0:
            return True
        return self.log_value > self._log(other)

    def __add__(self, other):
        if other < 0:
            return self - -other

        log_other = self._log(other)
        if self.log_value > log_other:
            log_l, log_s = self.log_value, log_other
        elif self.log_value < log_other:
            log_l, log_s = log_other, self.log_value
        else:  # Must be equal, so just double value
            return self * 2

        if log_s == float("-inf"):  # Just return largest value
            return Probability(log_l, log_value=True)

        return Probability(log_l + log1p(exp(log_s - log_l)),
                           log_value=True)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if other < 0:
            return self + -other

        log_other = self._log(other)
        if self.log_value > log_other:
            log_l, log_s = self.log_value, log_other
        elif self.log_value < log_other:  # Result will be negative
            return float(self) - other
        else:  # Must be equal, so return 0
            return Probability(float("-inf"), log_value=True)

        if log_s == float("-inf"):  # Just return largest value
            return Probability(log_l, log_value=True)

        exp_diff = exp(log_s - log_l)
        if exp_diff == 1:  # Diff too small, so result is effectively zero
            return Probability(float("-inf"), log_value=True)

        return Probability(log_l + log1p(-exp_diff),
                           log_value=True)

    def __rsub__(self, other):
        if other < 0:  # Result will be negative
            return other + -float(self)

        log_other = self._log(other)
        if log_other > self.log_value:
            log_l, log_s = log_other, self.log_value
        elif log_other < self.log_value:  # Result will be negative
            return other + -float(self)
        else:  # Must be equal, so return 0
            return Probability(float("-inf"), log_value=True)

        if log_s == float("-inf"):  # Just return largest value
            return Probability(log_l, log_value=True)

        exp_diff = exp(log_s - log_l)
        if exp_diff == 1:  # Diff too small, so result is effectively zero
            return Probability(float("-inf"), log_value=True)

        return Probability(log_l + log1p(-exp_diff),
                           log_value=True)

    def __mul__(self, other):
        try:
            return Probability(self.log_value + self._log(other),
                               log_value=True)
        except ValueError:
            return float(self) * other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            return Probability(self.log_value - self._log(other),
                               log_value=True)
        except ValueError:
            return float(self) / other

    def __rtruediv__(self, other):
        try:
            return Probability(self._log(other) - self.log_value,
                               log_value=True)
        except ValueError:
            return other / float(self)

    def __floordiv__(self, other):
        return floor(self / other)

    def __rfloordiv__(self, other):
        return floor(other / self)

    def __mod__(self, other):
        try:
            return Probability(float(self) % other)
        except ValueError:
            return float(self) % other

    def __rmod__(self, other):
        return Probability(other % float(self))

    def __pow__(self, exponent):
        if isinstance(exponent, Probability):
            exponent = float(exponent)
        return Probability(exponent * self.log_value, log_value=True)

    def __rpow__(self, base):
        return Probability(base ** float(self))

    def __neg__(self):
        return -float(self)

    def __pos__(self):
        return Probability(self)

    def __abs__(self):
        return Probability(self)

    def __float__(self):
        return exp(self.log_value)

    def __round__(self, ndigits=None):
        value = round(float(self), ndigits)
        if isinstance(value, Integral):
            return value
        else:
            return Probability(value)

    def __trunc__(self):
        return trunc(float(self))

    def __floor__(self):
        return floor(float(self))

    def __ceil__(self):
        return ceil(float(self))

    def __repr__(self):
        value = float(self)
        if value == 0 and self.log_value != float("-inf"):  # Too close to zero
            return "Probability({!r}, log_value=True)".format(self.log_value)
        else:
            return "Probability({!r})".format(float(self))

    def __str__(self):
        value = float(self)
        if value == 0 and self.log_value != float("-inf"):  # Too close to zero
            return "exp({})".format(self.log_value)
        else:
            return str(value)

    def sqrt(self):
        """Square root which can be called by NumPy"""
        return self ** 0.5

    def log(self):
        """Log which can be called by NumPy"""
        return self.log_value

    @classmethod
    def sum(cls, values):
        """Carry out LogSumExp"""
        log_values = [cls._log(value) for value in values]
        if not log_values:
            return Probability(0)
        max_log_value = max(log_values)
        value_sum = sum(exp(log_value - max_log_value)
                        for log_value in log_values)

        return Probability(cls._log(value_sum) + max_log_value, log_value=True)
