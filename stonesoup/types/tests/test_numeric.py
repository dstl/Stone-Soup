# -*- coding: utf-8 -*-
from math import log, floor, ceil, trunc, sqrt

import pytest
from pytest import approx

from ..numeric import Probability


def test_probability_init():
    probability = Probability(0.2)

    assert probability == 0.2
    assert probability.log_value == log(0.2)

    Probability(log(0.2), log_value=True)

    assert probability == 0.2
    assert probability.log_value == log(0.2)

    with pytest.raises(ValueError, match="value must be greater than 0"):
        Probability(-0.2)


def test_probability_string():
    probability1 = Probability(0.2)
    probability2 = Probability(-1000, log_value=True)
    assert "Probability(0.2)" == repr(probability1)
    assert "Probability(-1000, log_value=True)" == repr(probability2)

    assert "0.2" == str(probability1)
    assert "exp(-1000)" == str(probability2)


def test_probability_comparison():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert probability1 == probability1
    assert not probability1 == -1
    assert probability1 != probability2
    assert probability1 != -1

    assert probability1 < probability2
    assert not probability1 < -1
    assert probability1 <= probability1
    assert probability1 <= probability2
    assert not probability1 <= -1

    assert probability2 > probability1
    assert probability2 > -1
    assert probability2 >= probability1
    assert probability2 >= probability2
    assert probability2 >= -1


def test_probability_addition():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.5) == probability1 + probability2
    assert approx(0.5) == probability2 + probability1
    assert approx(0.5) == probability2 + 0.2
    assert approx(0.5) == 0.3 + probability1
    assert approx(0.4) == probability1 + probability1
    assert approx(0.1) == -0.2 + probability2
    assert approx(0.1) == probability2 + -0.2
    assert approx(0.2) == probability1 + 0
    assert approx(0.2) == 0 + probability1


def test_probability_subtraction():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.1) == probability2 - probability1
    assert approx(0.1) == probability2 - 0.2
    assert approx(0.1) == 0.3 - probability1

    assert approx(0) == probability1 - probability1
    assert approx(0) == 0.3 - probability2
    assert approx(0.2) == probability1 - 0
    assert approx(0.5) == probability2 - -0.2

    assert approx(-0.3) == probability1 - 0.5
    assert approx(-0.1) == 0.2 - probability2
    assert approx(-0.2) == 0 - probability1
    assert approx(-0.5) == -0.3 - probability1

    assert approx(0.2) == 0.2 - Probability(0)


def test_probability_multiply():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.06) == probability1 * probability2
    assert (probability1 * probability2).log_value == log(0.2) + log(0.3)

    assert approx(0.4) == probability1 * 2
    assert approx(0.6) == 2 * probability2

    assert approx(-0.4) == probability1 * -2
    assert approx(-0.6) == -2 * probability2


def test_probability_divide():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(2/3) == probability1 / probability2
    assert (probability1 / probability2).log_value == log(0.2) - log(0.3)

    assert approx(2/3) == probability1 / 0.3
    assert approx(2/3) == 0.2 / probability2

    assert approx(-2/3) == probability1 / -0.3
    assert approx(-2/3) == -0.2 / probability2

    assert approx(0) == probability1 // probability2
    assert approx(0) == 0.2 // probability2
    assert approx(0) == probability1 // 0.3
    assert approx(-1) == -0.2 // probability2
    assert approx(-1) == probability1 // -0.3


def test_probability_mod():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.1) == probability2 % probability1
    assert approx(0.1) == probability2 % 0.2
    assert approx(0.1) == 0.3 % probability1

    assert approx(-0.1) == probability2 % -0.2
    assert approx(0.1) == -0.3 % probability1


def test_probability_power():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.2**0.3) == probability1 ** probability2
    assert approx(0.2**0.3) == 0.2 ** probability2
    assert approx(0.2**0.3) == probability1 ** 0.3

    assert approx(-0.2**0.3) == -0.2 ** probability2
    assert approx(0.2**-0.3) == probability1 ** -0.3


def test_probability_integral():
    probability = Probability(0.55)

    assert 0 == floor(probability)
    assert 0 == trunc(probability)
    assert 1 == ceil(probability)
    assert 1 == round(probability)
    assert approx(0.6) == round(probability, 1)


def test_probability_sign():
    probability = Probability(0.5)

    assert approx(0.5) == abs(probability)
    assert approx(0.5) == +probability
    assert approx(-0.5) == -probability


def test_probability_sum():
    probability1 = Probability(0.2)
    probability2 = Probability(0.3)

    assert approx(0.5) == Probability.sum((probability1, probability2))
    assert approx(0.5) == Probability.sum((0.2, probability2))
    assert approx(0.5) == Probability.sum((probability1, 0.3))
    assert approx(0.5) == Probability.sum((probability1, probability2, 0))


def test_probability_numpy_methods():
    probability = Probability(0.2)

    assert approx(sqrt(0.2)) == probability.sqrt()
    assert approx(log(0.2)) == probability.log()
