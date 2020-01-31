# -*- coding: utf-8 -*-
from ..beam_shape import Beam2DGaussian
from pytest import approx


def test_beam_shape():
    two_dim_gaus = Beam2DGaussian(peak_power=100, beam_width=10)
    assert approx(two_dim_gaus.beam_power(5, 5), 3) == 25
    assert approx(two_dim_gaus.beam_power(0, 0), 3) == 100
