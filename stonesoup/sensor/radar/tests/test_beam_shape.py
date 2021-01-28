# -*- coding: utf-8 -*-
from pytest import approx, raises

from ..beam_shape import BeamShape, Beam2DGaussian


def test_abstract_beam_shape():
    with raises(TypeError):
        BeamShape()


def test_beam_shape():
    two_dim_gaus = Beam2DGaussian(peak_power=100, beam_width=10)
    assert approx(two_dim_gaus.beam_power(5, 5), 3) == 25
    assert approx(two_dim_gaus.beam_power(0, 0), 3) == 100
