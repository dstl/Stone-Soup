# -*- coding: utf-8 -*-
from ..beam_shape import Beam2DGaussian


def test_beam_shape():
    two_dim_gaus = Beam2DGaussian(peak_power=100, beam_width=10)
    assert round(two_dim_gaus.beam_power(5, 5), 3) == 25
