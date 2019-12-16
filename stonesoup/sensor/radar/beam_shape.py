# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property, Base



class BeamShape(Base):
    """Base class for beam shape"""
    peak_power = Property(float, doc="peak power of the main lobe in Watts")

    def beam_power(self, azimuth, elevation, **kwargs):
        """beam power sent in the direction of the target.
        azimuth = elevation = 0 for center of beam"""
        raise NotImplementedError


class Beam2DGaussian(BeamShape):
    """The beam is in the shape of a 2D gaussian in the azimuth and elevation.
     The width at half the maxima is the beam width
     .. math::
        P_p = \exp \left( 0.5 * \left(\left(\frac{az * 2.35}{B_w}\right)^2 +
        \left(\frac{el * 2.35}{B_w}\right)^2 \right) \right)
     where :math:'az' and :math:'el' are the azimuth and elevation away from
     the centre. and :math:'B_w' is the beam width.
     """
    beam_width = Property(float, default=None,
                          doc='Width of the radar beam')

    def beam_power(self, azimuth, elevation, **kwargs):
        return self.peak_power * np.exp(
            -0.5 * ((azimuth / (self.beam_width / 2.35482)) ** 2 +
                    (elevation / (self.beam_width / 2.35482)) ** 2))
