# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np

from ...base import Property, Base


class BeamShape(Base):
    """Base class for beam shape"""
    peak_power: float = Property(doc="peak power of the main lobe in Watts")

    @abstractmethod
    def beam_power(self, azimuth, elevation, beam_width, **kwargs):
        """beam power sent in the direction of the target.
        azimuth = elevation = 0 for center of beam"""
        raise NotImplementedError


class Beam2DGaussian(BeamShape):
    r"""The beam is in the shape of a 2D gaussian in the azimuth and elevation.
     The width at half the maxima is the beam width. It is described by:

     .. math::

        P = P_p\exp \left( 0.5 \times \left(\left(\frac{2.35\,az}{B_w}\right)
        ^2 +\left(\frac{2.35\,el}{B_w}\right)^2 \right) \right)

     where :math:`az` and :math:`el` are the azimuth and elevation angles away
     from the centre. :math:`B_w` is the beam width and :math:`P_p` is the peak
     power.
     """

    def beam_power(self, azimuth, elevation, beam_width, **kwargs):
        """
        Parameters
        ----------
        azimuth : `float`
            The angle of the target away from the boresight of the radar in
            azimuth
        elevation : `float`
            The angle of the target away from the boresight of the radar in
            elevation
        beam_width: `float`
            The width of the radar beam

        Returns
        -------
        `float`
            the power directed towards the target
        """
        return self.peak_power * np.exp(
            -0.5 * ((azimuth / (beam_width / 2.35482)) ** 2 +
                    (elevation / (beam_width / 2.35482)) ** 2))
