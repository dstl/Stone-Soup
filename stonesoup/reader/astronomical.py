# -*- coding: utf-8 -*-
"""Astronomical readers for Stone Soup.

This is a collection of readers for Stone Soup, allowing quick reading
of data that is in common astronomical formats.

Readers include:
    FITS
    TLE
    SATCAT
"""

from astropy.io import fits

from ..base import Property
from .file import FileReader


class FITSReader(FileReader):
    """A simple reader for FITS files. Returns a list of
    Header Data Units (HDUs) contained within the file

    FITS file must be valid i.e. have at least one Header Data Unit (HDU)

    Parameters
    ----------
    """

    hdu_list = Property(
        fits.HDUList, doc='List of Header Data Units (HDUs) contained in the FITS file')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with fits.open(self.path) as hdu_list:
            self._hdu_list = hdu_list

    @property
    def hdu_list(self):
        return self._hdu_list.copy()
