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
from .file import FileReader


class FITSReader(FileReader):
    """A simple reader for FITS files. Returns a list of
    Header Data Units (HDUs) contained within the file

    FITS file must be valid i.e. have at least one Header Data Unit (HDU)

    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with fits.open(self.path) as hdu_list:
            # self._hdu_list = hdu_list
            # self._hdu_list = hdu_list.copy()
            self._data = []
            self._header = []
            for index, hdu in enumerate(hdu_list):
                self._data.append(hdu_list[index].data)
                self._header.append(hdu_list[index].header)

    @property
    def data(self):
        return self._data

    @property
    def header(self):
        return self._header
