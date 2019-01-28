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
    """A simple reader for FITS files. Reads a FITS file and sets the contents
    as the data and header attributes. Uses the Astropy library for reading files.

    FITS file must be valid i.e. have at least one Header Data Unit (HDU)

    Parameters
    ----------
    Requires a String path to the file to be read
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


class TLEReader(FileReader):
    """A simple reader for text files containing a list of Two Line Elements (TLEs).

    Parameters
    ----------
    """

    
