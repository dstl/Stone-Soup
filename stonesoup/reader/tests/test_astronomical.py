# -*- coding: utf-8 -*-
import numpy as np
import os
import csv

from ..astronomical import FITSReader, TLEReader
from astropy.io import fits


def test_fits():
    fits_filename = "test.fits"
    n = np.arange(100.0)
    n.shape = (10, 10)
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Edwin Hubble'
    hdr['COMMENT'] = "Here's some commentary about this FITS file."
    hdu = fits.PrimaryHDU(n, header=hdr)
    hdu.writeto(fits_filename, overwrite=True)

    fits_reader = FITSReader(fits_filename)
    image_data = fits_reader.data[0]
    header = fits_reader.header[0]
    assert np.array_equal(image_data, n)
    assert header['OBSERVER'] == 'Edwin Hubble'
    assert header['COMMENT'] == "Here's some commentary about this FITS file."
    os.remove(fits_filename)


def test_tle():
    tle_filename = "tle.txt"
    line1 = "1     5U 58002B   19024.32871288 -.00000158  00000-0 -22140-3 0  9993\n"
    line2 = "2     5  34.2570 359.4940 1846904 217.0594 128.7222 10.84788078149955\n"
    with open(tle_filename, 'w') as file:
        file.writelines([line1, line2, line1, line2, line1, line2])
    tle_reader = TLEReader(tle_filename)
    tle_reader.parse_TLEs()


if __name__ == '__main__':
    test_tle()
