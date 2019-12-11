# -*- coding: utf-8 -*-
import numpy as np

from ..astronomical import FITSReader
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


if __name__ == '__main__':
    test_fits()
