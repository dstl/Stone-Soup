# -*- coding: utf-8 -*-
from textwrap import dedent
from datetime import datetime

import pytest
import numpy as np

from ..astronomical import FITSReader, TLEFileReader
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

@pytest.fixture()
def tle_gt_filename(tmpdir):
    tle_filename = tmpdir.join("test_tle.txt")
    with tle_filename.open('w') as tle_file:
        tle_file.write(dedent("""\
                1 25544U 98067A   18182.57105324 +.00001714 +00000-0 +33281-4 0  9991
                2 25544 051.6426 307.0095 0003698 252.8831 281.8833 15.53996196120757
                """))
    return tle_filename

def test_tle_reader(tle_gt_filename):
    # run test with:
    #   - 2d co-ordinates
    #   - default time field format
    #   - no special csv options
    tle_reader = TLEFileReader(tle_gt_filename.strpath)

    assert tle_reader.catalogue_number == 25544
    assert tle_reader.classification == "U"
    assert tle_reader.international_designator == "98067A  "
    assert tle_reader.epoch == datetime(2018, 7, 1, 13, 42, 18, 999936)
    assert np.isclose(tle_reader.ballistic_coefficient,0.00001714 * 2 * (2*np.pi) / (86400**2))
    assert tle_reader.second_derivative_mean_motion == 0.0
    assert tle_reader.bstar == 0.33281e-4 / 6.371e6
    assert tle_reader.ephemeris_type == 0
    assert tle_reader.element_set_number == 999
    assert tle_reader.inclination == 51.6426 * np.pi/180
    assert tle_reader.longitude_of_ascending_node == 307.0095 * np.pi/180
    assert tle_reader.eccentricity == 0.0003698
    assert np.isclose(tle_reader.arg_periapsis, 252.8831 * np.pi/180)
    assert tle_reader.mean_anomaly == 281.8833 * np.pi/180
    assert tle_reader.mean_motion == 15.53996196 * 2 * np.pi / 86400
    assert tle_reader.revolution_number == 12075
    assert tle_reader.checksum_declared == (1, 7)
    assert tle_reader.checksum_declared == tle_reader.checksum_calculated
