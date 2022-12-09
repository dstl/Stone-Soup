# -*- coding: utf-8 -*-
"""Providing some basic astronomical readers for Stone Soup, allowing import of data that is in
common astronomical formats.

"""
from datetime import datetime
import numpy as np
from astropy.io import fits

from ..base import Property
from .base import Reader
from .file import FileReader, TextFileReader


class FITSReader(FileReader):
    """A simple reader for FITS files. Returns a list of Header Data Units (HDUs) contained within
    the file.

    FITS file must be valid i.e. have at least one Header Data Unit (HDU)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with fits.open(self.path) as hdu_list:
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


class TLEDictReader(Reader):
    """A reader designed to accept Two-Line Element (TLE) inputs as a dictionary. These contain
    a single TLE without a Line 0 which should conform strictly to the TLE format. The key for
    Line 1 is "line_1" and that for Line 2 is "line_2". See the references [1]_, [2]_ for a full
    explanation of TLEs.

    References
    ----------
    .. [1] Kelso, T.S. 2019, CelesTrak: NORAD Two-Line Element Set Format,
       [CelesTrak](https://www.celestrak.com/NORAD/documentation/tle-fmt.php)

    .. [2] Kelso, T.S. 2019, Frequently Asked Questions: Two-Line Element Set Format,
       [CelesTrak](https://celestrak.com/columns/v04n03/)

    """
    tle: dict = Property(doc="")

    # Unit conversions
    _rad_deg = np.pi / 180
    _rad_rev = 2 * np.pi
    _day_s = 1.0 / 86400

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line1 = self.tle['line_1']
        self.line2 = self.tle['line_2']

    @classmethod
    def checksum(cls, line):
        """This class method just calculates the checksum based on the lines (not including the
        checksum itself). It's entirely due to
        https://space.stackexchange.com/users/4709/coastrogeek at
        https://space.stackexchange.com/questions/5358/what-does-check-sum-tle-mean
        """
        L = line[0:68].strip()
        cksum = 0
        for c in L:
            if c == ' ' or c == '.' or c == '+' or c.isalpha():
                continue
            elif c == '-':
                cksum = cksum + 1
            else:
                cksum = cksum + int(c)

        cksum %= 10

        return cksum

    @property
    def catalogue_number(self):
        """NORAD Catalog Number: a unique identifier for each earth-orbiting artificial satellite
        """
        return int(self.line1[2:7])

    @property
    def classification(self):
        """Classification (U=Unclassified, C=Classified, S=Secret)
        """
        return self.line1[7]

    @property
    def international_designator(self):
        """International designator incorporates the year of launch, launch number that year and
        place of launch. How to interpret this string can be found at [2]_"""
        return self.line1[9:17]

    @property
    def epoch(self):
        """The time at which the TLE is valid. Returned as a datetime object.
        """
        # Resolve the timestamp
        halfcentury = int(self.line1[17:20])
        year = 1900 + halfcentury if halfcentury > 56 else 2000 + halfcentury

        day = self.line1[20:23]

        hour = float(self.line1[23:32]) * 24
        fhour = int(np.floor(hour))

        minu = (hour - fhour) * 60
        fminu = int(np.floor(minu))

        seco = (minu - fminu) * 60
        fseco = int(np.floor(seco))

        mics = (seco - fseco) * 1e6
        fmics = int(np.round(mics))

        return datetime.strptime(str(year) + " " + str(day) + " " + str(fhour) + " " +
                                 str(fminu) + " " + str(fseco) + " " +
                                 f"{fmics:06.0f}", "%Y %j %H %M %S %f")

    @property
    def ballistic_coefficient(self):
        r"""Represents the first derivative of the mean motion , otherwise known as the ballistic
        coefficient. This is encoded in the TLE divided by two and in units of revolutions per
        day:math:`^2`. Here it is returned in units of :math:`mathrm{rad s}^{-2}`. It is unused in
        SGP4.
        """

        if self.line1[33] == '-':
            return 0 - float(self.line1[34:43]) * 2 * self._rad_rev * self._day_s**2
        else:
            return float(self.line1[34:43]) * 2 * self._rad_rev * self._day_s**2

    @property
    def second_derivative_mean_motion(self):
        """This is the second derivative of the mean motion. Again, it's not used by SGP4. In TLEs
        it's divided by six and given in units of revolutions per day:math:`^3`. Here it's returned
        as :math:`mathrm{rad s}^{-3}`"""

        mantissa = 0 - float(self.line1[45:50]) / 1e5 if self.line1[44] == '-' \
            else float(self.line1[45:50]) / 1e5

        exponent = 0 - int(self.line1[51]) if self.line1[50] == '-' else int(self.line1[51])

        return (mantissa * 10 ** (exponent)) * 6 * self._rad_rev * self._day_s**3

    @property
    def bstar(self):
        r"""The TLE drag coefficient. It's related to the ballistic coefficient,

        .. math:

            B = \frac{C_D A}{m}

        where :math:`C_D` is the coefficient of drag, :math:`A` is the cross-sectional area and
        :math:`m` is the mass. :math:`B*` is an adjusted value,

        .. math:

            B* = \frac{B \rho_0}{2}.

        In TLEs, :math:`B*` has units of (earth radii):math:^{-1}. This function returns
        :math:`\mathrm{m}^{-1}`
        """

        if self.line1[53] == "-":
            mantissa = 0 - float(self.line1[54:59]) / 1e5
        else:
            mantissa = float(self.line1[54:59]) / 1e5

        if self.line1[59] == "-":
            exponent = 0 - int(self.line1[60])
        else:
            exponent = int(self.line1[60])

        return mantissa * 10 ** (exponent) / 6.371e6

    @property
    def ephemeris_type(self):
        """Ephemeris type (NORAD use). Zero in distributed TLE data.
        """
        return int(self.line1[62])

    @property
    def element_set_number(self):
        """Element set number. Incremented when a new TLE is generated for this object.
        """
        return int(self.line1[64:68])

    @property
    def inclination(self):
        """The inclination (radians)"""
        return float(self.line2[8:16]) * self._rad_deg

    @property
    def longitude_of_ascending_node(self):
        """Longitude of the ascending node (radians)"""
        return float(self.line2[17:25]) * self._rad_deg

    @property
    def eccentricity(self):
        """Eccentricity"""
        return float(self.line2[26:33]) / 1e7

    @property
    def arg_periapsis(self):
        """Argument of periapsis (or perigee, because it refers to the Earth) in radians"""
        return float(self.line2[34:42]) * self._rad_deg

    @property
    def mean_anomaly(self):
        """Mean anomaly (radians)"""
        return float(self.line2[43:51]) * self._rad_deg

    @property
    def mean_motion(self):
        """Mean motion (radians/s)"""
        return float(self.line2[52:63]) * self._rad_rev * self._day_s

    @property
    def revolution_number(self):
        """Number of revolutions at the epoch"""
        return int(self.line2[63:68])

    @property
    def checksum_declared(self):
        """The checksum delivered in the file"""
        return int(self.line1[68]), int(self.line2[68])

    @property
    def checksum_calculated(self):
        "Return the TLE checksum calculated from the lines supplied"

        return self.checksum(self.line1), self.checksum(self.line2)


class TLEFileReader(TextFileReader):
    """A reader designed to accept Two-Line Element (TLE) inputs via files. Those files contain
    a single TLE without a Line 0 which should conform strictly to the TLE format. Converts the
    file to a dictionary and then uses :class:`~.TLEDictReader`. Also provides methods to access
    the properties in the same way as :class:`~.TLEDictReader`. See that class for descriptions of
    what's accessible.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f = open(self.path, 'r')
        self.line1 = f.readline()
        self.line2 = f.readline()
        self.tle = TLEDictReader({'line_1': self.line1, 'line_2': self.line2})

        # The following provides this class with the functionality from TLEDictReader to access
        # the TLE properties directly
        [setattr(self, m, getattr(self.tle, m)) for m in dir(self.tle) if not m.startswith('__')]
