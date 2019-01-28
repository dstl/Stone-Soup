# -*- coding: utf-8 -*-
"""Astronomical readers for Stone Soup.

This is a collection of readers for Stone Soup, allowing quick reading
of data that is in common astronomical formats.

Readers include:
    FITS
    TLE
    SATCAT
"""
import numpy as np

from astropy.io import fits
from .file import FileReader, TextFileReader
from ..type import TwoLineElement


class FITSReader(FileReader):
    """A simple reader for FITS files. Reads a FITS file and sets the contents
    as the data and header attributes. Uses the Astropy library for reading
    files.

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
                self._data.append(hdu_list[index]._data)
                self._header.append(hdu_list[index]._header)

    @property
    def data(self):
        return self._data

    @property
    def header(self):
        return self._header


class TLEReader(TextFileReader):
    """A simple reader for text files containing a list of Two Line Elements
    (TLEs).

    Parameters
    ----------
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_TLEs(self):
        """Parse TLEs from an opened file.

        Parameters
        ----------
        """

        self._tlelist = []
        # Read in file
        with self._file as tle_file:
            lines = tle_file.readlines()
            lines = [line.rstrip("\n") for line in lines]

        number_of_lines = len(lines)
        # Check to make sure the file is valid i.e. number of lines
        # are divisible by 2
        assert number_of_lines % 2 == 0
        # split file every 2 lines
        tles = list(zip(*(iter(lines),) * 2))
        for parsed_tle in tles:
            # Create TLE instance
            tle_instance = self.create_tle(parsed_tle)
            # Add TLE instance to TLE list
            # self._tlelist.append(tle_instance)
            # return self._tlelist

    def create_tle(tleobj):
        """ Create a TwoLineElement instance from a parsed TLE object.
        Follows the structure found at https://en.wikipedia.org/wiki/Two-line_element_set
        Parameters
        ----------
        tleobj: The parsed TLE in list format
        """

        # Read first line
        first_line = tleobj[0].split()
        first_line_number = int(first_line[0])
        satellite_number_first_line = int(first_line[1][0:-1])
        classification = first_line[1][-1]
        international_designator_number = int(first_line[2][0:-1])
        international_designator_type = first_line[2][-1]
        epoch_year = int(first_line[3][0:1])
        epoch_day = float(first_line[3][2:])
        first_time_derivative = float(first_line[4])
        second_time_derivative = float(first_line[5][0:-2])
        second_time_derivative_power_sign = first_line[5][-2]
        second_time_derivative_power_value = float(first_line[5][-1])
        if second_time_derivative_power_sign == "-":
            second_time_derivative_power_value *= -1
            second_time_derivative *= np.power(10, second_time_derivative_power_value)
        BSTAR_drag_term = float(first_line[6][0:-2])
        BSTAR_drag_term_power_sign = first_line[6][-2]
        BSTAR_drag_term_power_value = float(first_line[6][-1])
        if BSTAR_drag_term_power_sign == "-":
            BSTAR_drag_term_power_value *= -1
            BSTAR_drag_term *= np.power(10, BSTAR_drag_term_power_value)

        zero = int(first_line[7])
        element_set_number = int(first_line[8])
        checksum_mod_10_first = int(first_line[9])

        # Read second line
        second_line = tleobj[1].split()
        second_line_number = int(second_line[0])
        satellite_number_second_line = int(second_line[1])
        inclination = float(second_line[2])
        right_ascension_of_ascend_node = float(second_line[3])
        eccentricity = float(second_line[4])
        argument_of_perigee = float(second_line[5])
        mean_anomaly = float(second_line[6])
        mean_motion = float(second_line[7])
        rev_number_at_epoch = int(second_line[8])
        checksum_mod_10_second = int(second_line[9])

        # Check validity of TwoLineElement
        assert first_line_number == 1
        assert second_line_number == 2
        assert satellite_number_first_line == satellite_number_second_line
        assert zero == 0
        # Create metadata dict
        metadata = {
            "Satellite Number": satellite_number_first_line,
            "Classification": classification,
            "International Designator Number": international_designator_number,
            "International Designator Type": international_designator_type,
            "Epoch Year": epoch_year,
            "Epoch Day": epoch_day,
            "First Time Derivative": first_time_derivative,
            "Second Time Derivative": second_time_derivative,
            "BSTAR Drag Term": BSTAR_drag_term,
            "Zero": zero,
            "Element Set Number": element_set_number,
            "Checksum First Line": checksum_mod_10_first,
            "Inclination (Degrees)": inclination,
            "Right Ascension of Ascending Node (Degrees)": right_ascension_of_ascend_node,
            "Eccentricity": eccentricity,
            "Argument of Perigee (Degrees)": argument_of_perigee,
            "Mean Anomaly": mean_anomaly,
            "Mean Motion (Degrees)": mean_motion,
            "Revolution Number at Epoch": rev_number_at_epoch,
            "Checksum Second Line": checksum_mod_10_second
        }

        # Calculate semi-major axis
        grav_constant = 3.986004418e14
        a_numerator = np.power(grav_constant, 1/3)
        mean_motion_rads = mean_motion*((2*np.pi)/86400)
        a_denominator = np.power(mean_motion_rads, 2/3)
        a = a_numerator/a_denominator

        # Get the new eccentric anomaly from the  mean anomaly
        eccentric_anomaly = self.itr_eccentric_anomaly(mean_anomaly*(np.pi/180), eccentricity)

        # And use that to find the new true anomaly
        true_anomaly = 2 * np.arctan(np.sqrt((1+eccentricity) /
                                (1-eccentricity))*np.tan(eccentric_anomaly/2))
        orbital_state_vector = np.array([[eccentricity],
                                        [a],
                                        [inclination*(np.pi/180)],
                                        [right_ascension_of_ascend_node*(np.pi/180)],
                                        [argument_of_perigee*(np.pi/180)],
                                        [true_anomaly]])

        return TwoLineElement(orbital_state_vector, metadata)
        # Create orbital elements
        # Create TLE instance
        # return tle_instance

    def itr_eccentric_anomaly(self, mean_anomaly, eccentricity, tolerance=1e-8):
        r"""

        Approximately solve the transcendental equation :math:`E - e sin E = M_e` for E. This is an iterative process
        using Newton's method.

        :param mean_anomaly: Current mean anomaly
        :param eccentricity: Orbital eccentricity
        :param tolerance:
        :return: the eccentric anomaly


        """
        if mean_anomaly < np.pi:
            ecc_anomaly = mean_anomaly + eccentricity/2
        else:
            ecc_anomaly = mean_anomaly - eccentricity/2

        ratio = 1

        while ratio > tolerance:
            f = ecc_anomaly - eccentricity*np.sin(ecc_anomaly) - mean_anomaly
            fp = 1 - eccentricity*np.cos(ecc_anomaly)
            ratio = f/fp # Need to check conditioning
            ecc_anomaly = ecc_anomaly - ratio

        return ecc_anomaly
