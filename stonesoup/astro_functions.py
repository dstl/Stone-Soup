# convert to keplerian
# convert to cartesian
# convert to equinoctical
# convert to TLE
from .types.orbitalstate import KeplerianOrbitState, TLEOrbitState, EquinoctialOrbitState, CartesianOrbitState
import numpy as np


def convert_orbit(orbit, output_format):
    """
    Convert an orbit in a given format to another valid format.
    Valid formats : keplerian, tle, equinoctial and cartesian

    Parameters
    ==========
    orbit : :state:`OrbitState`
        Orbit to be converted
    output_format : string
        Orbit type to be converted to Valid formats : keplerian, tle,
        equinoctial and cartesian

    Returns
    =======
    converted_orbit : :state:`OrbitState`
        Orbit converted to the correct format

    """
    # Force string to lowercase to ease matching errors
    output_format = output_format.lower()

    valid_formats = {
                    "keplerian": "KeplerianOrbitState",
                    "tle": "TLEOrbitState",
                    "equinoctial": "EquinoctialOrbitState",
                    "cartesian": "CartesianOrbitState"
                    }
    if output_format in valid_formats.keys():
        if type(orbit) == valid_formats[output_format]:
                print("Warning: Orbit is already %s" % output_format)
                converted_orbit = orbit
        else:
            if output_format == "keplerian":
                converted_orbit = convert_to_keplerian_orbit(orbit)
            elif output_format == "tle":
                converted_orbit = convert_to_tle_orbit(orbit)
            elif output_format == "equinoctial":
                converted_orbit = convert_to_equinoctial_orbit(orbit)
            elif output_format == "cartesian":
                converted_orbit = convert_to_cartesian_orbit(orbit)
    else:
        raise AttributeError('Invalid output_format. Valid formats : keplerian, tle, equinoctial and cartesian')
    return converted_orbit


def convert_to_keplerian_orbit(orbit):
    """
    Convert orbit to a classical keplerian orbit
    Parameters
    ==========
    orbit : :state:`OrbitState`
        Orbit to be converted

    Returns
    =======
    keplerian_orbit : :state:`KeplerianOrbitState`
        Orbit converted to a Keplerian orbit
    """
    new_state_vector = np.array([[calculate_eccentricity(orbit)],
                                [calculate_semi_major_axis(orbit)],
                                [calculate_inclination(orbit)],
                                [calculate_long_asc_node(orbit)],
                                [calculate_arg_periapsis(orbit)],
                                [calculate_true_anomaly(orbit)]])
    keplerian_orbit = KeplerianOrbitState(new_state_vector, metadata=orbit.metadata)
    return keplerian_orbit


def convert_to_tle_orbit(orbit):
    """
    Convert orbit to a TLE-style orbit
    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to be converted

    Returns
    =======
    tle_orbit : :state:`TLEOrbitState`
        Orbit converted to a TLE orbit
    """
    new_state_vector = np.array([[calculate_eccentricity(orbit)],
                                [calculate_inclination(orbit)],
                                [calculate_long_asc_node(orbit)],
                                [calculate_arg_periapsis(orbit)],
                                [calculate_mean_motion(orbit)],
                                [calculate_mean_anomaly(orbit)]])
    tle_orbit = TLEOrbitState(new_state_vector, metadata=orbit.metadata)
    return tle_orbit


def convert_to_equinoctial_orbit(orbit):
    """
    Convert orbit to a equinoctical orbit
    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to be converted

    Returns
    =======
    equinoctial_orbit : :state:`EquinoctialOrbitState`
        Orbit converted to a Keplerian orbit
    """
    new_state_vector = np.array([[calculate_semi_major_axis(orbit)],
                                [calculate_horizontal_eccentricity(orbit)],
                                [calculate_vertical_eccentricity(orbit)],
                                [calculate_horizontal_inclination(orbit)],
                                [calculate_vertical_inclination(orbit)],
                                [calculate_mean_longitude(orbit)]])
    equinoctial_orbit = EquinoctialOrbitState(new_state_vector, metadata=orbit.metadata)
    return equinoctial_orbit


def convert_to_cartesian_orbit(orbit):
    """
    Convert orbit to a cartesian orbit
    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to be converted

    Returns
    =======
    equinoctial_orbit : :state:`CartesianOrbitState`
        Orbit converted to a Cartesian orbit
    """
    raise NotImplementedError


def orbit2cartpos(orbit):
    """
    Get current orbit position vector in Cartesian coordinates (xyz)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get position of

    Returns
    =======
    : Position vector in the parent-fixed inertial frame,
    r (sometimes confusingly called the state vector)


    """

    # Calculate position vector in perifocal coords.
    true_anomaly = calculate_true_anomaly(orbit)
    c_propo = (calculate_spec_angular_mom(orbit)**2 / orbit.grav_parameter) * \
              (1/(1 + calculate_eccentricity(orbit) * np.cos(true_anomaly)))
    r_peri = c_propo * np.array([[np.cos(true_anomaly)],
                                 [np.sin(true_anomaly)],
                                 [0]])

    # And transform to geocentric coordinates by means of matrix rotation
    return matrix_perifocal_to_geocentric(orbit) @ r_peri


def orbit2cartvel(orbit):
    """
    Get current orbit velocity vector in Cartesian coordinates (xyz)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get velocity of

    Returns
    =======
    : Velocity vector in the parent-centred inertial frame, :math:`v` or :math:`r^/dot`

    """
    true_anomaly = calculate_true_anomaly(orbit)
    # First calculate velocity vector in perifocal coordinates
    c_propo = (orbit.grav_parameter / calculate_spec_angular_mom(orbit))
    v_peri = c_propo * np.array([[-np.sin(true_anomaly)],
                                 [calculate_eccentricity(orbit) + np.cos(true_anomaly)],
                                 [0]])

    # then transform to geocentric coordinates by means of matrix rotation
    return matrix_perifocal_to_geocentric(orbit) @ v_peri


def matrix_perifocal_to_geocentric(orbit):
    """
    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get plane from

    Returns
    =======
    : Matrix to Rotate vectors from perifocal plane to the plane defined by the parent body

    """

    # Pre-compute some quantities
    s_lascn = np.sin(calculate_long_asc_node(orbit))
    c_lascn = np.cos(calculate_long_asc_node(orbit))

    s_inc = np.sin(calculate_inclination(orbit))
    c_inc = np.cos(calculate_inclination(orbit))

    s_argp = np.sin(calculate_arg_periapsis(orbit))
    c_argp = np.cos(calculate_arg_periapsis(orbit))

    # Populate the matrix
    return np.array([[-s_lascn * c_inc * s_argp + c_lascn * c_argp, -s_lascn * c_inc * c_argp - c_lascn * s_argp,
                   s_lascn * s_inc],
                  [c_lascn * c_inc * s_argp + s_lascn * c_argp, c_lascn * c_inc * c_argp - s_lascn * s_argp,
                   -c_lascn * s_inc],
                  [s_inc * s_argp, s_inc * c_argp, c_inc]])


def calculate_eccentricity(orbit):
    """
    Calculate eccentricity of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get eccentricity of

    Returns
    =======
    eccentricity : float
        Eccentricity of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        eccentricity = orbit.eccentricity
    elif isinstance(orbit, EquinoctialOrbitState):
        eccentricity = np.sqrt(np.square(orbit.horizontal_eccentricity)+np.square(orbit.vertical_eccentricity))
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised")
    return eccentricity


def calculate_semi_major_axis(orbit):
    """
    Calculate semi-major axis of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get semi-major axis of

    Returns
    =======
    semi_major_axis : float
        Semi-major axis of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, EquinoctialOrbitState):
        semi_major_axis = orbit.semimajor_axis
    elif isinstance(orbit, TLEOrbitState):
        numerator = np.power(orbit.grav_parameter, 1/3)
        denominator = np.power(orbit.mean_motion, 2/3)
        semi_major_axis = numerator/denominator
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised")
    return semi_major_axis


def calculate_inclination(orbit):
    """
    Calculate inclination of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get inclination of

    Returns
    =======
    inclination : float
        Inclination of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        inclination = orbit.inclination
    elif isinstance(orbit, EquinoctialOrbitState):
        inclination = 2 * np.arctan(np.sqrt(np.square(orbit.horizontal_inclination)+np.square(orbit.vertical_inclination)))
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised or implemented")
    inclination = np.remainder(inclination, np.pi/2)
    return inclination


def calculate_long_asc_node(orbit):
    """
    Calculate longitude of ascending node of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get longitude of ascending node of

    Returns
    =======
    long_asc_node : float
        longitude of ascending node of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        long_asc_node = orbit.long_asc_node
    elif isinstance(orbit, EquinoctialOrbitState):
        denominator = np.sqrt(np.square(orbit.horizontal_inclination)+np.square(orbit.vertical_inclination))
        sin_long_asc_node = orbit.horizontal_inclination/denominator
        cos_long_asc_node = orbit.vertical_inclination/denominator
        long_asc_node = np.arccos(cos_long_asc_node)
        if np.less(sin_long_asc_node, 0.0):
            long_asc_node = 2*np.pi - long_asc_node
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised or implemented")
    return long_asc_node


def calculate_arg_periapsis(orbit):
    """
    Calculate argument of periapsis of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get argument of periapsis  of

    Returns
    =======
    arg_periapsis : float
        Argument of periapsis of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        arg_periapsis = orbit.arg_periapsis
    elif isinstance(orbit, EquinoctialOrbitState):
        eta_cos = orbit.vertical_eccentricity/calculate_eccentricity(orbit)
        eta_sin = orbit.horizontal_eccentricity/calculate_eccentricity(orbit)
        eta = np.arcsin(eta_sin)
        arg_periapsis = eta - calculate_long_asc_node(orbit)
        if np.less(eta_sin, 0.0):
            arg_periapsis = 2*np.pi + arg_periapsis
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised or implemented")
    return arg_periapsis


def calculate_true_anomaly(orbit):
    """
    Calculate true anomaly of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get true anomaly of

    Returns
    =======
    true_anomaly : float
        True anomaly of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState):
        true_anomaly = orbit.true_anomaly
    elif isinstance(orbit, TLEOrbitState):
        mean_anomaly = orbit.mean_anomaly
        eccentricity = orbit.eccentricity
        # Get the new eccentric anomaly from the  mean anomaly
        eccentric_anomaly = calculate_itr_eccentric_anomaly(mean_anomaly, eccentricity)

        # And use that to find the true anomaly
        true_anomaly = 2 * np.arctan(np.sqrt((1+eccentricity) /
                                (1-eccentricity))*np.tan(eccentric_anomaly/2))
    elif isinstance(orbit, EquinoctialOrbitState):
        eta = np.arcsin(orbit.horizontal_eccentricity/calculate_eccentricity(orbit))
        true_anomaly = orbit.mean_longitude - eta
        if true_anomaly > 2*np.pi:
            true_anomaly -= 2*np.pi
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    else:
        raise AttributeError("Orbit type not recognised or implemented")
    return true_anomaly


def calculate_period(orbit):
    """

    Calculate the orbital period, :math:`T`, of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get orbital period of

    Returns
    =======
    : : float
        Orbital period of the input orbit
    """
    return ((2*np.pi)/np.sqrt(orbit.grav_parameter))*np.power(calculate_semi_major_axis(orbit), 3/2)


def calculate_spec_angular_mom(orbit):
    """
    Calculate the magnitude of the specific angular momentum, :math:`h`, of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get magnitude of the specific angular momentum of

    Returns
    =======
    : : float
        Magnitude of the specific angular momentum of the input orbit
    """
    return np.sqrt(orbit.grav_parameter * calculate_semi_major_axis(orbit) * (1 - calculate_eccentricity(orbit)**2))


def calculate_spec_orb_ener(orbit):
    """

    Calculate the specific orbital energy (:math:`\frac{-GM}{2a}`), of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get specific orbital energy of

    Returns
    =======
    : : float
        Specific orbital energy of the input orbit
    """
    return -orbit.grav_parameter/(2 * calculate_semi_major_axis(orbit))


def calculate_mean_anomaly(orbit):
    """

    Calculate the mean anomaly, :math:`M` (radians), of a given orbit

    If the input orbit is Keplerian, it uses the eccentric anomaly and Kepler's
    equation to get mean anomaly from true anomaly and eccentricity

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get mean anomaly of

    Returns
    =======
    mean_anomaly : float
        Mean anomaly of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState):
        eccentricity = calculate_eccentricity(orbit)
        ecc_anom = 2 * np.arctan(np.sqrt((1-eccentricity)/(1+eccentricity)) *
                                 np.tan(calculate_true_anomaly(orbit)/2))
        mean_anomaly = ecc_anom - (eccentricity * np.sin(ecc_anom)) # Kepler's equation
    elif isinstance(orbit, TLEOrbitState):
        mean_anomaly = orbit.mean_anomaly
    elif isinstance(orbit, EquinoctialOrbitState):
        eta = np.arccos(orbit.vertical_eccentricity/calculate_eccentricity(orbit))
        mean_anomaly = orbit.mean_longitude - eta
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    return mean_anomaly


def calculate_mean_motion(orbit):
    """

    Calculate the mean motion, :math:`M` (radians/s), of a given orbit

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get mean motion of

    Returns
    =======
    mean_anomaly : float
        Mean motion of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState):
        mean_motion = 2*np.pi/calculate_period(orbit)
    elif isinstance(orbit, TLEOrbitState):
        mean_motion = orbit.mean_motion
    elif isinstance(orbit, EquinoctialOrbitState):
        eta = np.arccos(orbit.vertical_eccentricity/calculate_eccentricity(orbit))
        mean_motion = orbit.mean_longitude - eta
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    return mean_motion


# Set of equinoctal elements
def calculate_horizontal_eccentricity(orbit):
    """
    Calculate the horizontal component of the eccentricity vector in
    Equinoctal coordinates (h)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get horizontal component of the eccentricity vector of

    Returns
    =======
    horizontal_eccentricity : float
        Horizontal component of the eccentricity vector of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        horizontal_eccentricity = calculate_eccentricity(orbit) * np.sin(calculate_arg_periapsis(orbit) + calculate_long_asc_node(orbit))
    elif isinstance(orbit, EquinoctialOrbitState):
        horizontal_eccentricity = orbit.horizontal_eccentricity
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")

    return horizontal_eccentricity


def calculate_vertical_eccentricity(orbit):
    """
    Calculate the vertical component of the eccentricity vector in
    Equinoctal coordinates (k)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get vertical component of the eccentricity vector of

    Returns
    =======
    vertical_eccentricity : float
        Vertical component of the eccentricity vector of the input orbit
    """

    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        vertical_eccentricity = calculate_eccentricity(orbit) * np.cos( calculate_arg_periapsis(orbit) + calculate_long_asc_node(orbit))
    elif isinstance(orbit, EquinoctialOrbitState):
        vertical_eccentricity = orbit.vertical_eccentricity
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")

    return vertical_eccentricity


def calculate_mean_longitude(orbit):
    """
    Calculate the  Equinoctal mean longitude (\lambda_0)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get mean longitude  of

    Returns
    =======
    mean_longitude : float
        Mean longitude of the input orbit
    """
    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        mean_longitude = calculate_mean_anomaly(orbit) + calculate_arg_periapsis(orbit) + calculate_long_asc_node(orbit)
    elif isinstance(orbit, EquinoctialOrbitState):
        mean_longitude = orbit.mean_longitude
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    return mean_longitude


def calculate_horizontal_inclination(orbit):
    """
    Calculate the horizontal component of the inclination vector (Equinoctal p)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get horizontal component of the inclination vector of

    Returns
    =======
    horizontal_inclination : float
        Horizontal component of the inclination vector of the input orbit
    """

    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        horizontal_inclination = (np.tan((calculate_inclination(orbit)/2))) * (np.sin(calculate_long_asc_node(orbit)))
    elif isinstance(orbit, EquinoctialOrbitState):
        horizontal_inclination = orbit.horizontal_inclination
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    return horizontal_inclination


def calculate_vertical_inclination(orbit):
    """

    Calculate the vertical component of the inclination vector (Equinoctal q)

    Parameters
    ==========
    orbit : :state:`OrbitalState`
        Orbit to get vertical component of the inclination vector of

    Returns
    =======
    vertical_inclination : float
        vertical component of the inclination vector of the input orbit
    """

    if isinstance(orbit, KeplerianOrbitState) | isinstance(orbit, TLEOrbitState):
        vertical_inclination = (np.tan((calculate_inclination(orbit)/2))) * (np.cos(calculate_long_asc_node(orbit)))
    elif isinstance(orbit, EquinoctialOrbitState):
        vertical_inclination = orbit.vertical_inclination
    elif isinstance(orbit, CartesianOrbitState):
        raise NotImplementedError("Cartesian not implemented yet")
    return vertical_inclination


def calculate_itr_eccentric_anomaly(mean_anomaly, eccentricity, tolerance=1e-8):
        """
        Approximately solve the transcendental equation :math:`E - e sin E = M_e` for E. This is an iterative process
        using Newton's method.

        Parameters
        ==========
        mean_anomaly: float
            Current mean anomaly
        eccentricity: float
            Orbital eccentricity
        tolerance: float
            Iteration tolerance

        Returns
        =======
        ecc_anomaly : float
            Eccentric anomaly of the input orbit
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
