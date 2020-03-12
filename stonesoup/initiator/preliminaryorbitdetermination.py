# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime

from ..functions import dotproduct as dot, crossproduct as cross
from ..orbital_functions import stumpf_s, stumpf_c, universal_anomaly_newton, \
    lagrange_coefficients_from_universal_anomaly
from ..astronomical_conversions import topocentric_to_geocentric, \
    topocentric_altaz_to_radec, topocentric_altaz_to_radecrate, \
    direction_cosine_unit_vector, direction_rate_cosine_unit_vector

from .base import Initiator
from ..base import Property
from ..types.track import Track
from ..types.orbitalstate import OrbitalState


class OrbitalInitiator(Initiator):
    r"""Parent class of orbital initiators. Will take various input
    detections  (according to type) and return initial estimate of
    :class:`~OrbitalState`.

    """
    grav_parameter = Property(
        float, default=3.986004418e14, doc=r"Standard gravitational "
                                             r"parameter. Defaults to the "
                                             r"Earth's value (3.986004418 "
                                             r"\times 10^{14} mathrm{m}^{3} "
                                             r"\mathrm{s}^{-2})."
    )

    latitude = Property(
        float, default=None, doc="The latitude of the observer's location "
                                 "(radians). Doesn't have to be supplied if "
                                 "included as an argument to the "
                                 ":attr:`initiate()` function."
    )

    longitude = Property(
        float, default=None, doc="The longitude of the observer's location "
                                 "(radians). Doesn't have to be supplied if "
                                 "included as an argument to the "
                                 ":attr:`initiate()` function."
    )

    height = Property(
        float, default=None, doc="The height of the observer's location "
                                 "(m) above the notional sea level. Doesn't "
                                 "have to be supplied if included as an "
                                 "argument to the :attr:`initiate()` function."
    )

    datetime_ut = Property(
        datetime, default=None, doc="The time (UT) at which the observation "
                                    "takes place, as a :class:`~datetime` "
                                    "object. If not supplied in the "
                                    ":attr:`initiate()` function or here then"
                                    ":attr:`datetime.utcnow()` should be used."
    )

    inertial_angular_velocity = Property(
        float, default=7.292115e-5, doc=r"The angular velocity of the primary "
                                        "body in its inertial frame. Defaults"
                                        "to the value of the Earth in rad "
                                        "s^{-1}"
    )

    def _check_lat_lon_hei(self, latitude, longitude, height):
        """Decide whether to one has been passed (lat, lon, hei). If not
        use the native ones.

        Parameters
        ----------
        latitude : float
            Latitude to check (rad)
        longitude : float
            Longitude to check (rad)
        height : float
            Height to check (m)

        Returns
        -------
        : float, float, float
            Latitude, Longitude, Height

        """
        if latitude is None:
            if self.latitude is None:
                raise ValueError("A latitude must be supplied somewhere")
            else:
                latitude = self.latitude

        if longitude is None:
            if self.longitude is None:
                raise ValueError("A longitude must be supplied somewhere")
            else:
                longitude = self.longitude

        if height is None:
            if self.height is None:
                raise ValueError("A height must be supplied somewhere")

        return latitude, longitude, height

    def _check_time(self, datetime_ut):
        """Check the datetime object, if None try to return the one in the
        parent.

        Parameters
        ----------
        datetime_ut : datetime.datetime
            The datetime object to check

        Returns
        -------
        : datetime.datetime
            The returned datetime object

        """
        if datetime_ut is None:
            if self.datetime_ut is None:
                ltd = datetime.utcnow()
            else:
                ltd = self.datetime_ut
        else:
            ltd = datetime_ut

        return ltd


class GibbsInitiator(OrbitalInitiator):
    """An initiator which uses the Gibbs method requires three position
    vectors. The method is outlined in S5.2 in [1].

    The :attr:`inititate()` function takes an input list of tuples of
    exactly 3 detections and returns a :class:`~Set` of :class:`~Track`
    each of which has three :class:`~OrbitalState`.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering Students,
        Third Edition, Elsevier

    """
    def initiate(self, detections, tol_check=1e-5, **kwargs):
        r"""Generate tracks from detections.

        Parameters
        ----------
        detections : list of :class:`~.Detection` triples
            Exactly 3 position vectors are used to generate a track. Each
            element in the list is a triple of detections. Position
            vectors must be in ECI coordinates.
        tol_check : float
            A parameter setting to what precision the assertion that the
            dot product of the unit vectors of r0 with the cross of r1
            and r2 is zero. The default is 10^{-5}.

        Returns
        -------
        : set of :class:`~.Track`
            Set of :class:`~.Track`, each of which is composed of
            :class:`~.OrbitalState`
        """

        # Initialise tracks container.
        tracks = set()
        # Run through the list of detections
        for detection in detections:

            # TODO: Work out how to do this with more than three input states
            # TODO: Presumably it's then a fitting problem
            if len(detection) != 3:
                raise TypeError("Number of detections must be 3")

            # extract the position vectors
            br = np.array([detection[0].state_vector,
                           detection[1].state_vector,
                           detection[2].state_vector])

            # position vector norms
            r = np.linalg.norm(br, axis=1)

            # vector cross products
            # (wish Python would duck-type vectors...)
            c01 = cross(br[0], br[1])
            c12 = cross(br[1], br[2])
            c20 = cross(br[2], br[0])

            # This must be true if vectors co-planar
            assert dot(br[0]/r[0], c12/np.linalg.norm(c12)) < tol_check

            # Some interim vector quantities
            bign = r[0]*c12 + r[1]*c20 + r[2]*c01
            bigd = c01 + c12 + c20
            bigs = br[0]*(r[1] - r[2]) + br[1]*(r[2] - r[0]) + br[2]*(r[0] -
                                                                      r[1])

            # Find the velocity for all three points and put them into a
            # Track() object. Note it's often practice to return only the
            # inferred state at the central point, i.e. br[1].
            track = Track()
            for det, rr in zip(detection, r):
                brr = det.state_vector
                bvv = np.sqrt(self.grav_parameter/
                              (np.linalg.norm(bign)*np.linalg.norm(bigd))) * \
                    (cross(bigd, brr)/rr + bigs)

                orbstate = OrbitalState(np.concatenate((brr, bvv), axis=0),
                                        timestamp=det.timestamp)

                # Put this state into a track object
                track.append(orbstate)

            # Add track to the set of tracks
            tracks.add(track)

        return tracks


class LambertInitiator(OrbitalInitiator):
    """Implements Lagrange's solution to the Lambert problem. Takes 2
    positional state vectors and a time interval as input. Returns the
    estimated state vector. The method is outlined in S5.3 in [1].

    The :attr:`inititate()` function takes an input list of tuples of
    exactly 2 timestamped detections and returns a :class:`~Set` of
    :class:`~Track` each of which has two :class:`~OrbitalState`.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """
    _z_precision = Property(float, default=1e-8,
                            doc="The precision with which to calculate the"
                            "value of z using Newton's method")

    def initiate(self, detections, true_anomalies=[], directions=[],
                 tol_check=1e-5, **kwargs):
        r"""Generate tracks from detections

        Parameters
        ----------
        detections : list of :class:`~.Detection` doubles
            Exactly 2 position vectors are used to generate a track. Each
            element in the list is a pair of timestamped detections.
            Postion vectors must be in ECI coordinates.
        true_anomalies : list of float (optional, default is empty)
            A list of true anomaly deltas with length equal to the number of
            detection pairs. If the list element is specified then the
            calculation of true anomaly from the detections (and the
            direction ambiguity) is bypassed. No checking for consistency
            between the detections and the true anomaly is undertaken.
        directions : list of char (optional, default is empty)
            A list of length equal to the number of detection pairs.
            Allowed terms are prograde or retrograde. The direction of
            travel used to calculate the angular deviation. In the event
            that it isn't specified, the smallest implied angular
            deviation is used. Alternatively, if a single item is
            specified then that direction is applied to all detections.
        tol_check : float
            A parameter setting to what precision the assertion that the
            dot product of the unit vectors of r0 with the cross of r1
            and r2 is zero. The default is 10^{-5}.

        Returns
        -------
        : set of :class:`~.Track`
            set of :class:`~.Tracks`, composed of :class:`~.OrbitalState`
        """

        # Initialise tracks container.
        tracks = set()

        # Check emptiness of true anomaly list
        if len(true_anomalies) == 0:
            true_anomalies = [None] * len(detections)

        # If we only have one direction then use this for all. If, on the other
        # hand we have 0, then set as undefined
        if len(directions) == 0:
            directions = [''] * len(detections)
        elif len(directions) == 1:
            directions = [directions[0]] * len(detections)

        # Run through the list of detections (and corresponding lists)
        for detection, true_anomaly, direction in \
                zip(detections, true_anomalies, directions):

            # TODO: Work out how to do this with more than two input states
            # TODO: Presumably it's then a fitting problem
            if len(detection) != 2:
                raise TypeError("Number of detections must be 2")

            # Get the time delta
            deltat = detection[1].timestamp - detection[0].timestamp

            # extract the position vectors
            br = np.array([detection[0].state_vector,
                           detection[1].state_vector])

            # position vector norms
            r = np.linalg.norm(br, axis=1)

            """If the true anomaly is not supplied then one must decide on 
            either a prograde or a retrograde orbit. There's a clear ambiguity
            in considering 2 points (it could have gone round either way). 
            We'll resolve this by assuming that the smallest angular deviation 
            is correct, unless the <direction> keyword is supplied."""
            if true_anomaly is None:
                crossr = cross(br[0], br[1])
                cterm = np.arccos(dot(br[0], br[1])/(r[0]*r[1]))

                if (direction.lower() == "prograde" and crossr[2] >= 0) or \
                        (direction.lower() == "retrograde" and crossr[2] < 0):
                    dtheta = cterm
                elif (direction.lower() == "prograde" and crossr[2] < 0) or \
                        (direction.lower() == "retrograde" and crossr[2] >= 0):
                    dtheta = 2*np.pi - cterm
                else:
                    # if direction isn't specified use the smallest angle (in 0 <
                    # theta <= pi)
                    dtheta = min(cterm % (2*np.pi), (2*np.pi-cterm) % (2*np.pi))
            else:
                dtheta = true_anomaly

            biga = np.sin(dtheta) * np.sqrt((r[0]*r[1])/(1 - np.cos(dtheta)))

            # Run the iterative process of finding z by Newton's method
            z = 0
            fratio = 1
            while abs(fratio) > self._z_precision:
                y = r[0] + r[1] + biga * (z*stumpf_s(z) - 1)/(np.sqrt(stumpf_c(z)))

                # Need to ensure that the units of the gravitational parameter are
                # in seconds
                bigf = (y/stumpf_c(z))**1.5 * stumpf_s(z) + biga * np.sqrt(y) - \
                    np.sqrt(self.grav_parameter) * deltat.total_seconds()

                if z == 0:
                    bigfp = (np.sqrt(2)/40) * y**1.5 + biga/8 * \
                            (np.sqrt(y) + biga*np.sqrt(1/(2*y)))
                else:
                    bigfp = (y/stumpf_c(z))**1.5 * (
                            (1/(2*z))*(stumpf_c(z) -
                                     3*stumpf_s(z)/(2*stumpf_c(z))) +
                            (3*stumpf_s(z)**2/(4*stumpf_c(z)))) + biga/8 * \
                            (3*stumpf_s(z)/stumpf_c(z) * np.sqrt(y) +
                             biga*np.sqrt(stumpf_c(z)/y))

                fratio = bigf/bigfp
                z = z - fratio

            f = 1 - y/r[0]
            g = biga*np.sqrt(y/self.grav_parameter)
            gdot = 1 - y/r[1]

            bv = np.array([1/g * (br[1] - f*br[0]), 1/g * (gdot*br[1] - br[0])])

            track = Track()
            for det, brr, bvv in zip(detection, br, bv):
                orbstate = OrbitalState(np.concatenate((brr, bvv), axis=0),
                                        timestamp=det.timestamp)
                track.append(orbstate)

            tracks.add(track)

        return tracks


class RangeAltAzInitiator(OrbitalInitiator):
    """Implements an initiator based on measurements of range, altitude,
    and azimuth, together with their rates, from an observatory. The
    method is outlined in S5.8 in [1]. It's a bunch of coordinate
    transforms, basically.

    The :attr:`inititate()` function takes an input list of timestamped
    detections and returns a :class:`~Set` of :class:`~Track` each of
    which has a single :class:`~OrbitalState`. At present the timestamp
    must be in UT. No checking for timezone conversions is done.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """

    def initiate(self, detections, latitude=None, longitude=None, height=None,
                 **kwargs):
        r"""Initiate tracks from detections

        Parameters
        ----------
        detections : list of :class:`~.Detection`
            A list of :class:`~Detection` objects with state vectors of
            the form :math:`[r, a, A, \frac{dr}{dt} \frac{da}{dt}
            \frac{dA}{dt}]^T`. It has a timestamp in UT from where the
            sidereal time is calculated.
        latitude : float
            The latitude of the observer's location (radians). If not
            supplied or None, the parent class will be checked for an
            instance before an error is thrown.
        longitude : float
            The longitude of the observer's location (radians). If not
            supplied or None, the parent class will be checked for an
            instance before an error is thrown.
        height : float
            The height of the observer's location (m) above the notional
            sea level. If not supplier, or None, the parent class is
            checked before an error is thrown.

        kwargs :

        Returns
        -------
        : set of :class:`~.Track`
            set of :class:`~.Tracks`, composed of :class:`~.OrbitalState`

        """
        # Figure out where and when we are. Note that an extra layer of
        # complexity will be required if we want to initialise tracks from
        # different locations.
        # TODO augment to be able to initilise tracks from different locations
        latitude, longitude, height = self._check_lat_lon_hei(latitude,
                                                              longitude,
                                                              height)

        # Initialise tracks container.
        tracks = set()
        # Run through the list of detections
        for detection in detections:
            # Extract information in the detection
            rn_al_az = detection.state_vector
            # Oberver's position
            bigr = topocentric_to_geocentric(latitude, longitude, height,
                                             datetime_ut=detection.timestamp)
            # Target's position on the sky
            ra, dec = topocentric_altaz_to_radec(rn_al_az[1], rn_al_az[2],
                                                 latitude, longitude,
                                                 detection.timestamp)
            # The cosine unit vector in the direction of the target
            du_ran = direction_cosine_unit_vector(ra, dec)

            # Geocentric position of the target
            r = bigr + rn_al_az[0]*du_ran

            # Observer's velocity
            omega = self.inertial_angular_velocity * np.array([[0], [0], [1]])
            bigrdot = cross(omega, bigr)

            # Rate of change in RA and Dec
            radot, decdot = topocentric_altaz_to_radecrate(rn_al_az[1],
                                                           rn_al_az[2],
                                                           rn_al_az[4],
                                                           rn_al_az[5],
                                                           latitude, longitude,
                                                           datetime_ut=
                                                           detection.timestamp,
                                                           inertial_angular_velocity=
                                                           self.inertial_angular_velocity)

            # Direction rate cosine vector
            du_rho = direction_rate_cosine_unit_vector(ra, dec, radot, decdot)

            # The velocity vector in geocentric coordinates is then
            v = bigrdot + rn_al_az[3]*du_ran + rn_al_az[0]*du_rho

            # Finally construct the state vector
            track = Track(OrbitalState(np.concatenate((r, v), axis=0),
                                       timestamp=detection.timestamp))
            # And add it to the track container
            tracks.add(track)

        return tracks


class GaussInitiator(OrbitalInitiator):
    """Implements Gauss's method for angles-only measurements. The method
    is outlined in S5.10 of [1].

    The :attr:`inititate()` function takes an input list of timestamped
    RA, Dec triples and returns a :class:`~Set` of :class:`~Track` each
    of which has three :class:`~OrbitalState`s.

    Warning
    -------
    This currently assumes a static observer with respect to the Earth's
    surface, i.e. the latitude, longitude and height do not change,
    though the position vectors do, as the Earth rotates.

    TODO: allow the possibility that the initiate() function takes
    TODO: varying lat, lon, hei

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """
    allowed_range = Property(
        np.array, default=np.array([6378100, 384400000]),
        doc="This is the range interval within which to restrict consideration "
            "of orbits when initiating tracks. The default extends between the "
            "earth's surface and the orbit of the moon."
    )

    itr_improvement_factor = Property(
        float, default=None, doc="Carry out the iterative improvement to the "
                                "preliminary orbit estimate via the universal "
                                "Kepler equation until the change in the slant"
                                 "ranges falls below this number."
    )


    def initiate(self, detections, latitude=None, longitude=None, height=None,
                 uanom_precision=1e-8, **kwargs):
        r"""Initiate tracks from detections

        Parameters
        ----------
        detections : list of :class:`~.Detection`
            A list of timestamped :class:`~Detection` triples each with state
            vectors of the form :math:`[RA, Dec]^T`
        latitude : float
            The latitude of the observer's location (radians). If not
            supplied or None, the parent class will be checked for an
            instance before an error is thrown.
        longitude : float
            The longitude of the observer's location (radians). If not
            supplied or None, the parent class will be checked for an
            instance before an error is thrown.
        height : float
            The height of the observer's location (m) above the notional
            sea level. If not supplier, or None, the parent class is
            checked before an error is thrown.
        uanom_precision : float (optional)
            The precision to which to calculate the universal anomaly via
            Newton's method.

        kwargs :

        Returns
        -------
        : set of :class:`~.Track`
            set of :class:`~.Tracks`, composed of (three)
            :class:`~.OrbitalState`

        """

        # This function used several times later. Gets the slant ranges from
        # various coefficients
        def slantrangefromcd(cc1, cc3, bd):
            # rhos are the slant ranges
            rho_1 = -bd[0][0] + bd[1][0] / cc1 - cc3 * bd[2][0] / cc1
            rho_2 = -cc1 * bd[0][1] + bd[1][1] - cc3 * bd[2][1]
            rho_3 = -cc1 * bd[0][2] / cc3 + bd[1][2] / cc3 - bd[2][2]

            return [rho_1, rho_2, rho_3]

        # Figure out where and when we are. Note that an extra layer of
        # complexity will be required if we want to initialise tracks from
        # different locations.
        # Figure out where and when we are. Note that an extra layer of
        # complexity will be required if we want to initialise tracks from
        # different locations.
        # TODO augment to be able to initilise tracks from different locations
        latitude, longitude, height = self._check_lat_lon_hei(latitude,
                                                              longitude,
                                                              height)

        # Initialise tracks container.
        tracks = set()
        # Run through the list of detections
        for detectiontriple in detections:

            if len(detectiontriple) != 3:
                raise TypeError("Number of detections must be 3")

            timetriple = []
            bigr = []
            dcuv = []
            for detection in detectiontriple:
                # times
                timetriple.append(detection.timestamp)

                # extract the position vectors as a list
                bigr.append(topocentric_to_geocentric(latitude, longitude,
                                                      height, datetime_ut=
                                                      detection.timestamp))

                # The cosine unit vector in the direction of the target
                dcuv.append(direction_cosine_unit_vector(
                    detection.state_vector[0], detection.state_vector[1]))

            # Time deltas
            tau1 = (timetriple[0] - timetriple[1]).total_seconds()
            tau3 = (timetriple[2] - timetriple[1]).total_seconds()
            tau = tau3 - tau1

            # Cross products of direction cosines
            boldp1 = cross(dcuv[1], dcuv[2])
            boldp2 = cross(dcuv[0], dcuv[2])
            boldp3 = cross(dcuv[0], dcuv[1])

            # Scalar triple products
            bigd0 = dot(dcuv[0], boldp1)

            # Triple products in a matrix
            bigd = np.array([[dot(bigr[0], boldp1), dot(bigr[0], boldp2),
                              dot(bigr[0], boldp3)],
                             [dot(bigr[1], boldp1), dot(bigr[1], boldp2),
                              dot(bigr[1], boldp3)],
                             [dot(bigr[2], boldp1), dot(bigr[2], boldp2),
                              dot(bigr[2], boldp3)]])/bigd0

            # Prepare to construct polynomial
            biga = (-bigd[0][1]*tau3/tau + bigd[1][1] + bigd[2][1]*tau1/tau)
            bigb = (1/6) * (bigd[0][1]*(tau3**2 - tau**2)*tau3/tau +
                            bigd[2][1]*(tau**2 - tau1**2)*tau1/tau)
            bige = dot(bigr[1], dcuv[1])
            bigr2sq = dot(bigr[1], bigr[1])

            # Coefficients of the eighth-order polynomial
            smla = -(biga**2 + 2*biga*bige + bigr2sq)
            smlb = -2*self.grav_parameter * bigb * (biga + bige)
            smlc = -self.grav_parameter**2 * bigb**2

            # Find the roots of this equation:
            '''Set a range of reasonable values within which to restrict the 
            roots (in the class). Then pick the (hopefully one) non-complex 
            non-negative root. If more than one is found, raise a warning and
             pick the one with lowest value...'''
            # Set up the coefficient matrix
            coeffs = np.array([1, 0, smla, 0, 0, smlb, 0, 0, smlc])
            roots = np.roots(coeffs)  # Calculate the roots

            # Now pick only the most sensible roots. That is those that are
            # wholly real, positive and exist within the limits defined by way
            # of the admitted_region attribute. The root is solution to the
            # equation for geocentric radius, r2.
            for r2 in roots:
                # Note that no attempt is made to check what number, if any,
                # 'in-range' slant ranges exists and so anywhere between 0 and
                # many tracks may be generated for each detection triple
                if np.isreal(r2) and min(self.allowed_range) < r2 < \
                        max(self.allowed_range):

                    # approximate the factors of the linear combination of r_
                    mu_rsq = self.grav_parameter / r2 ** 3
                    c1 = (tau3/tau) * (1 + (1/6)*mu_rsq*(tau**2 - tau3**2))
                    c3 = -(tau1/tau) * (1 + (1/6)*mu_rsq*(tau**2 - tau1**2))

                    # rhos are the slant ranges
                    rhos = slantrangefromcd(c1, c3, bigd)

                    # Approximate Lagrange coefficients
                    f1 = 1 - (1 / 2) * mu_rsq * tau1**2
                    f3 = 1 - (1 / 2) * mu_rsq * tau3**2
                    g1 = tau1 - (1 / 6) * mu_rsq * tau1**3
                    g3 = tau3 - (1 / 6) * mu_rsq * tau3**3

                    # Calculate the position vectors of three input points
                    boldr = []
                    for bigrr, rho, dcuvv in zip(bigr, rhos, dcuv):
                        boldr.append(bigrr + rho*dcuvv)

                    # middle velocity
                    boldv2 = (1 / (f1 * g3 - f3 * g1)) * (-f3 * boldr[0] + f1 * boldr[2])

                    # This is the approximate answer. Do we want to invoke the
                    # iterative improvement which uses the universal Kepler
                    # equation?
                    while self.itr_improvement_factor is not None:
                        # Use universal variables to work out better f1, g1,
                        # f3, g3 via the universal anomaly
                        f1, g1, _, _ = \
                            lagrange_coefficients_from_universal_anomaly(
                                np.concatenate((boldr[1], boldv2)),
                                timetriple[0] - timetriple[1],
                                grav_parameter=self.grav_parameter,
                                precision=uanom_precision)
                        f3, g3, _, _ = \
                            lagrange_coefficients_from_universal_anomaly(
                                np.concatenate((boldr[1], boldv2)),
                                timetriple[2] - timetriple[1],
                                grav_parameter=self.grav_parameter,
                                precision=uanom_precision)

                        # A handy combination of these is
                        c1 = g3/(f1*g3 - f3*g1)
                        c3 = -g1/(f1*g3 - f3*g1)

                        # Use these to get updated rhos and thereby updated
                        # boldrs. Stop if the rhos don't change enough
                        drhos = np.subtract(slantrangefromcd(c1, c3, bigd), rhos)

                        # Calculate the position vectors of three input points
                        boldr = []
                        for bigrr, rho, dcuvv in zip(bigr, np.add(rhos, drhos),
                                                     dcuv):
                            boldr.append(bigrr + rho * dcuvv)

                        # calculate the updated v2 and continue:
                        # middle velocity
                        boldv2 = (1 / (f1 * g3 - f3 * g1)) * (-f3 * boldr[0] +
                                                              f1 * boldr[2])

                        # Figure out how to compare each element in this list
                        # and stop if all items are within tolerance
                        if np.all(np.less(drhos, self.itr_improvement_factor)):
                            break
                        else:
                            rhos += drhos

                    """Concatenate r2 and v2 and construct the state and add it
                     to the tracks. Note that this 'solves' the issue of 
                     multiple roots by adding more tracks, which can't both
                     be true. TODO: consider choosing best track?"""
                    tracks.add(Track(OrbitalState(np.concatenate((boldr[1],
                                                                  boldv2),
                                                                 axis=0),
                                                  timestamp=
                                                  detection.timestamp)))


        return tracks


