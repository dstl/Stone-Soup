# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime

from ..orbital_functions import stumpf_s, stumpf_c
from ..astronomical_conversions import topocentric_to_geocentric, \
    topocentric_altaz_to_radec

from .base import Initiator
from ..base import Property
from ..types.track import Track
from ..types.orbitalstate import OrbitalState


class OrbitalInitiator(Initiator):
    r"""Parent class of orbital initiators. Will take various input
    detections  (according to type) and return initial estimate of
    :class:`~OrbitalState`.

    Properties
    ----------
    grav_parameter : float
            The gravitational parameter, :math:`\mu = GM', defaults to
            the Earth value of :math:`3.986004418 \times 10^{14} \,
            \mathrm{m}^{3} \mathrm{s}^{-2}

    """

    grav_parameter = Property(float, default="3.986004418e14",
                              doc=r"Standard gravitational parameter "
                                  r"(mathrm{m}^{3} \mathrm{s}^{-2}")


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
            c01 = np.atleast_2d(np.cross(br[0].ravel(), br[1].ravel())).T
            c12 = np.atleast_2d(np.cross(br[1].ravel(), br[2].ravel())).T
            c20 = np.atleast_2d(np.cross(br[2].ravel(), br[0].ravel())).T

            # This must be true if vectors co-planar
            assert np.dot(br[0].T/r[0], c12/np.linalg.norm(c12))[0][0] < \
                   tol_check

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
                    (np.atleast_2d(np.cross(bigd.ravel(),
                                            brr.ravel())).T/rr + bigs)

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
                crossr = np.atleast_2d(np.cross(br[0].ravel(), br[1].ravel())).T
                cterm = np.arccos(np.dot(br[0].T, br[1])/(r[0]*r[1]))[0][0]

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
    which has two :class:`~OrbitalState`.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """
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
                                 "argument to the "
                                 ":attr:`initiate()` function."
    )

    datetime_ut = Property(
        datetime, default=None, doc="The time (UT) at which the observation "
                                    "takes place, as a :class:`~datetime` "
                                    "object. If not supplied in the "
                                    ":attr:`initiate()` function or here then"
                                    ":attr:`datetime.utcnow()` is used."
    )

    inertial_angular_velocity = Property(
        float, default=7.292115e-5, doc=r"The angular velocity of the primary "
                                        "body in its inertial frame. Defaults"
                                        "to the value of the Earth in rad "
                                        "s^{-1}"
    )

    def initiate(self, detections, latitude=None, longitude=None, height=None,
                 datetime_ut=None, **kwargs):
        r"""Initiate tracks from detections

        Parameters
        ----------
        detections : list of :class:`~.Detection`
            A list of :class:`~Detection` objects with state vectors of
            the form :math:`[r, a, A, \frac{dr}{dt} \frac{da}{dt}
            \frac{dA}{dt}]^T`
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
        datetime_ut : datetime
            The time (UT) at which the observation takes place, as a
            :class:`~datetime` object. If not supplied here the parent
            class is checked and if that's none then
            :attr:`datetime.utcnow()` is used.

        kwargs :

        Returns
        -------
        : set of :class:`~.Track`
            set of :class:`~.Tracks`, composed of :class:`~.OrbitalState`

        """
        # Figure out where and when we are. Note that an extra layer of
        # complexity will be required if we want to initialised tracks from
        # different locations. We might add these a
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

        if datetime_ut is None:
            if self.datetime_ut is None:
                ltd = datetime.utcnow()
            else:
                ltd = self.datetime_ut
        else:
            ltd = datetime_ut


        # Initialise tracks container.
        tracks = set()
        # Run through the list of detections
        for detection in detections:
            # Extract information in the detection
            rn_al_az = detection.state_vector
            bigr = topocentric_to_geocentric(latitude, longitude, height,
                                          datetime_ut=datetime_ut)
            ra, dec = topocentric_altaz_to_radec(rn_al_az[1], rn_al_az[2],
                                                 latitude, longitude,
                                                 datetime_ut)

            # Caching some trig results
            sdec = np.sin(dec)
            cdec = np.cos(dec)
            cra = np.cos(ra)
            sra = np.sin(ra)

            # Unit vector direction
            du_ran = np.array([cdec*cra, cdec*sra, sdec])

            # Geocentric position
            r = bigr + rn_al_az[0]*du_ran
            omega = self.inertial_angular_velocity * np.array([[0], [0], [1]])
            bigrdot = np.atleast_2d(np.cross(omega.ravel(), bigr.ravel())).T

            # Caching more trig
            slat = np.sin(latitude)
            clat = np.cos(latitude)
            salt = np.sin(rn_al_az[1])
            calt = np.cos(rn_al_az[1])
            saz = np.sin(rn_al_az[2])
            caz = np.cos(rn_al_az[2])

            # The rates of change of ra and dec are
            decdot = (1/cdec) * (-rn_al_az[5] * clat * saz * cra + rn_al_az[4]
                                 * (slat*calt - clat*caz*salt))

            radot = self.inertial_angular_velocity + \
                (rn_al_az[5] * caz * calt - rn_al_az[4] * saz * salt +
                    decdot * saz * calt * np.tan(dec)) / \
                (clat*salt - slat*caz*calt)

            # Direction cosine rates vector
            du_rho = np.array([-radot * sra * cdec - decdot * cra * sdec,
                               radot * cra * cdec - decdot * sra * sdec,
                               decdot * cdec])

            # The velocity vector is then
            v = bigrdot + rn_al_az[3]*du_ran + rn_al_az[0]*du_rho

            # Finally construct the state vector
            track = Track(OrbitalState(np.concatenate((r, v), axis=0),
                                    timestamp=detection.timestamp))
            tracks.add(track)

        return tracks


class GaussInitiator(OrbitalInitiator):
    """Implements Gauss's method."""