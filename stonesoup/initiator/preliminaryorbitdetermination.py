# -*- coding: utf-8 -*-

import numpy as np

from .base import Initiator
from ..base import Property
from ..types.track import Track
from ..types.orbitalstate import OrbitalState

class OrbitalInitiator(Initiator):
    r"""Parent class of orbital initiators. Will take various input
    detections  (according to type) and return initial estimate of
    :class:`~OrbitalState`."""

    grav_parameter = Property(float, default="3.986004418e14",
                              doc=r"Standard gravitational parameter "
                                  r"(mathrm{m}^{3} \mathrm{s}^{-2}")


class GibbsInitiator(OrbitalInitiator):
    """An initiator which uses the Gibbs method requires three position
    vectors.

    The :attr:`inititate` function takes an input list of exactly 3
    detections.


    """
    def initiate(self, detections, tol_check=1e-5, **kwargs):
        r"""Generate tracks from detections.

        Parameters
        ----------
        detections : list of :class:`~.Detection` triples
            Exactly 3 position vectors are used to generate a track. Each
            element in the list is a triple of detections; the track is
            initiated from the middle element. (Or we could initiate a
            track at each point?)
        grav_parameter : float
            The gravitational parameter, :math:`\mu = GM', defaults to
            the Earth value of :math:`3.986004418 \times 10^{14} \,
            \mathrm{m}^{3} \mathrm{s}^{-2}
        tol_check : float
            A parameter setting to what precision the assertion that the
            dot product of the unit vectors of r0 with the cross of r1
            and r2 is zero. The default is 10^{-5}.

        Returns
        -------
        : set of :class:`~.Track`
            Tracks generated from detections
        """
        # Initialise tracks
        tracks = set()
        # Run through the list
        for detection in detections:

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
                   tol_check  # Not sure if this test is difficult or easy to
            # pass

            bign = r[0]*c12 + r[1]*c20 + r[2]*c01
            bigd = c01 + c12 + c20
            bigs = br[0]*(r[1] - r[2]) + br[1]*(r[2] - r[0]) + br[2]*(r[0] - r[1])

            # Finding the velocity for all three points. It's often practice to
            # use only the central point, i.e. br[1]
            track = Track()
            for det, rr in zip(detection, r):
                brr = det.state_vector
                bvv = np.sqrt(self.grav_parameter/
                              (np.linalg.norm(bign)*np.linalg.norm(bigd))) * \
                      (np.atleast_2d(np.cross(bigd.ravel(),
                                              brr.ravel())).T/rr + bigs)
                print(brr)
                print(np.cross(bigd.ravel(), brr.ravel())/rr + bigs.T)

                orbstate = OrbitalState(np.concatenate((brr, bvv), axis=0),
                                    timestamp=det.timestamp)
                # Put this state into a track object and into the set of tracks
                track.append(orbstate)

            tracks.add(track)

        return tracks

class LambertInitiator(OrbitalInitiator):
    """Implements Lagrange's solution to the Lambert problem. Takes 2
    positional state vectors and a time interval as input. Returns the
    estimated state vector."""

class GaussInitiator(OrbitalInitiator):
    """Implements Gauss's method."""