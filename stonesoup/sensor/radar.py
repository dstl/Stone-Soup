# -*- coding: utf-8 -*-
import numpy as np

from .base import Sensor3DCartesian
from ..base import Property
from ..models.measurement.nonlinear import CartesianToBearingRange
from ..types.array import CovarianceMatrix
from ..types.detection import Detection
from ..types.state import State, StateVector


class RadarRangeBearing(Sensor3DCartesian):
    """A simple radar sensor that generates measurements of targets, using a
    :class:`~.CartesianToBearingRange` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.CartesianToBearingRange`\
            model")
    mapping = Property(
        [np.array],
        doc="Mapping between the targets state space and the sensors\
            measurement capability")
    noise_covar = Property(
        CovarianceMatrix,
        doc="The sensor noise covariance matrix. This is utilised by\
            (and follow in format) the underlying \
            :class:`~.CartesianToBearingRange` model")

    def measure(self, ground_truth, noise=None, **kwargs):
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truth : :class:`~.State`
            A ground-truth state

        Returns
        -------
        :class:`~.Detection`
            A measurement generated from the given state. The timestamp of the\
            measurement is set equal to that of the provided state.
        """
        measurement_model = CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

        measurement_vector = measurement_model.function(
            ground_truth.state_vector, noise=noise, **kwargs)

        return Detection(measurement_vector,
                         measurement_model=measurement_model,
                         timestamp=ground_truth.timestamp)


class RadarRotatingRangeBearing(RadarRangeBearing):
    """A simple rotating radar, with set field-of-view (FOV) angle, range and\
     rotations per minute (RPM), that generates measurements of targets, using\
     a :class:`~.CartesianToBearingRange` model, relative to its\
     position.

    Note
    ----
    * The current implementation of this class assumes a 3D Cartesian plane.

    """

    dwell_center = Property(
        State, doc="A state object, whose `state_vector`\
        property describes the rotation angle of the center of the sensor's\
        current FOV (i.e. the dwell center) relative to the positive x-axis\
        of the sensor frame/orientation. The angle is positive if the rotation\
        is in the counter-clockwise direction when viewed by an observer\
        looking down the z-axis of the sensor frame, towards the origin.\
        Angle units are in radians"
    )
    rpm = Property(
        float,
        doc="The number of antenna rotations per minute (RPM)")
    max_range = Property(
        float,
        doc="The maximum detection range of the radar (in meters)")
    fov_angle = Property(
        float,
        doc="The radar field of view (FOV) angle (in radians).")

    def measure(self, ground_truth, noise=None, **kwargs):
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truth : :class:`~.State`
            A ground-truth state

        Returns
        -------
        :class:`~.Detection` or ``None``
            A measurement generated from the given state, if the state falls\
            in the sensor's field of view, or ``None``, otherwise. The\
            timestamp of the measurement is set equal to that of the provided\
            state.
        """

        # Read timestamp from ground truth
        timestamp = ground_truth.timestamp

        # Rotate the radar antenna and compute new heading
        self.rotate(timestamp)
        antenna_heading = self.orientation[2, 0] + \
            self.dwell_center.state_vector[0, 0]

        # Set rotation offset of underlying measurement model
        rot_offset = \
            StateVector(
                [[self.orientation[0, 0]],
                 [self.orientation[1, 0]],
                 [antenna_heading]])

        measurement_model = CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rot_offset)

        # Transform state to measurement space and generate
        # random noise
        measurement_vector = measurement_model.function(
            ground_truth.state_vector, noise=0, **kwargs)
        if (noise is None):
            measurement_noise = measurement_model.rvs()
        else:
            measurement_noise = noise

        # Check if state falls within sensor's FOV
        fov_min = -self.fov_angle / 2
        fov_max = +self.fov_angle / 2
        bearing_t = measurement_vector[0, 0]
        range_t = measurement_vector[1, 0]

        # Return None if state not in FOV
        if (bearing_t > fov_max or bearing_t < fov_min
                or range_t > self.max_range):
            return None

        # Else return measurement
        measurement_vector += measurement_noise  # Add noise
        return Detection(measurement_vector,
                         measurement_model=measurement_model,
                         timestamp=timestamp)

    def rotate(self, timestamp):
        """Rotate the sensor's antenna

        This method computes and updates the sensor's `dwell_center` property.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`
            A timestamp signifying when the rotation completes
        """

        # Compute duration since last rotation
        duration = timestamp - self.dwell_center.timestamp

        # Update dwell center
        rps = self.rpm / 60  # rotations per sec
        self.dwell_center = State(
            StateVector([[self.dwell_center.state_vector[0, 0]
                          + duration.total_seconds() * rps * 2 * np.pi]]),
            timestamp
        )


class RadarRasterScanRangeBearing(RadarRotatingRangeBearing):
    """A simple raster scan radar, with set field-of-regard (FoR) angle, \
     field-of-view (FoV) angle, range and rotations per minute (RPM), that \
     generates measurements of targets, using a \
     :class:`~.RangeBearingGaussianToCartesian` model, relative to its position

     This is a simple extension of the RadarRotatingRangeBearing class with \
     the rotate function changed to restrict the  dwell-center to within the \
     field of regard.
     It's important to note that this only works (has  been tested) in an 2D \
     environment

    Note
    ----
    * The current implementation of this class assumes a 3D Cartesian plane.

    """

    for_angle = Property(
        float, doc="The radar field of regard (FoR) angle (in radians).")

    def rotate(self, timestamp):
        """Rotate the sensor's antenna

        This method computes and updates the sensor's `dwell_center` property.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`
            A timestamp signifying when the rotation completes
        """

        super().rotate(timestamp)

        dwell_center_max = self.for_angle/2.0 - self.fov_angle/2.0
        dwell_center_min = -self.for_angle/2.0 + self.fov_angle/2.0

        # If the FoV is outside of the FoR:
        #   Correct the dwell_center
        #   Reverse the direction of the scan pattern
        if self.dwell_center.state_vector[0, 0] > dwell_center_max:
            self.dwell_center = State(
                StateVector([[(2.0 * dwell_center_max) -
                              self.dwell_center.state_vector[0, 0]
                              ]]), timestamp)

            self.rpm = -self.rpm

        elif self.dwell_center.state_vector[0, 0] < dwell_center_min:
            self.dwell_center = State(
                StateVector([[(2.0 * dwell_center_min) -
                              self.dwell_center.state_vector[0, 0]
                              ]]), timestamp)

            self.rpm = -self.rpm
