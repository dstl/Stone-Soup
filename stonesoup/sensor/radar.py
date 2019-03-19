# -*- coding: utf-8 -*-
import copy
import numpy as np

from .base import Sensor
from ..base import Property
from ..types.state import State, StateVector
from ..types.detection import Detection
from ..types.array import CovarianceMatrix
from ..models.measurement.nonlinear\
    import RangeBearingGaussianToCartesian


class RadarRangeBearing(Sensor):
    """A simple radar sensor that generates measurements of targets, using a
    :class:`~.RangeBearingGaussianToCartesian` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    position = Property(StateVector,
                        doc="The radar position on a 3D Cartesian plane,\
                             expressed as a 3x1 array of Cartesian coordinates\
                             in the order :math:`x,y,z`")
    orientation = Property(
        StateVector,
        doc="A 3x1 array of angles (rad), specifying the radar orientation in \
            terms of the counter-clockwise rotation around each Cartesian \
            axis in the order :math:`x,y,z`. The rotation angles are positive \
            if the rotation is in the counter-clockwise direction when viewed \
            by an observer looking along the respective rotation axis, \
            towards the origin")
    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in\
            format) the underlying :class:`~.RangeBearingGaussianToCartesian`\
            model")
    mapping = Property(
        [np.array], doc="Mapping between the targets state space and the\
                        sensors measurement capability")
    noise_covar = Property(CovarianceMatrix,
                           doc="The sensor noise covariance matrix. This is \
                                utilised by (and follow in format) the \
                                underlying \
                                :class:`~.RangeBearingGaussianToCartesian`\
                                model")

    def __init__(self, position, orientation, ndim_state, mapping, noise_covar,
                 *args, **kwargs):
        measurement_model = RangeBearingGaussianToCartesian(
            ndim_state=ndim_state,
            mapping=mapping,
            noise_covar=noise_covar,
            translation_offset=position,
            rotation_offset=orientation)

        super().__init__(position, orientation, ndim_state, mapping,
                         noise_covar, *args, measurement_model, **kwargs)

    def set_position(self, position):
        self.position = position
        self.measurement_model.translation_offset = position

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.measurement_model.rotation_offset = orientation

    def gen_measurement(self, ground_truth, noise=None, **kwargs):
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

        measurement_vector = self.measurement_model.function(
            ground_truth.state_vector, noise=noise, **kwargs)

        model_copy = copy.copy(self.measurement_model)

        return Detection(measurement_vector,
                         measurement_model=model_copy,
                         timestamp=ground_truth.timestamp)


class RadarRotatingRangeBearing(RadarRangeBearing):
    """A simple rotating radar, with set field-of-view (FOV) angle, range and\
     rotations per minute (RPM), that generates measurements of targets, using\
     a :class:`~.RangeBearingGaussianToCartesian` model, relative to its\
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
        float, doc="The number of antenna rotations per minute (RPM)")
    max_range = Property(
        float, doc="The maximum detection range of the radar (in meters)")
    fov_angle = Property(
        float, doc="The radar field of view (FOV) angle (in radians).")

    def __init__(self, position, orientation, ndim_state, mapping, noise_covar,
                 dwell_center, rpm, max_range, fov_angle, *args, **kwargs):

        super().__init__(position, orientation, ndim_state, mapping,
                         noise_covar, dwell_center, rpm, max_range,
                         fov_angle, *args, **kwargs)

    def gen_measurement(self, ground_truth, noise=None, **kwargs):
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
        rot_offset =\
            StateVector(
                [[self.orientation[0, 0]],
                 [self.orientation[1, 0]],
                 [antenna_heading]])
        self.measurement_model.rotation_offset = rot_offset

        # Transform state to measurement space and generate
        # random noise
        measurement_vector = self.measurement_model.function(
            ground_truth.state_vector, noise=0, **kwargs)
        if(noise is None):
            measurement_noise = self.measurement_model.rvs()
        else:
            measurement_noise = noise

        # Check if state falls within sensor's FOV
        fov_min = -self.fov_angle/2
        fov_max = +self.fov_angle/2
        bearing_t = measurement_vector[0, 0]
        range_t = measurement_vector[1, 0]

        # Return None if state not in FOV
        if(bearing_t > fov_max or bearing_t < fov_min
           or range_t > self.max_range):
            return None

        # Else return measurement
        model_copy = copy.copy(self.measurement_model)
        measurement_vector += measurement_noise  # Add noise
        return Detection(measurement_vector,
                         measurement_model=model_copy,
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
        rps = self.rpm/60  # rotations per sec
        self.dwell_center = State(
            StateVector([[self.dwell_center.state_vector[0, 0]
                          + duration.total_seconds()*rps*2*np.pi]]),
            timestamp
        )
