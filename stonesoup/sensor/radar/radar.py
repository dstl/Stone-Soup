# -*- coding: utf-8 -*-
import copy
import numpy as np
from math import erfc
from ...functions import cart2sphere, rotx, roty, rotz
from ..base import Sensor
from ...base import Property
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...types.array import CovarianceMatrix
from ...types.detection import Detection
from ...types.state import State, StateVector
from .beam_shape import BeamShape
from .beam_pattern import BeamTransitionModel
from ...models.measurement.base import MeasurementModel
from ...types.numeric import Probability
import scipy.constants as const


class RadarRangeBearing(Sensor):
    """A simple radar sensor that generates measurements of targets, using a
    :class:`~.RangeBearingGaussianToCartesian` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    position = Property(
        StateVector,
        doc="The radar position on a 3D Cartesian plane,"
            " expressed as a 3x1 array of Cartesian "
            "coordinates in the order :math:`x,y,z`")
    orientation = Property(
        StateVector,
        doc="A 3x1 array of angles (rad), specifying the radar orientation in "
            "terms of the counter-clockwise rotation around each Cartesian "
            "axis in the order :math:`x,y,z`. The rotation angles are "
            "positive if the rotation is in the counter-clockwise direction "
            "when viewed by an observer looking along the respective rotation "
            "axis, towards the origin")
    ndim_state = Property(
        int,
        doc="Number of state dimensions. This is utilised by (and follows in"
            "format) the underlying :class:`~.RangeBearingGaussianToCartesian`"
            "model")
    mapping = Property(
        [np.array],
        doc="Mapping between the targets state space and "
            "the sensors measurement capability")
    noise_covar = Property(CovarianceMatrix,
                           doc="The sensor noise covariance matrix. This is "
                               "utilised by (and follow in format) the "
                               "underlying "
                               ":class:`~.RangeBearingGaussianToCartesian`"
                               "model")

    def __init__(self, position, orientation, ndim_state, mapping, noise_covar,
                 *args, **kwargs):
        measurement_model = CartesianToBearingRange(
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
        if noise is None:
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


class AESARadar(Sensor):
    r"""An AESA (Active electronically scanned array) radar model that
    calculates the signal to noise ratio (SNR) of a target and the subsequent
    probability of detection (PD). The SNR is calculated using:

    .. math::

        \mathit{SNR} = \dfrac{c^2\, n_p \,\beta}{64\,\pi^3 \,kT_0 \,B \,F \,f^2
         \,L} \times\dfrac{\sigma\, G_a^2\, P_t}{R^4}

    where
    :math:`c` is the speed of light
    :math:`n_p` is the number of pulses in a burst
    :math:`\beta` is the duty cycle which is unitless
    :math:`k` is the boltzmann constant
    :math:`T_0` is system temperature in kelvin
    :math:`B` is the bandwidth in hertz
    :math:`F` is the noise figure unitless
    :math:`f` is the frequency in hertz
    :math:`L` is the loss which is unitless
    The probability of detection (:math:`P_{d}`) is calculated
    using the North's approximation,

    .. math::

        P_{d} = 0.5\, erfc\left(\sqrt{-\ln{P_{fa}}}-\sqrt{ \mathit{SNR}
         +0.5} \right)

    where :math:`P_{fa}` is the probability of false alarm.
    In this model the AESA scan angle effects the gain by:

    .. math::

        G_a = G_a \cos{\left(\theta\right)}
        \cos{\left(\phi\right)}

    where :math:`\theta` and :math:`\phi` are respectively the azimuth and
    elevation angles in respects to the boresight of the antenna.
    The effect of beam spoiling on the beam width is :

    .. math::

        \Delta\theta = \dfrac{\Delta\theta}{\cos{\left(\theta\right)}
        \cos{\left(\phi\right)}}

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.
    This model does not generate false alarms.
    """

    rotation_offset = Property(
        StateVector, default=StateVector([0, 0, 0]),
        doc="A 3x1 array of angles (rad), specifying "
            "the radar orientation in terms of the "
            "counter-clockwise rotation around the "
            ":math:`x,y,z` axis. i.e Roll, Pitch and Yaw.")
    translation_offset = Property(
        StateVector, default=StateVector([0, 0, 0]),
        doc="The radar position in 3D Cartesian space.[x,y,z]")
    mapping = Property(
        np.array, default=[0, 1, 2],
        doc="Mapping between or positions and state "
            "dimensions. [x,y,z]")

    measurement_model = Property(
        MeasurementModel, default=None,
        doc="The Measurement model used to generate "
            "measurements.")

    beam_shape = Property(
        BeamShape,
        doc="Object describing the shape of the beam.")

    beam_transition_model = Property(
        BeamTransitionModel,
        doc="Object describing the "
            "movement of the beam in azimuth and "
            "elevation from the perspective of "
            "the radar.")
    # SNR variables
    number_pulses = Property(
        int, default=1,
        doc="The number of pulses in the"
            " radar burst.")
    duty_cycle = Property(
        float,
        doc="Duty cycle is the fraction of the time "
            "the radar it transmitting.")
    band_width = Property(
        float, doc="Bandwidth of the receiver in hertz.")
    receiver_noise = Property(
        float, doc="Noise figure of the radar in decibels.")
    frequency = Property(
        float, doc="Transmitted frequency in hertz.")
    antenna_gain = Property(
        float, doc="Total Antenna gain in decibels.")
    beam_width = Property(
        float, doc="Radar beam width in radians.")
    loss = Property(
        float, default=0, doc="Loss in decibels.")

    swerling_on = Property(
        bool, default=False,
        doc="Is the Swerling 1 case used. If True the RCS"
            " of the target will change for each timestep. "
            "The random RCS follows the probability "
            "distribution of the Swerling 1 case.")
    rcs = Property(
        float, default=None,
        doc="The radar cross section of targets in meters squared.")

    probability_false_alarm = Property(
        Probability, default=1e-6,
        doc="Probability of false alarm used in the North's approximation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _snr_constant(self):
        temp = 290  # noise reference temperature (room temperature kelvin)
        # convert from dB
        noise_figure = 10 ** (self.receiver_noise / 10)
        loss = 10 ** (self.loss / 10)
        # calculate part of snr that is independent of:
        #   rcs, transmitted power, gain and range
        return (const.c ** 2 * self.number_pulses * self.duty_cycle) / \
               (64 * np.pi ** 3 * const.k * temp * self.band_width *
                noise_figure * self.frequency ** 2 * loss)

    @property
    def _rotation_matrix(self):
        """_rotation_matrix getter method

        Calculates and returns the (3D) axis rotation matrix.

        Returns
        -------
        :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
        """

        theta_x = -self.rotation_offset[0, 0]  # roll
        theta_y = -self.rotation_offset[1, 0]  # pitch#elevation
        theta_z = -self.rotation_offset[2, 0]  # yaw#azimuth

        return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

    @staticmethod
    def _swerling_1(rcs):
        return -rcs * np.log(np.random.rand())

    def gen_probability(self, sky_state):
        """Generates probability of detection of a given State.

        Parameters
        ----------
        sky_state : The target state.

        Returns
        -------
        det_prob : `float`
            Probability of detection.
        snr : `float`
            Signal to noise ratio.
        rcs : `float`
            RCS after the Swerling 1 case is applied.
        directed_power : `float`
            Power in the direction of the target.
        spoiled_gain : `float`
            Gain in decibels with beam spoiling applied.
        spoiled_width : `float`
            Beam width with beam spoiling applied.
        """
        if getattr(sky_state, 'rcs', None) is not None:
            # use state's rcs if it has one
            rcs = sky_state.rcs
        else:
            rcs = self.rcs
        # apply swerling 1 case?
        if self.swerling_on:
            rcs = self._swerling_1(rcs)

        # e-scan beam steer
        [beam_az, beam_el] = self.beam_transition_model.move_beam(
            sky_state.timestamp)  # [az,el]

        # effects of e-scan on gain and beam width
        spoiled_gain = 10 ** (self.antenna_gain / 10) * np.cos(beam_az) * \
            np.cos(beam_el)
        spoiled_width = self.beam_width / (np.cos(beam_az) * np.cos(beam_el))
        # state relative to radar (in cartesian space)
        relative_vector = sky_state.state_vector[self.mapping] - \
            self.translation_offset[self.mapping]
        relative_vector = self._rotation_matrix @ relative_vector

        # calculate target position in spherical coordinates
        [r, pos_az, pos_el] = cart2sphere(*relative_vector)

        # target position relative to beam position
        relative_az = pos_az[0] - beam_az
        relative_el = pos_el[0] - beam_el
        # calculate power directed towards target
        self.beam_shape.beam_width = spoiled_width  # beam spoiling to width
        directed_power = self.beam_shape.beam_power(relative_az, relative_el)
        # calculate signal to noise ratio
        snr = self._snr_constant * rcs * spoiled_gain ** 2 * directed_power / \
            (r[0] ** 4)
        # calculate probability of detection using the North's approximation
        det_prob = 0.5 * erfc(
            (-np.log(self.probability_false_alarm)) ** 0.5 - (
                    snr + 1 / 2) ** 0.5)
        return det_prob, snr, rcs, directed_power,\
            10 * np.log10(spoiled_gain), spoiled_width

    def gen_measurement(self, sky_state, **kwargs):
        """Generate a measurement for a given state

        Parameters
        ----------
        sky_state : :class:`~.State`
            A target state in 3-D cartesian space

        Returns
        -------
        :class:`~.Detection` or ``None``
            A measurement generated from the given state, if np.random.rand()
            is less than the probability of detection, or returns ``None``.
            The timestamp of the measurement is equal to that of
            the input state.
        """
        det_prob = self.gen_probability(sky_state)[0]
        # Is the state detected?
        if np.random.rand() <= det_prob:
            self.measurement_model.translation_offset = self.translation_offset
            self.measurement_model.rotation_offset = self.rotation_offset
            measured_pos = self.measurement_model.function(
                sky_state.state_vector)

            return Detection(measured_pos, timestamp=sky_state.timestamp,
                             measurement_model=self.measurement_model)
