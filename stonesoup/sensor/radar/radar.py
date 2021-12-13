# -*- coding: utf-8 -*-
import copy
from math import erfc
from typing import Tuple, Set, Union

import numpy as np
import scipy.constants as const

from .beam_pattern import BeamTransitionModel
from .beam_shape import BeamShape
from ...base import Property
from ...functions import cart2sphere, rotx, roty, rotz, mod_bearing
from ...models.measurement.base import MeasurementModel
from ...models.measurement.nonlinear import \
    (CartesianToBearingRange, CartesianToElevationBearingRange,
     CartesianToBearingRangeRate, CartesianToElevationBearingRangeRate)
from ...sensor.sensor import Sensor
from ...types.array import CovarianceMatrix
from ...types.detection import TrueDetection
from ...types.groundtruth import GroundTruthState
from ...types.numeric import Probability
from ...types.state import State, StateVector
from ...models.clutter import ClutterModel


class RadarBearingRange(Sensor):
    """A simple radar sensor that generates measurements of targets, using a
    :class:`~.CartesianToBearingRange` model, relative to its position.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    ndim_state: int = Property(
        default=2,
        doc="Number of state dimensions. This is utilised by (and follows in format) "
            "the underlying :class:`~.CartesianToBearingRange` model")
    position_mapping: Tuple[int, int] = Property(
        doc="Mapping between the targets state space and the sensors "
            "measurement capability")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by "
            "(and follow in format) the underlying "
            ":class:`~.CartesianToBearingRange` model")
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` ojects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detections = set.union(detections, clutter)

        return detections


class RadarRotatingBearingRange(RadarBearingRange):
    """A simple rotating radar, with set field-of-view (FOV) angle, range and\
     rotations per minute (RPM), that generates measurements of targets, using\
     a :class:`~.CartesianToBearingRange` model, relative to its\
     position.

    Note
    ----
    * The current implementation of this class assumes a 3D Cartesian plane.

    """

    dwell_center: State = Property(
        doc="A state object, whose `state_vector` "
            "property describes the rotation angle of the center of the sensor's "
            "current FOV (i.e. the dwell center) relative to the positive x-axis "
            "of the sensor frame/orientation. The angle is positive if the rotation "
            "is in the counter-clockwise direction when viewed by an observer "
            "looking down the z-axis of the sensor frame, towards the origin. "
            "Angle units are in radians"
    )
    rpm: float = Property(doc="The number of antenna rotations per minute (RPM)")
    max_range: float = Property(doc="The maximum detection range of the radar (in meters)")
    fov_angle: float = Property(doc="The radar field of view (FOV) angle (in radians).")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        # Read timestamp from ground truth
        try:
            timestamp = next(iter(ground_truths.copy())).timestamp
        except StopIteration:
            # No ground truths to get timestamp from
            return set()

        # Rotate the radar antenna and compute new heading
        self.rotate(timestamp)
        antenna_heading = self.orientation[2, 0] + self.dwell_center.state_vector[0, 0]

        # Set rotation offset of underlying measurement model
        rot_offset = \
            StateVector(
                [[self.orientation[0, 0]],
                 [self.orientation[1, 0]],
                 [antenna_heading]])

        measurement_model = CartesianToBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rot_offset)

        detections = set()
        for truth in ground_truths:
            # Transform state to measurement space and generate
            # random noise
            measurement_vector = measurement_model.function(truth, noise=False, **kwargs)

            if noise is True:
                measurement_noise = measurement_model.rvs()
            else:
                measurement_noise = noise

            # Check if state falls within sensor's FOV
            fov_min = -self.fov_angle / 2
            fov_max = +self.fov_angle / 2
            bearing_t = measurement_vector[0, 0]
            range_t = measurement_vector[1, 0]

            # Do not measure if state not in FOV
            if (bearing_t > fov_max or bearing_t < fov_min
                    or range_t > self.max_range):
                continue

            # Else add measurement
            measurement_vector += measurement_noise  # Add noise

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections

    def rotate(self, timestamp):
        """Rotate the sensor's antenna

        This method computes and updates the sensor's `dwell_center` property.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`
            A timestamp signifying when the rotation completes
        """

        # Check if dwell_center has a timestamp instantiated if not sets it to incoming timestamp
        if self.dwell_center.timestamp is None:
            self.dwell_center.timestamp = timestamp

        # Compute duration since last rotation
        duration = timestamp - self.dwell_center.timestamp

        # Update dwell center
        rps = self.rpm / 60  # rotations per sec
        angle = self.dwell_center.state_vector[0, 0] + duration.total_seconds()*rps*2*np.pi
        self.dwell_center = State(StateVector([[mod_bearing(angle)]]), timestamp)


class RadarElevationBearingRange(RadarBearingRange):
    """A  radar sensor that generates measurements of targets, using a
    :class:`~.CartesianToElevationBearingRange` model, relative to its position.

    Note
    ----
    This implementation of this class assumes a 3D Cartesian space.

    """

    ndim_state: int = Property(
        default=3,
        doc="Number of state dimensions. This is utilised by (and follows in format) "
            "the underlying :class:`~.CartesianToBearingRange` model")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by "
            "(and follow in format) the underlying "
            ":class:`~.CartesianToElevationBearingRange` model")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = CartesianToElevationBearingRange(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detections = set.union(detections, clutter)

        return detections


class RadarBearingRangeRate(RadarBearingRange):
    """ A radar sensor that generates measurements of targets, using a
    :class:`~.CartesianToBearingRangeRate` model, relative to its position
    and velocity.

    Note
    ----
    This class implementation assuming at 3D cartesian space, it therefore\
     expects a 6D state space.

    """

    velocity_mapping: Tuple[int, int, int] = Property(
        default=(1, 3, 5),
        doc="Mapping to the target's velocity information within its state space")
    ndim_state: int = Property(
        default=3,
        doc="Number of state dimensions. This is utilised by (and follows in format) "
            "the underlying :class:`~.CartesianToBearingRangeRate` model")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by "
            "(and follow in format) the underlying "
            ":class:`~.CartesianToBearingRangeRate` model")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = CartesianToBearingRangeRate(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            velocity_mapping=self.velocity_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            velocity=self.velocity,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections


class RadarElevationBearingRangeRate(RadarBearingRangeRate):
    """ A radar sensor that generates measurements of targets, using a
    :class:`~.CartesianToElevationBearingRangeRate` model, relative to its position
    and velocity.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """

    velocity_mapping: Tuple[int, int, int] = Property(
        default=(1, 3, 5),
        doc="Mapping to the target's velocity information within its state space")
    ndim_state: int = Property(
        default=6,
        doc="Number of state dimensions. This is utilised by (and follows in format) "
            "the underlying :class:`~.CartesianToElevationBearingRangeRate` model")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by "
            "(and follow in format) the underlying "
            ":class:`~.CartesianToElevationBearingRangeRate` model")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = CartesianToElevationBearingRangeRate(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            velocity_mapping=self.velocity_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            velocity=self.velocity,
            rotation_offset=self.orientation)

        detections = set()
        for truth in ground_truths:
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)
            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        return detections


class RadarRasterScanBearingRange(RadarRotatingBearingRange):
    """A simple raster scan radar, with set field-of-regard (FoR) angle, \
     field-of-view (FoV) angle, range and rotations per minute (RPM), that \
     generates measurements of targets, using a \
     :class:`~.CartesianToBearingRange` model, relative to its position

     This is a simple extension of the RadarRotatingBearingRange class with \
     the rotate function changed to restrict the  dwell-center to within the \
     field of regard.
     It's important to note that this only works (has  been tested) in an 2D \
     environment

    Note
    ----
    This class implementation assuming at 3D cartesian space, it therefore\
     expects a 6D state space.

    """

    for_angle: float = Property(doc="The radar field of regard (FoR) angle (in radians).")

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
    rotation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array of angles (rad), specifying the radar orientation in terms of the "
            "counter-clockwise rotation around the :math:`x,y,z` axis. i.e Roll, Pitch and Yaw. "
            "Default is ``StateVector([0, 0, 0])``")
    position_mapping: Tuple[int, int, int] = Property(
        default=(0, 1, 2),
        doc="Mapping between or positions and state "
            "dimensions. [x,y,z]")
    measurement_model: MeasurementModel = Property(
        doc="The Measurement model used to generate "
            "measurements.")
    beam_shape: BeamShape = Property(
        doc="Object describing the shape of the beam.")
    beam_transition_model: BeamTransitionModel = Property(
        doc="Object describing the movement of the beam in azimuth and elevation from the "
            "perspective of the radar.")
    # SNR variables
    number_pulses: int = Property(
        default=1, doc="The number of pulses in the radar burst.")
    duty_cycle: float = Property(
        doc="Duty cycle is the fraction of the time the radar it transmitting.")
    band_width: float = Property(
        doc="Bandwidth of the receiver in hertz.")
    receiver_noise: float = Property(
        doc="Noise figure of the radar in decibels.")
    frequency: float = Property(
        doc="Transmitted frequency in hertz.")
    antenna_gain: float = Property(
        doc="Total Antenna gain in decibels.")
    beam_width: float = Property(
        doc="Radar beam width in radians.")
    loss: float = Property(
        default=0, doc="Loss in decibels.")
    swerling_on: bool = Property(
        default=False,
        doc="Is the Swerling 1 case used. If True the RCS"
            " of the target will change for each timestep. "
            "The random RCS follows the probability "
            "distribution of the Swerling 1 case.")
    rcs: float = Property(
        default=None,
        doc="The radar cross section of targets in meters squared. Used if rcs not present on "
            "truth. Default `None`, where 'rcs' must be present on truth.")
    probability_false_alarm: Probability = Property(
        default=1e-6, doc="Probability of false alarm used in the North's approximation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rotation_offset is None:
            self.rotation_offset = StateVector([0, 0, 0])

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
        : :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
        """

        theta_x = -self.rotation_offset[0, 0]  # roll
        theta_y = -self.rotation_offset[1, 0]  # pitch#elevation
        theta_z = -self.rotation_offset[2, 0]  # yaw#azimuth

        return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

    @staticmethod
    def _swerling_1(rcs):
        return -rcs * np.log(np.random.rand())

    def gen_probability(self, truth):
        """Generates probability of detection of a given State.

        Parameters
        ----------
        truth : The target state.

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
        if getattr(truth, 'rcs', None) is not None:
            # use state's rcs if it has one
            rcs = truth.rcs
        else:
            rcs = self.rcs
        if rcs is None:
            raise ValueError("Truth missing 'rcs' attribute and no default 'rcs' provided")

        # apply swerling 1 case?
        if self.swerling_on:
            rcs = self._swerling_1(rcs)

        # e-scan beam steer
        [beam_az, beam_el] = self.beam_transition_model.move_beam(
            truth.timestamp)  # [az,el]

        # effects of e-scan on gain and beam width
        spoiled_gain = 10 ** (self.antenna_gain / 10) * np.cos(beam_az) * np.cos(beam_el)
        spoiled_width = self.beam_width / (np.cos(beam_az) * np.cos(beam_el))
        # state relative to radar (in cartesian space)

        relative_vector = truth.state_vector[self.position_mapping, :] - self.position

        relative_vector = self._rotation_matrix @ relative_vector

        # calculate target position in spherical coordinates
        [r, pos_az, pos_el] = cart2sphere(*relative_vector)

        # target position relative to beam position
        relative_az = pos_az - beam_az
        relative_el = pos_el - beam_el
        # calculate power directed towards target
        directed_power = self.beam_shape.beam_power(relative_az, relative_el, spoiled_width)
        # calculate signal to noise ratio
        snr = self._snr_constant * rcs * spoiled_gain ** 2 * directed_power / (r ** 4)
        # calculate probability of detection using the North's approximation
        det_prob = 0.5 * erfc(
            (-np.log(self.probability_false_alarm)) ** 0.5 - (
                    snr + 1 / 2) ** 0.5)
        return det_prob, snr, rcs, directed_power, 10 * np.log10(spoiled_gain), spoiled_width

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        detections = set()

        measurement_model = copy.deepcopy(self.measurement_model)
        measurement_model.translation_offset = self.position.copy()
        measurement_model.rotation_offset = self.rotation_offset.copy()

        for truth in ground_truths:
            det_prob = self.gen_probability(truth)[0]
            # Is the state detected?
            if np.random.rand() <= det_prob:
                measured_pos = measurement_model.function(truth, noise=noise)

                detection = TrueDetection(measured_pos,
                                          timestamp=truth.timestamp,
                                          measurement_model=measurement_model,
                                          groundtruth_path=truth)
                detections.add(detection)

        return detections
