# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..radar import (
    RadarRangeBearing, RadarRotatingRangeBearing, RadarRasterScanRangeBearing)
from ....functions import cart2pol
from ....types.angle import Bearing
from ....types.array import StateVector, CovarianceMatrix
from ....types.state import State
from ....types.groundtruth import GroundTruthState
from ..radar import RadarRangeBearing, RadarRotatingRangeBearing, AESARadar
from ..beam_pattern import StationaryBeam
from ..beam_shape import Beam2DGaussian
from ....models.measurement.linear import LinearGaussian


def h2d(state_vector, translation_offset, rotation_offset):

    xyz = [[state_vector[0, 0] - translation_offset[0, 0]],
           [state_vector[1, 0] - translation_offset[1, 0]],
           [0]]

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    rot_z = np.array([[cos_z, -sin_z, 0],
                      [sin_z, cos_z, 0],
                      [0, 0, 1]])

    theta_y = - rotation_offset[1, 0]
    cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
    rot_y = np.array([[cos_y, 0, sin_y],
                      [0, 1, 0],
                      [-sin_y, 0, cos_y]])

    theta_x = - rotation_offset[0, 0]
    cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
    rot_x = np.array([[1, 0, 0],
                      [0, cos_x, -sin_x],
                      [0, sin_x, cos_x]])

    rotation_matrix = rot_z@rot_y@rot_x

    xyz_rot = rotation_matrix @ xyz
    x = xyz_rot[0, 0]
    y = xyz_rot[1, 0]
    # z = 0  # xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([[Bearing(phi)], [rho]])


def test_simple_radar():

    # Input arguments
    # TODO: pytest parametarization
    noise_covar = CovarianceMatrix([[0.015, 0],
                                   [0, 0.1]])
    radar_position = StateVector([1, 1])
    radar_orientation = StateVector([0, 0, 0])
    target_state = State(radar_position +
                         np.array([[1], [1]]),
                         timestamp=datetime.datetime.now())
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRangeBearing(
        position=radar_position,
        orientation=radar_orientation,
        ndim_state=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=0)
    rho, phi = cart2pol(target_state.state_vector[0, 0]
                        - radar_position[0, 0],
                        target_state.state_vector[1, 0]
                        - radar_position[1, 0])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector,
                    StateVector([phi, rho])).all())


def test_rotating_radar():

    # Input arguments
    # TODO: pytest parametarization
    timestamp = datetime.datetime.now()
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # The radar is positioned at (1,1)
    radar_position = StateVector(
        np.array(([[1], [1]])))
    # The radar is facing left/east
    radar_orientation = StateVector([[0], [0], [np.pi]])
    # The radar antenna is facing opposite the radar orientation
    dwell_center = State(StateVector([[-np.pi]]),
                         timestamp=timestamp)
    rpm = 20            # 20 Rotations Per Minute
    max_range = 100     # Max range of 100m
    fov_angle = np.pi/3       # FOV angle of pi/3

    target_state = State(radar_position +
                         np.array([[5], [5]]),
                         timestamp=timestamp)
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRotatingRangeBearing(
        position=radar_position,
        orientation=radar_orientation,
        ndim_state=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar,
        dwell_center=dwell_center,
        rpm=rpm,
        max_range=max_range,
        fov_angle=fov_angle)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=0)

    # Assert measurement is None since target is not in FOV
    assert(measurement is None)

    # Rotate radar such that the target is in FOV
    timestamp = timestamp + datetime.timedelta(seconds=0.5)
    target_state = State(radar_position +
                         np.array([[5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=0)
    eval_m = h2d(target_state.state_vector,
                 radar.position,
                 radar.orientation+[[0],
                                    [0],
                                    [radar.dwell_center.state_vector[0, 0]]])

    # Assert correction of generated measurement
    assert(measurement.timestamp == target_state.timestamp)
    assert(np.equal(measurement.state_vector, eval_m).all())


def test_raster_scan_radar():
    # Input arguments
    # TODO: pytest parametarization
    timestamp = datetime.datetime.now()
    noise_covar = CovarianceMatrix(np.array([[0.015, 0],
                                             [0, 0.1]]))

    # The radar is positioned at (1,1)
    radar_position = StateVector(
        np.array(([[1], [1]])))
    # The radar is facing left/east
    radar_orientation = StateVector([[0], [0], [np.pi]])
    # The radar antenna is facing opposite the radar orientation
    dwell_center = State(StateVector([[np.pi / 4]]),
                         timestamp=timestamp)
    rpm = 20  # 20 Rotations Per Minute Counter-clockwise
    max_range = 100  # Max range of 100m
    fov_angle = np.pi / 12  # FOV angle of pi/12 (15 degrees)
    for_angle = np.pi + fov_angle  # FOR angle of pi*(13/12) (195 degrees)
    # This will be mean the dwell center will reach at the limits -pi/2 and
    # pi/2. As the edge of the beam will reach the full FOV

    target_state = State(radar_position +
                         np.array([[-5], [5]]),
                         timestamp=timestamp)
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRasterScanRangeBearing(
        position=radar_position,
        orientation=radar_orientation,
        ndim_state=2,
        mapping=measurement_mapping,
        noise_covar=noise_covar,
        dwell_center=dwell_center,
        rpm=rpm,
        max_range=max_range,
        fov_angle=fov_angle,
        for_angle=for_angle)

    # Assert that the object has been correctly initialised
    assert np.array_equal(radar.position, radar_position)

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=0)

    # Assert measurement is None since target is not in FOV
    assert measurement is None

    # Rotate radar
    timestamp = timestamp + datetime.timedelta(seconds=0.5)
    target_state = State(radar_position +
                         np.array([[-5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=0)
    # Assert measurement is None since target is not in FOV
    assert measurement is None

    # Rotate radar such that the target is in FOV
    timestamp = timestamp + datetime.timedelta(seconds=1.0)
    target_state = State(radar_position +
                         np.array([[-5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=0)
    eval_m = h2d(target_state.state_vector,
                 radar.position,
                 radar.orientation + [[0],
                                      [0],
                                      [radar.dwell_center.state_vector[0, 0]]])

    # Assert correction of generated measurement
    assert measurement.timestamp == target_state.timestamp
    assert np.array_equal(measurement.state_vector, eval_m)


def test_aesaradar():
    target = State([75e3, 0, 10e3, 0, 20e3, 0],
                   timestamp=datetime.datetime.now())

    radar = AESARadar(antenna_gain=30,
                      mapping=[0, 2, 4],
                      translation_offset=StateVector([0.0] * 6),
                      frequency=100e6,
                      number_pulses=5,
                      duty_cycle=0.1,
                      band_width=30e6,
                      beam_width=np.deg2rad(10),
                      probability_false_alarm=1e-6,
                      rcs=10,
                      receiver_noise=3,
                      swerling_on=False,
                      beam_shape=Beam2DGaussian(peak_power=50e3),
                      beam_transition_model=StationaryBeam(
                          centre=[np.deg2rad(15), np.deg2rad(20)]),
                      measurement_model=None)

    [prob_detection, snr, swer_rcs, tran_power, spoil_gain,
     spoil_width] = radar.prob_gen(target)
    assert round(swer_rcs, 1) == 10.0
    assert round(prob_detection, 3) == 0.688
    assert round(spoil_width, 2) == 0.19
    assert round(spoil_gain, 2) == 29.58
    assert round(tran_power, 2) == 7715.00
    assert round(snr, 2) == 16.01


def test_swer(repeats=10000):
    # initialise list or rcs (radar cross sections)
    list_rcs = np.zeros(repeats)
    # generic target
    target = State([75e3, 0, 10e3, 0, 20e3, 0],
                   timestamp=datetime.datetime.now())
    # define sensor
    radar = AESARadar(antenna_gain=30,
                      frequency=100e6,
                      number_pulses=5,
                      duty_cycle=0.1,
                      band_width=30e6,
                      beam_width=np.deg2rad(10),
                      probability_false_alarm=1e-6,
                      rcs=10,
                      receiver_noise=3,
                      swerling_on=True,
                      beam_shape=Beam2DGaussian(peak_power=50e3),
                      beam_transition_model=StationaryBeam(
                          centre=[np.deg2rad(15), np.deg2rad(20)]),
                      measurement_model=None)
    # populate list of random rcs
    for i in range(0, repeats):
        list_rcs[i] = radar.prob_gen(target)[2]
    # check histogram follows the Swerling 1 case probability distribution
    bin_height, bin_edge = np.histogram(list_rcs, 20, normed=True)
    x = (bin_edge[:-1] + bin_edge[1:]) / 2
    height = 1 / (float(radar.rcs)) * np.exp(-x / float(radar.rcs))

    assert np.allclose(height, bin_height, rtol=0.05,
                       atol=0.01 * np.max(bin_height))


def test_detection():
    radar = AESARadar(antenna_gain=30,
                      translation_offset=StateVector([0.0] * 3),
                      frequency=100e6,
                      number_pulses=5,
                      duty_cycle=0.1,
                      band_width=30e6,
                      beam_width=np.deg2rad(10),
                      probability_false_alarm=1e-6,
                      rcs=10,
                      receiver_noise=3,
                      swerling_on=False,
                      beam_shape=Beam2DGaussian(peak_power=50e3),
                      beam_transition_model=StationaryBeam(
                          centre=[np.deg2rad(15), np.deg2rad(20)]),
                      measurement_model=LinearGaussian(
                          noise_covar=np.diag([1, 1, 1]),
                          mapping=[0, 1, 2],
                          ndim_state=3))

    target = State([50e3, 10e3, 20e3], timestamp=datetime.datetime.now())
    measurement = radar.gen_measurement(target)

    assert np.allclose(measurement.state_vector,
                       StateVector([50e3, 10e3, 20e3]), atol=5)


def test_failed_detect():
    target = State([75e3, 0, 10e3, 0, 20e3, 0],
                   timestamp=datetime.datetime.now())

    radar = AESARadar(antenna_gain=30,
                      mapping=[0, 2, 4],
                      translation_offset=StateVector([0.0] * 6),
                      frequency=100e6,
                      number_pulses=5,
                      duty_cycle=0.1,
                      band_width=30e6,
                      beam_width=np.deg2rad(10),
                      probability_false_alarm=1e-6,
                      rcs=10,
                      receiver_noise=3,
                      swerling_on=False,
                      beam_shape=Beam2DGaussian(peak_power=50e3),
                      beam_transition_model=StationaryBeam(
                          centre=[np.deg2rad(30), np.deg2rad(40)]),
                      measurement_model=LinearGaussian(
                          noise_covar=np.diag([1, 1, 1]),
                          mapping=[0, 2, 4],
                          ndim_state=6))

    assert radar.gen_measurement(target) is None


def test_target_rcs():
    # targets with the rcs
    rcs_10 = (GroundTruthState([150e3, 0.0, 0.0], timestamp=None))
    rcs_10.rcs = 10
    rcs_20 = (GroundTruthState([250e3, 0.0, 0.0], timestamp=None))
    rcs_20.rcs = 20

    radar_model = AESARadar(antenna_gain=36,
                            mapping=[0, 1, 2],
                            translation_offset=StateVector([0.0]*3),
                            frequency=10e9,
                            number_pulses=10,
                            duty_cycle=0.18,
                            band_width=24591.9,
                            beam_width=np.deg2rad(5),
                            rcs=None, # no default rcs
                            receiver_noise=5,
                            probability_false_alarm=5e-3,
                            beam_shape=Beam2DGaussian(peak_power=1e4),
                            measurement_model=None,
                            beam_transition_model=StationaryBeam(centre=[0,
                                                                         0]))

    (det_prob, snr, swer_rcs, _, _, _) = radar_model.prob_gen(rcs_10)
    assert swer_rcs == 10
    assert round(snr, 3) == 8.197
    (det_prob, snr, swer_rcs, _, _, _) = radar_model.prob_gen(rcs_20)
    assert swer_rcs == 20
    assert round(snr, 3) == 2.125

