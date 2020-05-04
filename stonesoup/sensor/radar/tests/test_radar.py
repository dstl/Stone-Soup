# -*- coding: utf-8 -*-
import datetime
import pytest
from pytest import approx
import numpy as np

from ....functions import rotz, rotx, roty, cart2sphere
from ....types.angle import Bearing, Elevation
from ....types.array import StateVector, CovarianceMatrix
from ....types.state import State
from ....types.groundtruth import GroundTruthState
from ..radar import RadarRangeBearing, RadarRangeBearingElevation, RadarRotatingRangeBearing, \
    AESARadar, RadarRasterScanRangeBearing, RadarRangeRateBearing, RadarRangeRateBearingElevation
from ..beam_pattern import StationaryBeam
from ..beam_shape import Beam2DGaussian
from ....models.measurement.linear import LinearGaussian


def h2d(state, translation_offset, rotation_offset):

    xyz = [[state.state_vector[0, 0] - translation_offset[0, 0]],
           [state.state_vector[1, 0] - translation_offset[1, 0]],
           [0]]

    # Get rotation matrix
    theta_z = -rotation_offset[2, 0]
    theta_y = -rotation_offset[1, 0]
    theta_x = -rotation_offset[0, 0]

    rotation_matrix = rotz(theta_z)@roty(theta_y)@rotx(theta_x)

    xyz_rot = rotation_matrix @ xyz
    x = xyz_rot[0, 0]
    y = xyz_rot[1, 0]
    # z = 0  # xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([[Bearing(phi)], [rho]])


def h3d(state, translation_offset, rotation_offset):

    xyz = np.array([[state.state_vector[0, 0] - translation_offset[0, 0]],
                    [state.state_vector[1, 0] - translation_offset[1, 0]],
                    [state.state_vector[2, 0] - translation_offset[2, 0]]])

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
    z = xyz_rot[2, 0]

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)

    return np.array([[Elevation(theta)], [Bearing(phi)], [rho]])


@pytest.mark.parametrize(
    "h, sensorclass, ndim_state, pos_mapping, noise_covar, position, target",
    [
        (
            h2d,  # h
            RadarRangeBearing,  # sensorclass
            2,
            np.array([0, 1]),  # pos_mapping
            np.array([[0.015, 0],
                      [0, 0.1]]),  # noise_covar
            StateVector([[1], [1]]),  # position
            np.array([[200], [10]])  # target
         ),
        (
            h3d,  # h
            RadarRangeBearingElevation,  # sensorclass
            3,
            np.array([0, 1, 2]),  # pos_mapping
            np.array([[0.015, 0, 0],
                      [0, 0.015, 0],
                      [0, 0, 0.1]]),  # noise_covar
            StateVector([[1], [1], [0]]),  # position
            np.array([[200], [10], [10]])  # target
        )
    ],
    ids=["RadarRangeBearing", "RadarRangeBearingElevation"]
)
def test_simple_radar(h, sensorclass, ndim_state, pos_mapping, noise_covar, position, target):
    # Instantiate the rotating radar
    radar = sensorclass(ndim_state=ndim_state,
                        position_mapping=pos_mapping,
                        noise_covar=noise_covar,
                        position=position)

    assert (np.equal(radar.position, position).all())

    target_state = State(target, timestamp=datetime.datetime.now())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=False)

    # Assert correction of generated measurement
    assert (measurement.timestamp == target_state.timestamp)
    assert (np.equal(measurement.state_vector, h(target_state,
                                                 translation_offset=position,
                                                 rotation_offset=radar.orientation)).all())


def h2d_rr(state, pos_map, vel_map, translation_offset, rotation_offset, velocity):

    xyz = np.array([[state.state_vector[pos_map[0], 0] - translation_offset[0, 0]],
                    [state.state_vector[pos_map[1], 0] - translation_offset[1, 0]],
                    [0]])

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    theta_y = - rotation_offset[1, 0]
    theta_x = - rotation_offset[0, 0]

    rotation_matrix = rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, _ = cart2sphere(*xyz_rot[:, 0])

    # Calculate range rate extension
    # Determine the net velocity component in the engagement
    xyz_vel = np.array([[state.state_vector[vel_map[0], 0] - velocity[0, 0]],
                        [state.state_vector[vel_map[1], 0] - velocity[1, 0]],
                        [0]])

    # Use polar to calculate range rate
    rr = np.dot(xyz[:, 0], xyz_vel[:, 0]) / np.linalg.norm(xyz)

    return np.array([[Bearing(phi)], [rho], [rr]])


def h3d_rr(state, pos_map, vel_map, translation_offset, rotation_offset, velocity):

    xyz = state.state_vector[pos_map, :] - translation_offset

    # Get rotation matrix
    theta_z = - rotation_offset[2, 0]
    theta_y = - rotation_offset[1, 0]
    theta_x = - rotation_offset[0, 0]

    rotation_matrix = rotz(theta_z)@roty(theta_y)@rotx(theta_x)
    xyz_rot = rotation_matrix @ xyz

    rho, phi, theta = cart2sphere(*xyz_rot[:, 0])

    # Calculate range rate extension
    # Determine the net velocity component in the engagement
    xyz_vel = state.state_vector[vel_map, :] - velocity

    # Use polar to calculate range rate
    rr = np.dot(xyz[:, 0], xyz_vel[:, 0]) / np.linalg.norm(xyz)

    return np.array([[Elevation(theta)], [Bearing(phi)], [rho], [rr]])


@pytest.mark.parametrize(
    "h, sensorclass, pos_mapping, vel_mapping, noise_covar, position",
    [
        (
            h2d_rr,  # h
            RadarRangeRateBearing,  # sensorclass
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            np.array([[0.05, 0, 0],
                      [0, 0.015, 0],
                      [0, 0, 10]]),  # noise_covar
            StateVector([[100], [0], [0]])  # position
         ),
        (
            h3d_rr,
            RadarRangeRateBearingElevation,
            np.array([0, 2, 4]),  # pos_mapping
            np.array([1, 3, 5]),  # vel_mapping
            np.array([[0.05, 0, 0, 0],
                      [0, 0.05, 0, 0],
                      [0, 0, 0.015, 0],
                      [0, 0, 0, 10]]),  # noise_covar
            StateVector([[100], [0], [0]])  # position
        )
    ],
    ids=["RadarRangeRateBearing", "RadarRangeRateBearingElevation"]
)
def test_range_rate_radar(h, sensorclass, pos_mapping, vel_mapping, noise_covar, position):

    # Instantiate the rotating radar
    radar = sensorclass(ndim_state=6,
                        position_mapping=pos_mapping,
                        velocity_mapping=vel_mapping,
                        noise_covar=noise_covar,
                        position=position)

    assert (np.equal(radar.position, position).all())

    target_state = State(np.array([[200], [10], [0], [0], [0], [0]]),
                         timestamp=datetime.datetime.now())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=False)

    # Assert correction of generated measurement
    assert (measurement.timestamp == target_state.timestamp)
    assert (np.equal(measurement.state_vector, h(target_state,
                                                 pos_map=pos_mapping,
                                                 vel_map=vel_mapping,
                                                 translation_offset=position,
                                                 rotation_offset=radar.orientation,
                                                 velocity=radar.velocity)).all())


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
    radar = RadarRotatingRangeBearing(position=radar_position,
                                      orientation=radar_orientation,
                                      ndim_state=2,
                                      position_mapping=measurement_mapping,
                                      noise_covar=noise_covar,
                                      dwell_center=dwell_center,
                                      rpm=rpm,
                                      max_range=max_range,
                                      fov_angle=fov_angle)

    # Assert that the object has been correctly initialised
    assert(np.equal(radar.position, radar_position).all())

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=False)

    # Assert measurement is None since target is not in FOV
    assert(measurement is None)

    # Rotate radar such that the target is in FOV
    timestamp = timestamp + datetime.timedelta(seconds=0.5)
    target_state = State(radar_position +
                         np.array([[5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=False)
    eval_m = h2d(target_state,
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
    dwell_center = State(StateVector([[np.pi / 4]]), timestamp=timestamp)
    rpm = 20  # 20 Rotations Per Minute Counter-clockwise
    max_range = 100  # Max range of 100m
    fov_angle = np.pi / 12  # FOV angle of pi/12 (15 degrees)
    for_angle = np.pi + fov_angle  # FOR angle of pi*(13/12) (195 degrees)
    # This will be mean the dwell center will reach at the limits -pi/2 and
    # pi/2. As the edge of the beam will reach the full FOV

    target_state = State(radar_position + np.array([[-5], [5]]), timestamp=timestamp)
    measurement_mapping = np.array([0, 1])

    # Create a radar object
    radar = RadarRasterScanRangeBearing(position=radar_position,
                                        orientation=radar_orientation,
                                        ndim_state=2,
                                        position_mapping=measurement_mapping,
                                        noise_covar=noise_covar,
                                        dwell_center=dwell_center,
                                        rpm=rpm,
                                        max_range=max_range,
                                        fov_angle=fov_angle,
                                        for_angle=for_angle)

    # Assert that the object has been correctly initialised
    assert np.array_equal(radar.position, radar_position)

    # Generate a noiseless measurement for the given target
    measurement = radar.measure(target_state, noise=False)

    # Assert measurement is None since target is not in FOV
    assert measurement is None

    # Rotate radar
    timestamp = timestamp + datetime.timedelta(seconds=0.5)
    target_state = State(radar_position +
                         np.array([[-5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=False)
    # Assert measurement is None since target is not in FOV
    assert measurement is None

    # Rotate radar such that the target is in FOV
    timestamp = timestamp + datetime.timedelta(seconds=1.0)
    target_state = State(radar_position +
                         np.array([[-5], [5]]),
                         timestamp=timestamp)
    measurement = radar.measure(target_state, noise=False)
    eval_m = h2d(target_state,
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
                      position_mapping=[0, 2, 4],
                      position=StateVector([0.0] * 3),
                      orientation=StateVector([0.0] * 3),
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
     spoil_width] = radar.gen_probability(target)
    assert approx(swer_rcs, 1) == 10.0
    assert approx(prob_detection, 3) == 0.688
    assert approx(spoil_width, 2) == 0.19
    assert approx(spoil_gain, 2) == 29.58
    assert approx(tran_power, 2) == 7715.00
    assert approx(snr, 2) == 16.01


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
                      measurement_model=None,
                      position=StateVector([0.0]*3),
                      orientation=StateVector([0.0]*3))
    # populate list of random rcs
    for i in range(0, repeats):
        list_rcs[i] = radar.gen_probability(target)[2]
    # check histogram follows the Swerling 1 case probability distribution
    bin_height, bin_edge = np.histogram(list_rcs, 20, density=True)
    x = (bin_edge[:-1] + bin_edge[1:]) / 2
    height = 1 / (float(radar.rcs)) * np.exp(-x / float(radar.rcs))

    assert np.allclose(height, bin_height, rtol=0.03,
                       atol=0.05 * np.max(bin_height))


def test_detection():
    radar = AESARadar(antenna_gain=30,
                      position=StateVector([0.0] * 3),
                      orientation=StateVector([0.0] * 3),
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
    measurement = radar.measure(target)

    assert np.allclose(measurement.state_vector,
                       StateVector([50e3, 10e3, 20e3]), atol=5)


def test_failed_detect():
    target = State([75e3, 0, 10e3, 0, 20e3, 0],
                   timestamp=datetime.datetime.now())

    radar = AESARadar(antenna_gain=30,
                      position_mapping=[0, 2, 4],
                      position=StateVector([0.0] * 3),
                      orientation=StateVector([0.0] * 3),
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
                      measurement_model=LinearGaussian(noise_covar=np.diag([1, 1, 1]),
                                                       mapping=[0, 2, 4],
                                                       ndim_state=6))

    assert radar.measure(target) is None


def test_target_rcs():
    # targets with the rcs
    rcs_10 = (GroundTruthState([150e3, 0.0, 0.0], timestamp=None))
    rcs_10.rcs = 10
    rcs_20 = (GroundTruthState([250e3, 0.0, 0.0], timestamp=None))
    rcs_20.rcs = 20

    radar = AESARadar(antenna_gain=36,
                      position_mapping=[0, 1, 2],
                      position=StateVector([0.0]*3),
                      orientation=StateVector([0.0] * 3),
                      frequency=10e9,
                      number_pulses=10,
                      duty_cycle=0.18,
                      band_width=24591.9,
                      beam_width=np.deg2rad(5),
                      rcs=None,  # no default rcs
                      receiver_noise=5,
                      probability_false_alarm=5e-3,
                      beam_shape=Beam2DGaussian(peak_power=1e4),
                      measurement_model=None,
                      beam_transition_model=StationaryBeam(centre=[0, 0]))

    (det_prob, snr, swer_rcs, _, _, _) = radar.gen_probability(rcs_10)
    assert swer_rcs == 10
    assert approx(snr, 3) == 8.197
    (det_prob, snr, swer_rcs, _, _, _) = radar.gen_probability(rcs_20)
    assert swer_rcs == 20
    assert round(snr, 3) == 2.125
