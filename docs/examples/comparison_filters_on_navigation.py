#!/usr/bin/env python
# coding: utf-8

"""
====================================================================
Comparing different tracking algorithm using navigation measurements
====================================================================
"""

# %%
# This example compares the performances of various filters in tracking objects with
# navigation-like measurements models. We are interested in this scenario to show how we can use
# measurements models in the navigation context, and how different tracking algorithms perform.
#
# In an different example, we have explained how to set up the problem involving Euler angles and
# forces acting onto a sensor, using measurement models components coming from
# :class:`~.AccelerometerMeasurementModel`, :class:`~.GyroscopeMeasurementModel` and fixed targets, landmarks,
# in Stone soup.
# This example will show the performances in a 1-to-1 comparison using Extended Kalman filter (EKF),
# Unscented Kalman Filter (UKF) and Particle filter (PF) in a single target-sensor scenario.
#
# This example follows this schema:
# 1. Instantiate the target-sensor ground truths and gather the measurements;
# 2. Prepare and load the various filters components;
# 3. Run the trackers and obtain the tracks;
# 4. Create and visualise the performances of the tracking algorithms.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, \
    ConstantAcceleration, ConstantVelocity, Singer, CombinedLinearGaussianTransitionModel
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.functions.navigation import getEulersAngles


# Simulation parameters
np.random.seed(2010) # fix a random seed
simulation_steps = 100
timesteps = np.linspace(1, simulation_steps+1, simulation_steps+1)
start_time = datetime.now().replace(microsecond=0)
# Lets assume a sensor with these specifics
radius = 5000
speed = 200
center = np.array([0, 0, 1000]) # latitude, longitude, altitutde (meters)

# %%
# 1) Instantiate the target ground truth path
# -------------------------------------------
# For this example we consider a different approach for describing the target ground truth.
# We evaluate the sensor motion on a circular  trajectory by modelling simply the 3D movements
# and measure the Euler angles associated with  the object direction.
#

from stonesoup.types.detection import TrueDetection

# Create a function to create the groundtruth paths
def describe_sensor_motion(target_speed: float,
                           target_radius: float,
                           starting_position: np.array,
                           start_time: datetime,
                           number_of_timesteps: np.array
                           ) -> (list, set):

    """
        Auxuliary function to create the sensor-target dynamics in the
        specific case of circular motion.

    Parameters:
    -----------
    target_speed: float
        Speed of the sensor;
    target_tadius: float
        radius of the circular trajectory;
    starting_position: np.array
        starting point of the trajectory, latitude, longitude
        and altitude;
    start_time: datetime,
        start of the simulation;
    number_of_timesteps: np.array
        simulation lenght

    Return:
    -------
    (list, set):
        list of timestamps of the simulation and
        groundtruths path.
    """

    # Instantiate the 15 dimension object describing
    # the positions, dynamics and angles of the target
    sensor_dynamics = np.zeros((15))

    # Generate the groundTruthpath
    truths = GroundTruthPath([])

    # instantiate a list for the timestamps
    timestamps = []

    # indexes of the array
    position_indexes = [0, 3, 6]
    velocity_indexes = [1, 4, 7]
    acceleration_indexes = [2, 5, 8]
    angles_indexes = [9, 11, 13]
    vangles_indexes = [10, 12, 14]

    # loop over the timestep
    for i in number_of_timesteps:
        theta = target_speed * i / target_radius + 0

        # positions
        sensor_dynamics[position_indexes] += target_radius * \
                                      np.array([np.cos(theta), np.sin(theta),
                                                0.001*np.random.choice(np.arange(-5, 5), 1)[0]]) + \
                                      starting_position

        # velocities
        sensor_dynamics[velocity_indexes] += target_speed * \
                                      np.array([-np.sin(theta), np.cos(theta), 0])

        # acceleration
        sensor_dynamics[acceleration_indexes] += ((-target_speed * target_speed) / target_radius) * \
                           np.array([np.cos(theta), np.sin(theta), 0])

        # Now using the velocity and accelerations terms we get the Euler angles
        angles, dangles = getEulersAngles(sensor_dynamics[velocity_indexes],
                                          sensor_dynamics[acceleration_indexes])

        # add the Euler angles and their time derivative
        # please check that are all angles
        sensor_dynamics[angles_indexes] += angles
        sensor_dynamics[vangles_indexes] += dangles

        # append all those as ground state
        truths.append(GroundTruthState(state_vector=sensor_dynamics,
                                       timestamp=start_time +
                                                 timedelta(seconds=int(i))))
        # restart the array
        sensor_dynamics = np.zeros((15))
        timestamps.append(start_time + timedelta(seconds=int(i)))

    return (timestamps, truths)


# Instantiate the transition model, We consider the Singer model for
# an exponential declining acceleration in the z-coordinate.
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantAcceleration(1.5),
    ConstantAcceleration(1.5),
    Singer(0.1, 10),
    ConstantVelocity(0),
    ConstantVelocity(0),
    ConstantVelocity(0)
    ])


# Generate the ground truths
timestamps, groundtruths = describe_sensor_motion(speed, radius, center, start_time,
                                                  timesteps)
# %%
# Load the measurement model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# As for the other example, we compose our measurement model
# using the accelearation, gyroscope and landmarks measurement models.
# We for the landmarks we consider a :class:`~.CartesianAzimuthElevationRangeMeasurementModel`
# which provides the Azimuth, Elevation and Range between the
# sensor and the fixed target to ease the tracking efficiency.
#

from stonesoup.models.measurement.nonlinear import AccelerometerMeasurementModel, GyroscopeMeasurementModel, \
    CartesianAzimuthElevationRangeMeasurementModel, CombinedReversibleGaussianMeasurementModel

# Instantiate the measurement model
measurement_model_list = []

# Instantiate the landmarks - the z-coordinate is randomly drawn
target1 = np.array([3000, 3000, 0.0096])
target2 = np.array([-3000, 3000, 1.6034])
target3 = np.array([0, -3000, 0.93])
targets = [target1, target2, target3]

# Instantantiate a reference frame for the Gravity
# forces
reference_frame = StateVector([55, 0, 0])  # Latitude, longitude, Altitude

# Model list
measurement_model_list = []

accelerometer = AccelerometerMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag([1, 1, 10]),  # Acceleration
    reference_frame=reference_frame
)

gyroscope = GyroscopeMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag([1e-5, 1e-5, 1e-5]),  # Gyroscope, noise in micro radiands
    reference_frame=reference_frame
)

# add the measurements models
measurement_model_list.append(accelerometer)
measurement_model_list.append(gyroscope)

# loop over the various targets to initilise the
# azimuth-elevation-range models.
for target in targets:
    measurement_model_list.append(
        CartesianAzimuthElevationRangeMeasurementModel(
            ndim_state=15,
            mapping=(0, 3, 6),
            noise_covar=np.diag([1, 1, 10]),
            target_location=StateVector(target),
            translation_offset=None)
    )


measurement_model = CombinedReversibleGaussianMeasurementModel(measurement_model_list)
measurements_set = []

# Now create the measurements
for truth in groundtruths:
    measurement = measurement_model.function(truth, noise=True)
    measurements_set.append(Detection(state_vector=measurement,
                                      timestamp=truth.timestamp,
                                      measurement_model=measurement_model))
# %%
# 2) Prepare and load the various filters components;
# ---------------------------------------------------
# So far we have generated the sensor original track and
# we have gathered the measurements using the :class:`~.CombinedReversibleGaussianMeasurementModel`.
# We can now load the various filter components for the EKF, UKF and PF.
# Then, we need to instantiate the components as the prior and the tracks.

# Load the Kalman components
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor, ExtendedKalmanPredictor

# Load the Particle filter components
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler

# Extended Kalman filter
EKF_predictor = ExtendedKalmanPredictor(transition_model)
EKF_updater = ExtendedKalmanUpdater(measurement_model=None)

# Unscented Kalman filter
UKF_predictor = UnscentedKalmanPredictor(transition_model)
UKF_updater = UnscentedKalmanUpdater(measurement_model=None)

# Particle filter
PF_predictor = ParticlePredictor(transition_model)
resampler = ESSResampler()
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)

# Create a starting covarinace
covar_starting_position = np.repeat(10, 15)

# Instantiate the prior, with a known location.
prior = GaussianState(
    state_vector=groundtruths[0].state_vector,
    covar=np.diag(covar_starting_position),
    timestamp=timestamps[0])

# instantate the PF prior
from stonesoup.types.state import ParticleState
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle

# Number of particles
number_particles=1024

samples = multivariate_normal.rvs(
    np.array(prior.state_vector).reshape(-1),
    np.diag([10, 0.1, 0.1, 10, 0.1, 0.1, 10, 0.1, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]),
    size=number_particles)

particles = [Particle(sample.reshape(-1, 1),
                      weight=Probability(1.))
             for sample in samples]

# Particle prior
particle_prior = ParticleState(state_vector=None,
                               particle_list=particles,
                               timestamp=timestamps[0])

from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

# Instantiate the various tracks
track_ukf, track_ekf, track_pf = Track(), Track(), Track()

# Loop over the measurement
updaters = [UKF_updater, UKF_updater, PF_updater]
predictors = [UKF_predictor, EKF_predictor, PF_predictor]
tracks = [track_ukf, track_ekf, track_pf]
priors = [prior, prior, particle_prior]

# %%
# 3) Run the trackers and obtain the tracks;
# ------------------------------------------
# We can run the various trackers and generate some tracks.
# Then, we evaluate the tracking results and perform a 1-to-1 comparison on the accuracy of the
# tracking algorithms using the RMSE.
#

# Loop over the various trackers
for predictor, updater, track, prior in zip(predictors, updaters, tracks, priors):
    for k, measurement in enumerate(measurements_set):
        predictions = predictor.predict(prior, timestamp=measurement.timestamp)
        hyps = SingleHypothesis(predictions, measurement)
        post = updater.update(hyps)
        track.append(post)
        prior = track[-1]

# Add the landmarks as fixed platforms
from stonesoup.platform.base import FixedPlatform

platforms = []
for target in targets:
    state = np.array([target[0], 0,
                      target[1], 0,
                      target[2], 0])
    platforms.append(
        FixedPlatform(
        states=GaussianState(state,
                             np.diag([1,1,1,1,1,1])
                             ),
        position_mapping=(0, 2, 4)
        ))

from stonesoup.plotter import Plotter, Dimension

plotter = Plotter(dimension=Dimension.THREE)
# Visualise with the landmarks
plotter.plot_ground_truths(groundtruths, mapping=[0, 3, 6])
plotter.plot_tracks(track_ukf, mapping=[0, 3, 6], track_label='UKF')
plotter.plot_tracks(track_ekf, mapping=[0, 3, 6], track_label='EKF')
plotter.plot_tracks(track_pf, mapping=[0, 3, 6], track_label='PF')
plotter.plot_sensors({*platforms}, mapping=[0, 1, 2],
                     sensor_label='Landmarks')
plotter.fig

# %%
# 4) Create and visualise the performances of the tracking algorithms
# -------------------------------------------------------------------
# We have all the components from the tracking and now we can measure how well the
# various filter perform. We consider the RMSE (root mean square error) between the tracks and the
# groundtruth over the various simulation timesteps.
#

pf_track, ekf_track, ukf_track, = np.zeros((1, simulation_steps)), \
                                  np.zeros((1, simulation_steps)), \
                                  np.zeros((1, simulation_steps))

for j in range(simulation_steps):
    pf_track[:, j] = np.sqrt((track_pf[j].state_vector[0].mean() - groundtruths[j].state_vector[0])**2. +
                             (track_pf[j].state_vector[3].mean() - groundtruths[j].state_vector[3])**2. +
                             (track_pf[j].state_vector[6].mean() - groundtruths[j].state_vector[6])**2.)

    ekf_track[:, j] = np.sqrt((track_ekf[j].state_vector[0] - groundtruths[j].state_vector[0])**2. +
                             (track_ekf[j].state_vector[3] - groundtruths[j].state_vector[3])**2. +
                             (track_ekf[j].state_vector[6] - groundtruths[j].state_vector[6])**2.)

    ukf_track[:, j] = np.sqrt((track_ukf[j].state_vector[0] - groundtruths[j].state_vector[0])**2. +
                             (track_ukf[j].state_vector[3] - groundtruths[j].state_vector[3])**2. +
                             (track_ukf[j].state_vector[6] - groundtruths[j].state_vector[6])**2.)

plt.plot(timesteps[0:simulation_steps], pf_track[0,:], color='red', linestyle='--', label='PF track')
plt.plot(timesteps[0:simulation_steps], ukf_track[0,:], color='blue', linestyle='--', label='UKF track')
plt.plot(timesteps[0:simulation_steps], ekf_track[0,:], color='orange', linestyle='--', label='EKF track')
plt.xlabel('Timesteps')
plt.ylabel('RMSE')
plt.title('Positional accuracy')
plt.legend()
plt.show()

# %%
# Conclusion
# ----------
# In this example we have compared the performances of some tracking algorithms available in
# Stone Soup providing measurements from the accelerometer and gyroscope on board of a
# sensor and using some landmarks on the ground to adjust the tracking. By adressing the RMSE (root mean
# square error) of the tracks obtained we can see that the Kalman Filter based algorithms offer good performances
# over the simulation, while the Particle filter seems to suffer from the high dimensionality of the
# problem. Overall, the intent of this example was to show how to use and perform a 1-to-1 comparison
# between these algorithms in Stone Soup.
#
