#!/usr/bin/env python
# coding: utf-8

"""
==========================================
Example using navigation measurement model
==========================================
"""

# %%
r"""In this example, we present how to perform the tracking task using an inertia
navigation measurement model making use of instruments mounted on the sensor.
This example is relevant for tracking sensors in environments where GPS tracking is not
available and we integrate the information obtained from instruments on board, as the
accelerometer and gyroscope, with fixed target locations, also refereed as landmarks. 
In this example, we simulate a three dimensional sensor, moving in 3D cartesian space,
we have the measurements from on-board instruments that evaluates the Euler angles, whose describe the
sensor rotations and orientation during the flight, as well as the the 3D forces acting on the sensor.
This example aims to provide an idea of how to use the combination of the measurement models
:class:`~.AccelerometerMeasurementModel` and :class:`~.GyroscopeMeasurementModel` to model
the inertia navigation measurements.
In this example we ignore GPS measurements, therefore we employ the knowledge of fixed targets
to adjust the navigation tracking from drifting, a common problem in navigation scenario.
The state space we are considering is a 15 dimensions object, which combines 3D 
nearly-constant Acceleration model and the 3D Euler angles, whose are the heading (
:math:`\psi`), the pitch (:math:`\theta`) and the roll (:math:`\phi`) and their time derivative.
"""
#
# This example follows these points:
# 1. Describe the transition model;
# 2. Obtain the ground truth and measurements;
# 3. Instantiate the tracker components;
# 4. Run the tracker and obtain the final track.
#

# %%
# 1) Describe the transition model
# --------------------------------
# As we have previously said, we want a 15 dimensions transition model for the sensor,
# in the simplest form we can combine :class:`~.ConstantAcceleration` and :class:`~.ConstantVelocity`
# transition model. Since the sensor is moving onto a fixed plane placed 1km above ground, we employ an
# exponential declining acceleration model, using :class:`~.Singer` model, to address
# the z- movements. A more complex, and realistic, approach would involve Van-Loan models for the transition.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta

# %%
# Stone Soup and transition models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.detection import Detection
from stonesoup.types.state import State, StateVector, StateVectors, GaussianState
from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration, Singer
from stonesoup.functions.navigation import getEulersAngles
from stonesoup.types.angle import Angle

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Lets assume a target sensor with these specifics
radius = 5000   # meters
speed = 200     # meters/seconds
center = np.array([0, 0, 1000])  # 3D center placed at 1km in height
n_timesteps = 100

timesteps = np.linspace(0, n_timesteps+1, n_timesteps+1)
simulation_start = datetime.now().replace(microsecond=0)
np.random.seed(2000)  # fix a random seed for reproducibility

# %%
# Describe the ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# In a different manner from other examples, we create the groundtruth of the sensor without considering
# the process noise, and at the same time, we calculate the sensor Euler angles. It is possible to still use the
# existing transition models and extend the state vectors to include such angles.
#

# Create a function to create the groundtruth paths
def describe_sensor_motion(target_speed: float,
                           target_radius: float,
                           starting_position: np.array,
                           start_time: datetime,
                           number_of_timesteps: np.array
                           ) -> (list, set):

    """
        Auxiliary function to create the sensor-target dynamics in the
        specific case of circular motion.

    Parameters:
    -----------
    target_speed: float
        Speed of the sensor;
    target_radius: float
        radius of the circular trajectory;
    starting_position: np.array
        starting point of the trajectory, latitude, longitude
        and altitude;
    start_time: datetime,
        start of the simulation;
    number_of_timesteps: np.array
        simulation length

    Return:
    -------
    (list, set):
        list of timestamps of the simulation and
        groundtruths path.
    """

    # Instantiate the 15 dimension object describing
    # the positions, dynamics and angles of the target
    sensor_dynamics = np.zeros((15))

    # Generate the GroundTruthPath
    truths = GroundTruthPath([])

    # instantiate a list for the timestamps
    timestamps = []

    # indexes of the array
    position_indexes = [0, 3, 6]
    velocity_indexes = [1, 4, 7]
    acceleration_indexes = [2, 5, 8]
    angles_indexes = [9, 11, 13]
    vangles_indexes = [10, 12, 14]

    sensor_dynamics[angles_indexes]

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
        sensor_dynamics[acceleration_indexes] += \
            ((-target_speed * target_speed) / target_radius) * np.array(
                [np.cos(theta), np.sin(theta), 0])

        # Now using the velocity and accelerations terms we get the Euler angles
        angles, dangles = getEulersAngles(sensor_dynamics[velocity_indexes],
                                          sensor_dynamics[acceleration_indexes])

        # add the Euler angles and their time derivative
        # please check that are all angles
        sensor_dynamics[angles_indexes] += angles
        sensor_dynamics[vangles_indexes] += dangles

        # append all those as ground state
        truths.append(GroundTruthState(state_vector=sensor_dynamics,
                                       timestamp=start_time + timedelta(seconds=int(i))))
        # restart the array
        sensor_dynamics = np.zeros((15))
        timestamps.append(start_time + timedelta(seconds=int(i)))

    return (timestamps, truths)


# Instantiate the transition model, We consider the Singer model for
# an exponential declining acceleration in the z- coordinate.
transition_model = CombinedGaussianTransitionModel([ConstantAcceleration(1.5),
                                                    ConstantAcceleration(1.5),
                                                    Singer(0.1, 10),
                                                    ConstantVelocity(0),
                                                    ConstantVelocity(0),
                                                    ConstantVelocity(0)
                                                    ])

# %%
# 2) Obtain the ground truth and gather the measurements;
# -------------------------------------------------------
# We have instantiated a function to describe the # target-sensor dynamics, obtaining the Euler angles
# from the vessel acceleration and velocity # adopting the ad-hoc function :class:`~.getEulerAngles`.
# Likewise, we have instantiated the 15 dimension transition # model using a constant acceleration
# model for the 3D dynamics and a constant velocity for modelling the Euler angles
# dynamics. We consider as well the :class:`~.Singer` model for an exponential declining acceleration
# model for the z-coordinate, since the sensor is moving on a fixed plane at 1 km above the surface.
# At this stage we can start collecting both the groundtruths and # the measurement using a composite
# measurement model merging the measurements from the :class:`~.AccelerometerMeasurementModel`,
# the :class:`~.GyroscopeMeasurementModel` and the landmarks, using an
# :class:`~.CartesianAzimuthElevationMeasurementModel`.
# This measurement model combines the specific forces measured by the accelerometer instrument
# and the angular rotation from the inertia movements of the target. The landmarks helps reducing the
# navigation drift.
#

# %%
# Get the ground truth paths
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
timestamps, truths = describe_sensor_motion(speed,
                                            radius,
                                            center,
                                            simulation_start,
                                            timesteps)

# %%
# Load and instantiate the measurement model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We consider a case with the three fixed targets, landmarks, and we use the on-board
# measurements. To merge all these measurements # we employ a :class:`~.CombinedReversibleGaussianMeasurementModel`
# to concatenate all the different measurement models.
# We specify a reference frame to evaluate the gravity forces applied onto the sensor, and it is needed for the
# accelerometer and gyroscope measurements. The landmarks are placed on the ground (z~0).
# Overall the measurement model will have 14 dimensions space.
#

from stonesoup.models.measurement.nonlinear import AccelerometerMeasurementModel, \
    GyroscopeMeasurementModel, CartesianAzimuthElevationMeasurementModel, \
    CombinedReversibleGaussianMeasurementModel

# Instantiate the measurement model
measurement_model_list = []

# Instantiate the landmarks - the z-coordinate is randomly drawn
target1 = np.array([3000, 3000, 0.0096])
target2 = np.array([-3000, 3000, 1.6034])
target3 = np.array([0, -3000, 0.93])
target4 = np.array([0, 0, 1.5])

targets = [target1, target2, target3, target4]

# Specify the reference frame for the Accelerometer
# and Gyroscope measurements.
reference_frame = StateVector([55, 0, 0])  # Latitude, longitude, Altitude

accelerometer = AccelerometerMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag([1, 1, 5]),  # Acceleration
    reference_frame=reference_frame
)

gyroscope = GyroscopeMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag([1e-7, 1e-7, 1e-7]),  # Gyroscope
    reference_frame=reference_frame
)

# add the measurements models
measurement_model_list.append(accelerometer)
measurement_model_list.append(gyroscope)

# loop over the various targets to initialise the
# azimuth-elevation models.
for target in targets:
    measurement_model_list.append(
        CartesianAzimuthElevationMeasurementModel(
            ndim_state=15,
            mapping=(0, 3, 6),
            noise_covar=np.diag([1, 1]),
            target_location=StateVector(target),
            translation_offset=None)
    )

# Combine all the measurement model into a unique
# model
measurement_model = CombinedReversibleGaussianMeasurementModel(measurement_model_list)

# Now create the measurements
measurement_set = []

for truth in truths:
    measurement = measurement_model.function(truth, noise=True)
    measurement_set.append(Detection(state_vector=measurement,
                                     timestamp=truth.timestamp,
                                     measurement_model=measurement_model))


# %%
# 3) instantiate the tracker components;
# --------------------------------------
# We have the truths and the detections, in this simple example we do not include measurement clutter.
# Now we can set up the tracker components.
# In this example we consider an UnscentedKalmanFilter given the non-linearity of the problem.

# %%
# Load the filter components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(None)

# Covariance of the starting location
covar_starting_position = np.repeat(10, 15)

# Instantiate the prior, with a known location of the sensor
prior = GaussianState(
    state_vector=truths[0].state_vector,
    covar=np.diag(covar_starting_position),
    timestamp=timestamps[0]
)

# %%
# 4) Run the tracker and obtain the final track.
# ----------------------------------------------
# We have the tracker components and the starting (prior) knowledge, now we can loop over the
# various measurements and using a :class:`~.SingleHypothesis` we can perform the tracking.
#

# Load these components to do the tracking
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

track = Track()

# Loop over the measurement
for k, measurement in enumerate(measurement_set):
    predictions = predictor.predict(prior, timestamp=measurement.timestamp)
    hyps = SingleHypothesis(predictions, measurement)
    post = updater.update(hyps)
    track.append(post)
    prior = track[-1]

# %%
# Load the plotter
# ^^^^^^^^^^^^^^^^
# To plot the various landmarks we make use of the fixed platform object.

from stonesoup.platform.base import FixedPlatform

platforms = []
for target in targets:
    state = np.array([target[0], 0,
                      target[1], 0,
                      target[2], 0])
    platforms.append(
        FixedPlatform(
            states=GaussianState(state,
                                 np.diag([1, 1, 1, 1, 1, 1])
                                 ),
            position_mapping=(0, 2, 4)
            ))

from stonesoup.plotter import Plotter, Dimension

plotter = Plotter(dimension=Dimension.THREE)

plotter.plot_ground_truths(truths, mapping=[0, 3, 6])
plotter.plot_sensors({*platforms}, mapping=[0, 1, 2],
                     sensor_label='Landmarks')
plotter.plot_tracks(track, mapping=[0, 3, 6], uncertainty=False, track_label='Track')

plotter.fig

# %%
# Conclusion
# ----------
# In this example we have shown how to use the inertia navigation functions and how to integrate the tracking using
# fixed landmarks. As it is evident from the tracking result this scenario is particularly complex
# and it is not possible to run a perfect track with the limited information available.
# Using different measurements for the landmarks, e.g. including the range between the target and sensor
# (i.e., see :class:`~.CartesianAzimuthElevationRangeMeasurementModel`),
# would improve the tracking. However this example aims to give an opportunity to show how to perform tracking
# in the inertia navigation context.
#
