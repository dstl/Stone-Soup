#!/usr/bin/env python

"""
Multi-Sensor Moving Platform Simulation Example
===============================================
This example looks at how multiple sensors can be mounted on a single moving platform and
exploiting a defined moving platform as a sensor target.
"""

# %%
# Building a Simulated Multi-Sensor Moving Platform
# -------------------------------------------------
# This example shows how to set-up and configure a simulation environment
# to provide a multi-sensor moving platform, as such the application of a tracker will not be
# covered in detail. For more information about trackers and how to configure them review of the
# tutorials and demonstrations is recommended.
#
# This example makes use of Stone Soup :class:`~.MovingPlatform`,
# :class:`~.MultiTransitionMovingPlatform` and :class:`~.Sensor` objects.
#
# In order to configure platforms, sensors, and the simulation, we will need to import some
# specific Stone Soup objects. As these have been introduced in previous tutorials, they are
# imported upfront. New functionality within this example will be imported at the relevant point
# to draw attention to the new features.

# Some general imports and set-up
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt

import numpy as np

# Stone Soup imports:
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.array import CovarianceMatrix
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.tracker.simple import SingleTargetTracker

# Define the simulation start time
start_time = datetime.now()

# %%
# Create a multi-sensor platform
# ------------------------------
# We have previously demonstrated how to create a :class:`~.FixedPlatform` which exploited a
# :class:`~.RadarRangeBearingElevation` *Sensor* in order to detect and track targets generated
# within a :class:`~.MultiTargetGroundTruthSimulator`.
#
# In this example, we are going to create a moving platform which will be mounted with a pair
# of sensors and moves within a 6 dimensional state space according to the
# following :math:`\mathbf{x}`.
#
# .. math::
#           \mathbf{x} = \begin{bmatrix}
#                          x\\ \dot{x}\\ y\\ \dot{y}\\ z\\ \dot{z} \end{bmatrix}
#                      = \begin{bmatrix}
#                          0\\ 0\\ 0\\ 50\\ 8000\\ 0 \end{bmatrix}
#
# The platform will be initiated with a near-constant velocity model which has been
# parameterised to have zero noise. Therefore, the platform location at time
# :math:`k` is given by :math:`F_{k}x_{k-1}` where :math:`F_{k}` is given by:
#
# .. math::
#           F_{k} = \begin{bmatrix}
#            1 & \triangle k & 0 & 0 & 0 & 0\\
#            0 & 1 & 0 & 0 & 0 & 0\\
#            0 & 0 & 1 & \triangle k & 0 & 0\\
#            0 & 0 & 0 & 1 & 0 & 0\\
#            0 & 0 & 0 & 0 & 1 & \triangle k \\
#            0 & 0 & 0 & 0 & 0 & 1\\
#              \end{bmatrix}

# First import the Moving platform
from stonesoup.platform.base import MovingPlatform

# Define the initial platform position, in this case the origin
initial_loc = StateVector([[0], [0], [0], [50], [8000], [0]])
initial_state = State(initial_loc, start_time)

# Define transition model and position for 3D platform
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.), ConstantVelocity(0.), ConstantVelocity(0.)])

# create our fixed platform
sensor_platform = MovingPlatform(states=initial_state,
                                 position_mapping=(0, 2, 4),
                                 velocity_mapping=(1, 3, 5),
                                 transition_model=transition_model)

# %%
# With our platform generated, we now need to build a set of sensors which will be mounted onto
# the platform. In this case, we will exploit a :class:`~.RadarElevationBearingRangeRate`
# and a :class:`~.PassiveElevationBearing` sensor (e.g. an optical sensor, which has no capability
# to directly measure range).
#
# First, we will create a radar which is capable of measuring bearing (:math:`\phi`),
# elevation (:math:`\theta`), range (:math:`r`), and range-rate (:math:`\dot{r}`) of the target
# platform.

# Import a range rate bearing elevation capable radar
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate

# Create a radar sensor
radar_noise_covar = CovarianceMatrix(np.diag(
    np.array([np.deg2rad(3),  # Elevation
              np.deg2rad(3),  # Bearing
              100.,  # Range
              25.])))  # Range Rate

# radar mountings
radar_mounting_offsets = StateVector([10, 0, 0])  # e.g. nose cone
radar_rotation_offsets = StateVector([0, 0, 0])

# Mount the radar onto the platform

radar = RadarElevationBearingRangeRate(ndim_state=6,
                                       position_mapping=(0, 2, 4),
                                       velocity_mapping=(1, 3, 5),
                                       noise_covar=radar_noise_covar,
                                       mounting_offset=radar_mounting_offsets,
                                       rotation_offset=radar_rotation_offsets,
                                       )
sensor_platform.add_sensor(radar)

# %%
# Our second sensor is a passive sensor, capable of measuring the bearing (:math:`\phi`) and
# elevation (:math:`\theta`) of the target platform. For the purposes of this example, we will
# assume that the passive sensor is an imager.
# The imager sensor model is described by the following equations:
#
# .. math::
#           \mathbf{z}_k = h(\mathbf{x}_k, \dot{\mathbf{x}}_k)
#
# where:
#
# * :math:`\mathbf{z}_k` is a measurement vector of the form:
#
# .. math::
#           \mathbf{z}_k = \begin{bmatrix} \theta \\ \phi \end{bmatrix}
#
# * :math:`h` is a non - linear model function of the form:
#
# .. math::
#           h(\mathbf{x}_k,\dot{\mathbf{x}}_k) = \begin{bmatrix}
#               \arcsin(\mathcal{z} /\sqrt{\mathcal{x} ^ 2 + \mathcal{y} ^ 2 +\mathcal{z} ^ 2}) \\
#               \arctan(\mathcal{y},\mathcal{x}) \ \
#               \end{bmatrix} + \dot{\mathbf{x}}_k
#
# * :math:`\mathbf{z}_k` is Gaussian distributed with covariance :math:`R`, i.e.:
#
# .. math::
#           \mathbf{z}_k  \sim \mathcal{N}(0, R)
#
# .. math::
#           R = \begin{bmatrix}
#             \sigma_{\theta}^2 & 0 \\
#             0 & \sigma_{\phi}^2  \\
#             \end{bmatrix}

# Import a passive sensor capability
from stonesoup.sensor.passive import PassiveElevationBearing

imager_noise_covar = CovarianceMatrix(np.diag(np.array([np.deg2rad(0.05),  # Elevation
                                                        np.deg2rad(0.05)])))  # Bearing

# imager mounting offset
imager_mounting_offsets = StateVector([0, 8, -1])  # e.g. wing mounted imaging pod
imager_rotation_offsets = StateVector([0, 0, 0])

# Mount the imager onto the platform
imager = PassiveElevationBearing(ndim_state=6,
                                 mapping=(0, 2, 4),
                                 noise_covar=imager_noise_covar,
                                 mounting_offset=imager_mounting_offsets,
                                 rotation_offset=imager_rotation_offsets,
                                 )
sensor_platform.add_sensor(imager)

# %%
# Notice that we have added sensors to specific locations on the aircraft, defined by the
# mounting_offset parameter. The values in this array are defined in the platforms local
# coordinate frame of reference. Here, an offset of :math:`[0, 8, -1]` means the sensor is
# located 8 meters to the right and 1 meter below the center point of the platform.
#
# Now that we have mounted the two sensors we can see that the platform object has both associated
# with it:
sensor_platform.sensors


# %%
# Create a Target Platform
# ------------------------
# There are two ways of generating a target in Stone Soup. Firstly, we can use the inbuilt
# ground-truth generator functionality within Stone Soup, which we demonstrated in the previous
# example, which creates a random target based on our selected parameters. The second method
# provides a means to generate a target which will perform specific behaviours, this is the
# approach we will take here.
#
# To create a target which moves in pre-defined sequences, we exploit the fact that platforms can
# be used as sensor targets within a simulation, coupled with the
# :class:`~.MultiTransitionMovingPlatform` which provides a platform with a pre-defined list of
# transition models and transition times. The platform will continue to loop over the transition
# sequence provided until the simulation ends.
#
# When simulating sensor platforms, it is important to note that, within the simulation, Stone Soup
# treats all platforms as potential targets. Therefore, multiple sensor platforms *sense* all other
# platforms within the simulation (sensor-target geometry dependant).
#
# For this example, we will create an air target which will fly a sequence of straight and level,
# followed by a coordinated turn in the :math:`x-y` plane. This is configured such that the target
# will perform each manoeuvre for 8 seconds and will turn through 45 degrees over the course of the
# turn manoeuvre.

# Import a Constant Turn model to enable target to perform basic manoeuvre
from stonesoup.models.transition.linear import KnownTurnRate

straight_level = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.), ConstantVelocity(0.), ConstantVelocity(0.)])

# Configure the aircraft turn behaviour
turn_noise_diff_coeffs = np.array([0., 0.])

turn_rate = np.pi/32  # specified in radians per seconds...

turn_model = KnownTurnRate(turn_noise_diff_coeffs=turn_noise_diff_coeffs, turn_rate=turn_rate)

# Configure turn model to maintain current altitude
turning = CombinedLinearGaussianTransitionModel(
    [turn_model, ConstantVelocity(0.)])

manoeuvre_list = [straight_level, turning]
manoeuvre_times = [timedelta(seconds=8),
                   timedelta(seconds=8)]

# %%
# Now that we have created a list of manoeuvre behaviours and durations we can build our
# multi-transition moving platform. Because we intend for this platform to be a target we do not
# need to attach any sensors to it.

# Import a multi-transition moving platform
from stonesoup.platform.base import MultiTransitionMovingPlatform

initial_target_location = StateVector([[0], [-40], [1800], [0], [8000], [0]])
initial_target_state = State(initial_target_location, start_time)
target = MultiTransitionMovingPlatform(transition_models=manoeuvre_list,
                                       transition_times=manoeuvre_times,
                                       states=initial_target_state,
                                       position_mapping=(0, 2, 4),
                                       velocity_mapping=(1, 3, 5),
                                       sensors=None)

# %%
# Creating the simulator
# ----------------------
# Now that we have built our sensor platform and a target platform, we need to wrap them in a
# simulator. Because we do not want any additional ground truth objects, which is how most
# simulators work in Stone Soup, we need to use a :class:`~.DummyGroundTruthSimulator` which
# returns a set of empty ground truth paths with timestamps. These are then feed into a
# :class:`~.PlatformDetectionSimulator` with the two platforms we have already built.

# Import the required simulators
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.simulator.platform import PlatformDetectionSimulator

# %%
#  We now need to create an array of timestamps which starts at *datetime.now()* and enable the
#  simulator to run for 25 seconds.

times = np.arange(0, 24, 1)  # 25 seconds

timestamps = [start_time + timedelta(seconds=float(elapsed_time)) for elapsed_time in times]

truths = DummyGroundTruthSimulator(times=timestamps)
sim = PlatformDetectionSimulator(groundtruth=truths, platforms=[sensor_platform, target])

# %%
# Create a Tracker
# ------------------------------------
# Now that we have set-up our sensor platform, target, and simulation, we need to create a tracker.
# For this example we will use a Particle Filter as this enables us to handle the non-linear
# nature of the imaging sensor. We will also use an inflated constant noise model to account for
# target motion uncertainty.
#
# Note that we don't add a measurement model to the updater. This is because each sensor adds their
# measurement model to each detection they generate. The tracker handles this internally by
# checking for a measurement model with each detection it receives and applying only the relevant
# measurement model.

target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(5), ConstantVelocity(5), ConstantVelocity(1)])

# First add a Particle Predictor
predictor = ParticlePredictor(target_transition_model)

# Now create a resampler and particle updater
resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model=None,
                          resampler=resampler)

# Create a particle initiator
from stonesoup.initiator.simple import GaussianParticleInitiator, SinglePointInitiator
single_point_initiator = SinglePointInitiator(
    GaussianState([[0], [-40], [2000], [0], [8000], [0]],
                  np.diag([10000, 1000, 10000, 1000, 10000, 1000])),
    None)

initiator = GaussianParticleInitiator(number_particles=500,
                                      initiator=single_point_initiator)

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                    missed_distance=np.inf)
data_associator = GNNWith2DAssignment(hypothesiser)

from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter = UpdateTimeStepsDeleter(time_steps_since_update=10)

# Create a Kalman single-target tracker
tracker = SingleTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=sim,
    data_associator=data_associator,
    updater=updater
)

# %%
# The final step is to iterate our tracker over the simulation and plot out the results. Because we
# have a bearing-only sensor, it makes sense to animate the plotting of detections. This
# animation shows the sensor platform (blue) moving towards the true target position (red).
# The estimated target position is shown in black, radar detections are shown in yellow, while the
# bearing-only imager detections are coloured green.

from matplotlib import animation
import matplotlib

matplotlib.rcParams['animation.html'] = 'jshtml'

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.functions import sphere2cart

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)


frames = []
for time, ctracks in tracker:
    artists = []

    ax.set_xlabel("$East$")
    ax.set_ylabel("$North$")
    ax.set_ylim(0, 2250)
    ax.set_xlim(-1000, 1000)
    X = [state.state_vector[0] for state in sensor_platform]
    Y = [state.state_vector[2] for state in sensor_platform]
    artists.extend(ax.plot(X, Y, color='b'))

    for detection in sim.detections:
        if isinstance(detection.measurement_model, CartesianToElevationBearingRangeRate):
            x, y = detection.measurement_model.inverse_function(detection)[[0, 2]]
            color = 'y'
        else:
            r = 10000000
            # extract the platform rotation offsets
            _, el_offset, az_offset = sensor_platform.orientation
            # obtain measurement angles and map to cartesian
            e, a = detection.state_vector
            x, y, _ = sphere2cart(r, a + az_offset, e + el_offset)
            x += detection.measurement_model.translation_offset[0]
            y += detection.measurement_model.translation_offset[1]
            color = 'g'
        X = [sensor_platform.state_vector[0], x]
        Y = [sensor_platform.state_vector[2], y]
        artists.extend(ax.plot(X, Y, color=color))

    X = [state.state_vector[0] for state in target]
    Y = [state.state_vector[2] for state in target]
    artists.extend(ax.plot(X, Y, color='r'))

    for track in ctracks:
        X = [state.mean[0] for state in track]
        Y = [state.mean[2] for state in track]
        artists.extend(ax.plot(X, Y, color='k'))

    frames.append(artists)

animation.ArtistAnimation(fig, frames)


# %%
# To increase your confidence with simulated platform targets, it would be good practice to modify
# the target to fly pre-defined shapes such as a race-track oval. You could also experiment with
# different sensor performance levels to see at what point the tracker is no longer able to
# generate a reasonable estimate of the target location.

# %%
# Key points
# ----------
# 1. Platforms, static or moving, can be used as targets for sensor platforms.
# 2. Simulations can only be built with known platform behaviours when you want to test specific
#    scenarios.
# 3. A tracker can be configured to exploit all sensor data created in a simulation.
