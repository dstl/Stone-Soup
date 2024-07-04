#!/usr/bin/env python

"""
Autonomous Source Term Estimation
=================================
"""

# %%
# This example demonstrates how to perform airborne release source term estimation (STE)
# using particle filtering and how to use sensor management to optimise sensing locations
# and arrive at an estimate of the source term.
#
# STE involves taking concentration measurements of an airborne chemical in order to
# estimate the release location, release rate, diffusivity and possibly other environment
# factors. The problem can be considered as a state estimation with a large state dimension
# and small measurement dimension as we can usually only measure the concentration at a
# point location. The underlying models for STE are adapted from [#]_ and the scenario
# presented here is based on work by Hutchinson et al [#]_.

# %%
# Setup
# ^^^^^
# First, some general packages that are used throughout this example are imported. The
# random number generator will also be fixed here for repeatability.

# General imports and environment setup
import numpy as np
from datetime import datetime, timedelta

np.random.seed(1990)

# %%
# Generate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
#
# Here we need to generate the ground truth source term that we seek to estimate.
#
# Unlike the usual target tracking examples, the state we are estimating is static,
# so we do not need to generate a ground truth trajectory.
#
# The source term is constructed of 8 variables, as shown below.
#
# .. math::
#     \mathbf{S} = \left[\begin{array}{c}
#             x \\
#             y \\
#             z \\
#             Q \\
#             u \\
#             \phi \\
#             \zeta_1 \\
#             \zeta_2
#         \end{array}\right],
#
# where :math:`x, y` and :math:`z` are the source position in 3D Cartesian space, :math:`Q`
# is the release rate/strength in g/s, :math:`u` is the wind speed in m/s, :math:`\phi` is
# the wind direction in radians, :math:`\zeta_1` is the diffusivity of the gas in the
# environment and :math:`\zeta_2` is the lifetime of the gas.
#

from stonesoup.types.groundtruth import GroundTruthState

start_time = datetime.now()
theta = GroundTruthState([30,  # x
                          40,  # y
                          1,  # z
                          5,  # Q
                          4,  # u
                          np.radians(90),  # phi
                          1,  # ci
                          8],  # cii
                         timestamp=start_time)


# %%
# Plot the resulting plume from the source term. The :class:`~.IsotropicPlume`
# measurement model is used under ideal conditions to provide concentration values in a
# planar grid, to visualise the release. Although the source term contains :math:`z`,
# the sensor will be constrained to a single plane in this example. Therefore,
# the plume is generated at the same level as the sensor will be later on. The
# plume is visualised by using the :class:`~.Plotter` class from Stone Soup.

from stonesoup.plotter import Plotter
from stonesoup.types.state import StateVector
from stonesoup.models.measurement.gas import IsotropicPlume

plotter = Plotter()
measurement_model = IsotropicPlume()

z = 0
n_values = 200
intensity = np.zeros([n_values, n_values])
pos_x = np.zeros([n_values])
pos_y = np.zeros([n_values])
for n_x, x in enumerate(np.linspace(0, 50, n_values)):
    pos_x[n_x] = x
    for n_y, y in enumerate(np.linspace(0, 50, n_values)):
        pos_y[n_y] = y
        pos = StateVector([x, y, z])
        measurement_model.translation_offset = pos
        intensity[n_x, n_y] = measurement_model.function(theta)

plotter.ax.set_xlim(left=0, right=50)
plotter.ax.set_ylim(bottom=0, top=50)
plotter.ax.set_box_aspect(1)
gas_distribution = plotter.ax.pcolor(pos_x, pos_y, intensity.T)
plotter.fig.colorbar(gas_distribution, label='Concentration')

# %%
# Create sensor and platform
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The sensor used here is a :class:`~.GasIntensitySensor` that provides point concentration
# measurements. The documentation for the sensor provides insight into the various noise
# parameters used here.
#
# The sensor alone is not controllable and therefore will be mounted onto an
# actionable platform with the :class:`~.NStepDirectionalGridMovable` movement controller,
# that allows the platform to move on the :math:`xy` plane in fixed step sizes.
# The resulting trajectory from a controller of this type can be seen later
# in the results.
#

from stonesoup.types.state import State
from stonesoup.platform import FixedPlatform
from stonesoup.movable.grid import NStepDirectionalGridMovable
from stonesoup.sensor.gas import GasIntensitySensor

gas_sensor = GasIntensitySensor(min_noise=1e-4,
                                missed_detection_probability=0.3,
                                sensing_threshold=5e-4)

sensor_platform = FixedPlatform(
    movement_controller=NStepDirectionalGridMovable(states=[State([[5], [5], [0.]],
                                                                  timestamp=start_time)],
                                                    position_mapping=(0, 1, 2),
                                                    resolution=1,
                                                    n_steps=2,
                                                    step_size=2,
                                                    action_mapping=(0, 1),
                                                    action_space=np.array([[0, 50], [0, 50]])),
    sensors=[gas_sensor])

# %%
# Create particle predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now the :class:`~.ParticlePredictor` and :class:`~.ParticleUpdater` are constructed.
# The particle predictor will be created with a :class:`~.RandomWalk` motion model with 0
# magnitude, meaning that the predictor will not change the estimated source term. This
# is due to the static nature of the source term in this work. If a mobile release was
# to be estimated, this assumption should change.
#
# The :class:`~.ParticleUpdater` is created with an effective sample size resampling
# technique (:class:`~.ESSResampler`) and Markov Chain Monte Carlo regularisation
# (:class:`~.MCMCRegulariser`). A constraint function is also provided to the
# :class:`~.ParticleUpdater` and :class:`~.MCMCRegulariser` which enforces the fact
# that :math:`Q`, :math:`u`, :math:`\zeta_1` and :math:`\zeta_2` can not be negative.

from stonesoup.resampler.particle import ESSResampler
from stonesoup.regulariser.particle import MCMCRegulariser
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.models.transition.linear import RandomWalk, CombinedGaussianTransitionModel


def constraint_function(particle_state):
    logical_indx = ((particle_state.state_vector[3, :] < 0) |
                    (particle_state.state_vector[4, :] < 0) |
                    (particle_state.state_vector[6, :] < 0) |
                    (particle_state.state_vector[7, :] < 0))
    return logical_indx


resampler = ESSResampler()
regulariser = MCMCRegulariser(constraint_func=constraint_function)
predictor = ParticlePredictor(CombinedGaussianTransitionModel([RandomWalk(0.0)] * 8))
updater = ParticleUpdater(measurement_model,
                          resampler=resampler,
                          regulariser=regulariser,
                          constraint_func=constraint_function)

# %%
# Create reward function and sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The Kullback-Leibler divergence (KLD) is used to decide on future sensing locations.
# To do this, a :class:`~.MultiUpdateExpectedKLDivergence` reward function is created.
# This reward function generates multiple synthetic detections to decided on how rewarding
# candidate actions may be. This is not essential but for STE when sensing can be
# unreliable, it helps to improve robustness in reward estimation, preventing potentially
# rewarding actions from being overlooked. A separate :class:`~.ParticleUpdater` is
# used for the reward function as there will be no resampling or regularisation when
# evaluating candidate actions.
#
# The next step is to create the sensor manager which in this case will be the
# :class:`~.BruteForceSensorManager` as this is a myopic implementation.

from stonesoup.sensormanager.reward import MultiUpdateExpectedKLDivergence
from stonesoup.sensormanager import BruteForceSensorManager

reward_updater = ParticleUpdater(measurement_model=None)
reward_func = MultiUpdateExpectedKLDivergence(updater=reward_updater, updates_per_track=4)
sensormanager = BruteForceSensorManager(sensors={gas_sensor},
                                        platforms={sensor_platform},
                                        reward_function=reward_func)

# %%
# Create prior distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The particle filter is initialised with a broad distribution of particles. Each position
# component receives a uniform distribution across the entire simulation environment. The
# release rate receives a Gamma distribution and the wind speed, direction and diffusivity
# parameters all receive uniform distributions about their respective ground truth values.
# It would be expected in an STE scenario that wind speed, direction and properties of
# the agent would all be known or estimated by a separate system and are therefore not
# the main focus, but their inclusion helps to improve robustness of the plume model.
# Notice that in this particle filter 10000 particles are used to capture the distribution.
# This is because of the large state dimension which requires large numbers of particles
# to get good coverage and effective estimation.

from stonesoup.types.state import StateVectors, ParticleState

n_parts = 10000
prior = ParticleState(StateVectors([np.random.uniform(0, 50, n_parts),
                                    np.random.uniform(0, 50, n_parts),
                                    np.random.uniform(0, 5, n_parts),
                                    np.random.gamma(2, 5, n_parts),
                                    np.random.uniform(0, 6, n_parts),
                                    np.random.uniform(np.radians(0), np.radians(180), n_parts),
                                    np.random.uniform(0, 2, n_parts),
                                    np.random.uniform(6, 10, n_parts)]),
                      weight=np.array([1 / n_parts] * n_parts),
                      timestamp=start_time)

prior.parent = prior

# %%
# Run the filter and sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that all components have been created, the particle filter and sensor managers
# can be run. Here 50 iterations are carried out but this process is usually dependent
# on the estimated distribution covariance.
#
# Storing the sensor :math:`xy` positions is not required to run the algorithms,
# but aids plotting later on.

from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

n_iter = 50
track = Track(prior)
sensor_x = [sensor_platform.position[0]]
sensor_y = [sensor_platform.position[1]]
for n in range(n_iter):
    time = (start_time + timedelta(seconds=n + 1))
    if n > 0:
        chosen_actions = sensormanager.choose_actions({track}, time, n_steps=2, step_size=2,
                                                      action_mapping=(0, 1))
        for sensor, actions in chosen_actions[0].items():
            sensor.add_actions(actions)
            sensor.act(time)

    prediction = predictor.predict(prior, timestamp=time)
    theta.timestamp = time
    detection = (gas_sensor.measure({theta}, noise=True).pop())
    hypothesis = SingleHypothesis(prediction, detection)
    update = updater.update(hypothesis)
    track.append(update)
    prior = track[-1]
    sensor_x.append(sensor_platform.position[0])
    sensor_y.append(sensor_platform.position[1])

# %%
# Plot the result
# ^^^^^^^^^^^^^^^
#
# To illustrate the result of the simulation, the sensor trajectory created by the management
# algorithm is plotted over the plume resulting from the source term. Concentration
# measurements are also illustrated along the trajectory by the markers along the path
# at the location that the detection was made. The size of the marker is proportional to
# the concentration received. The :math:`xy` position from the particle distribution is
# also shown to illustrate the estimation accuracy.

from matplotlib import animation
import matplotlib
matplotlib.rcParams['animation.html'] = 'jshtml'

# Plot ground truth plume
plotter = Plotter()
plotter.ax.set_xlim(left=0, right=50)
plotter.ax.set_ylim(bottom=0, top=50)
plotter.ax.set_box_aspect(1)
gas_distribution = plotter.ax.pcolor(pos_x, pos_y, intensity.T)
plotter.fig.colorbar(gas_distribution, label='Concentration')

# Plot platform start location
plotter.ax.plot(sensor_platform.movement_controller.states[0].state_vector[0],
                sensor_platform.movement_controller.states[0].state_vector[1],
                'go',
               label='Start Location')

# Plot initial particles and store line object for setting later
parts, = plotter.ax.plot(track[0].state_vector[0],
                         track[0].state_vector[1],
                         'g.',
                         markersize=0.5,
                         linewidth=0,
                         label='Particles')

# Plot scatter of detection locations, intially with large marker for the legend
detections = plotter.ax.scatter(sensor_x,
                                sensor_y,
                                s=np.array([10]*len(sensor_x)),
                                c='r',
                                linewidth=0,
                                label='Detections')

# Plot first sensor position for constructing trajectory, storing the line object for setting later
trajectory, = plotter.ax.plot(sensor_platform.movement_controller.states[0].state_vector[0],
                              sensor_platform.movement_controller.states[0].state_vector[0],
                              'r-',
                              label='Sensor Path')

plotter.ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))

# Reset the marker sizes after creating legend
detections.set_sizes(np.array([0]*len(sensor_x)))

def anim_func(i):
    # Update particle line object according to latest track
    parts.set_data(track[i].state_vector[0], track[i].state_vector[1])

    states = np.array([[sensor_x[0]], [sensor_y[0]], [0]])
    detection_marker_sizes = np.zeros(51)
    for n in range(i):
        # Collect states upto timestep i
        states = np.append(states, np.array([[sensor_x[n+1]], [sensor_y[n+1]], [0]]), axis=1)

        # Collect detections upto timestep i
        if hasattr(track[n+1], 'hypothesis'):
            detection_marker_sizes[n+1] = track[n+1].hypothesis.measurement.state_vector[0][0]*3000

    # Update trajectory data and detection scatter marker sizes
    trajectory.set_data(states[0], states[1])
    detections.set_sizes(detection_marker_sizes)

animation.FuncAnimation(plotter.fig, anim_func, interval=500, frames=len(track))

# sphinx_gallery_thumbnail_number = 2

# %%
# References
# ----------
#
# .. [#] Hutchinson, Michael & Liu, Cunjia & Chen, Wen-Hua, "Source term estimation of
#        a hazardous airborne release using an unmanned aerial vehicle", Journal of Field
#        Robotics, Vol. 36, 797-917, 2019
# .. [#] Hutchinson, Michael & Liu, Cunjia & Chen, Wen-Hua, "Information-based search
#        for an atmospheric release using a mobile robot: algorithms and experiements",
#        IEEE Transactions on Control Systems Technology, Vol. 27, No. 6, 2388-2402, 2019
#
