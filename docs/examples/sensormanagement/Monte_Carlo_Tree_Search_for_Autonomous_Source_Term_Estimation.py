#!/usr/bin/env python

"""
Monte Carlo Tree Search for Autonomous Source Term Estimation
=============================================================
"""

# %%
# This example demonstrates how to set up and run a Monte Carlo tree search (MCTS)
# sensor management algorithm for autonomous source term estimation (STE). More details
# about the problem of autonomous STE can be found in the Autonomous Source Term Estimation
# example, so this will not be replicated here. Instead, the focus is on MCTS and how to
# implement it this for the problem.
#
# MCTS is a technique for solving MDP and POMDP problems by simultaneously constructing
# and evaluating a search tree. The process consists of 4 stages that are iterated
# until we reach some predefined computational budget. These processes are: Selection,
# Expansion, Simulation and Backpropagation. The purpose of the algorithm is to
# arrive at the optimal action policy by sequentially estimating the action value
# function, :math:`Q`, and returning the maximal argument to this at the end of
# the process.
#
# The process is described as follows. Starting from the root node (current
# state or estimated state) the best child node is selected. The most common
# way, and the way implemented here, is to select this node according to the
# upper confidence bound (UCB) for trees. This is given by
#
# .. math::
#     \text{argmax}_{a} \frac{Q(h, a)}{N(h, a)}+c\sqrt{\frac{\log N(h)}{N(h,a)}},
#
# where :math:`a` is the action, :math:`h` is the history (for POMDP problems a
# history or belief is commonly used but in MDP problems this would be replaced with a
# state), :math:`Q(h, a)` is the current cumulative action value estimate,
# :math:`N(h, a)` is the number of visits or simulations of this node, :math:`N(h)`
# is the number of visits to the parent node and :math:`c` is the exploration factor,
# defined with :attr:`exploration_factor`. The purpose of the UCB is to trade off
# between exploitation of the most rewarding nodes in the tree and exploration of
# those that have been visited as frequently.
#
# Once the best child node has been selected, this becomes a parent node and a
# new child node added according to the available set of unvisited actions. This
# selection happens at random. This node is then simulated by predicting the
# current state estimate in the parent node and updating this estimate with a
# generated detection after applying the candidate action. This provides a
# predicted future state which is used to calculate the action value of this
# node. This is done by providing a :attr:`reward_function`. If a rollout is
# conducted, then the value of the simulation will consist of the discounted
# sum of rewards from the current node until the specified depth. The final
# step of the iteration is to incorporate the action value into each parent
# node action value to maintain the cumulative simulation action value.
# This creates a tradeoff between future and immediate rewards during the
# next iteration of the search process. Once a predefined computational budget
# has been reached, which in this implementation is the :attr:`niterations`
# attribute, the best child to the root node in the tree is determined and
# returned from the :meth:`choose_actions`. The user can select which criteria
# used to select this best action by defining the :attr:`best_child_policy`.
# Further detail on MCTS can be found in the literature, including
# variations and alternative POMDP approaches [#]_.
#
# This example and MCTS implementation is based on the work by Glover et al [#]_.
# The simulation shown here is similar to that work, with some modification
# to reduce execution time. The reward implemented for this problem is the
# Kullback-Leibler divegence (KLD) using :class:`~.ExpectedKLDivergence`
# and this will be used with the :class:`~.MCTSRolloutSensorManager` and
# benchmarked against a myopic alternative with :class:`~.BruteForceSensorManager`.
#

# %%
# Setup
# ^^^^^
# First, some general packages used throughout the example are imported and
# random number generation is seeded for repeatability.
#

# General imports and environment setup
import numpy as np
from datetime import datetime, timedelta

np.random.seed(1991)

# %%
# Generate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
#
# Here we generate the source term ground truth and import the
# :class:`~.IsotropicPlume` measurement model to plot the resulting plume
# from the ground truth.

from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.models.measurement.gas import IsotropicPlume

start_time = datetime.now()
theta = GroundTruthState([30, # x
                          40, # y
                          1, # z
                          5, # Q
                          4, # u
                          np.radians(90), # phi
                          1, # ci
                          8], # cii
                         timestamp=start_time)

measurement_model = IsotropicPlume()

# %%
# Plot the resulting plume from the ground truth source term. This uses
# the :class:`~.Plotter` class from Stone Soup.

from stonesoup.plotter import Plotter
from stonesoup.types.state import StateVector

plotter = Plotter()

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
# Create sensors and platforms
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The sensor used here is a :class:`~.GasIntensitySensor` that provides
# point concentration measurements. The documentation for the sensor provides
# insight into the various noise parameters used here.
#
# The sensor alone is not controllable and therefore will be mounted onto an
# actionable platform with the :class:`~.NStepDirectionalGridMovable` movement
# controller, that allows movement in the :math:`xy` plane in fixed step sizes. The
# resulting trajectory from a controller of this type can be seen later in the
# results.
#
# The `gas_sensorA` and `sensor_platformA` is to be used with the benchmark
# algorithm and `gas_sensorB` and `sensor_platformB` is to be used with the
# MCTS sensor manager.

from stonesoup.types.state import State
from stonesoup.platform import FixedPlatform
from stonesoup.movable.grid import NStepDirectionalGridMovable
from stonesoup.sensor.gas import GasIntensitySensor

gas_sensorA = GasIntensitySensor(min_noise=1e-4,
                                 missed_detection_probability=0.3,
                                 sensing_threshold=5e-4)

sensor_platformA = FixedPlatform(
    movement_controller=NStepDirectionalGridMovable(states=[State([[5], [15], [0.]],
                                                                  timestamp=start_time)],
                                                    position_mapping=(0, 1, 2),
                                                    resolution=2,
                                                    n_steps=1,
                                                    step_size=1,
                                                    action_mapping=(0, 1),
                                                    action_space=np.array([[0, 50], [0, 50]])),
    sensors=[gas_sensorA])

gas_sensorB = GasIntensitySensor(min_noise=1e-4,
                                 missed_detection_probability=0.3,
                                 sensing_threshold=5e-4)

sensor_platformB = FixedPlatform(
    movement_controller=NStepDirectionalGridMovable(states=[State([[5], [15], [0.]],
                                                                  timestamp=start_time)],
                                                    position_mapping=(0, 1, 2),
                                                    resolution=2,
                                                    n_steps=1,
                                                    step_size=1,
                                                    action_mapping=(0, 1),
                                                    action_space=np.array([[0, 50], [0, 50]])),
    sensors=[gas_sensorB])


# %%
# Create particle predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now the :class:`~.ParticlePredictor` and :class:`~.ParticleUpdater` are
# constructed. The particle predictor will be created with a :class:`~.RandomWalk`
# motion model with 0 magnitude, meaning that the predictor will not change
# the estimated source term.
#
# The :class:`~.ParticleUpdater` is created with an effective sample size
# resampling technique (:class:`~.ESSResampler`) and Markov Chain Monte Carlo
# regularisation (:class:`~.MCMCRegulariser`). A constraint function is also
# provided to the :class:`~.ParticleUpdater` and :class:`~.MCMCRegulariser`
# to prevent invalid states from being generated.


from stonesoup.resampler.particle import ESSResampler
from stonesoup.regulariser.particle import MCMCRegulariser
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.models.transition.linear import RandomWalk, CombinedGaussianTransitionModel

def constraint_function(particle_state):
    logical_indx = ((particle_state.state_vector[3, :]<0) |
        (particle_state.state_vector[4, :]<0) |
        (particle_state.state_vector[6, :]<0) |
        (particle_state.state_vector[7, :]<0))
    return logical_indx

resampler = ESSResampler()
regulariser = MCMCRegulariser(constraint_func=constraint_function)
predictor = ParticlePredictor(CombinedGaussianTransitionModel([RandomWalk(0.0)]*8))
updater = ParticleUpdater(measurement_model,
                          resampler=resampler,
                          regulariser=regulariser,
                          constraint_func=constraint_function)

# %%
# Create reward functions and sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Both the myopic benchmark and MCTS algorithms will be using the KLD
# reward function (:class:`~.ExpectedKLDivergence`) but two separate
# instances are created as the MCTS version will be required to return
# tracks to store states in the search tree as it is constructed. Both
# are created with a separate :class:`~.ParticleUpdater` that does
# not perform resampling or regularisation.
#
# The :class:`~.BruteForceSensorManager` is defined in the usual way,
# but :class:`~.MCTSRolloutSensorManager` requires some more arguments.
# :attr:`niterations` defines the computational budget for MCTS,
# :attr:`exploration_factor` controls how exploratory the tree search
# behaviour will be, :attr:`discount_factor` discounts future rewards
# calculated in the rollout to reflect increasing uncertainty of rewards
# with increasing future depth, :attr:`rollout_depth` controls the
# rollout horizon and :attr:`best_child_policy` determines how to select
# the best child at the end of the MCTS process. Choices include maximum
# action value (``'max_cumulative_reward'``), average action value per
# visit (``'max_average_reward'``) and maximum number of visits
# (``'max_visits'``).


from stonesoup.sensormanager.reward import ExpectedKLDivergence
from stonesoup.sensormanager import BruteForceSensorManager
from stonesoup.sensormanager.tree_search import MCTSRolloutSensorManager

reward_updater = ParticleUpdater(measurement_model=None)

# Myopic benchmark approach
reward_funcA = ExpectedKLDivergence(updater=reward_updater, measurement_noise=True)
sensormanagerA = BruteForceSensorManager(sensors={gas_sensorA},
                                         platforms={sensor_platformA},
                                         reward_function=reward_funcA)

# MCTS with rollout approach
reward_funcB = ExpectedKLDivergence(updater=reward_updater,
                                    measurement_noise=True,
                                    return_tracks=True)
sensormanagerB = MCTSRolloutSensorManager(sensors={gas_sensorB},
                                          platforms={sensor_platformB},
                                          reward_function=reward_funcB,
                                          niterations=100,
                                          exploration_factor=0.05,
                                          discount_factor=0.9,
                                          search_depth=5,
                                          best_child_policy='max_average_reward')

# %%
# Create prior distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each component of the source term will receive its own distribution.
# Source location received a uniform distribution across the environment,
# release rate received a Gamma distribution and the wind speed,
# direction and diffusivity parameters all receive normal distributions.
# The prior for both myopic and MCTS algorithms will be identical.


from stonesoup.types.state import StateVectors, ParticleState
from stonesoup.types.track import Track
import copy

n_parts = 10000
priorA = ParticleState(StateVectors([np.random.uniform(0, 50, n_parts),
                                    np.random.uniform(0, 50, n_parts),
                                    np.random.uniform(0, 5, n_parts),
                                    np.random.gamma(2, 5, n_parts),
                                    np.random.normal(theta.state_vector[4],
                                                     2,
                                                     n_parts),
                                    np.random.normal(theta.state_vector[5],
                                                     np.radians(10),
                                                     n_parts),
                                    np.random.normal(theta.state_vector[6],
                                                     0.1,
                                                     n_parts),
                                    np.random.normal(theta.state_vector[7],
                                                     1,
                                                     n_parts)]),
                      weight=np.array([1/n_parts]*n_parts),
                      timestamp=start_time)

priorA.parent = priorA
priorB = copy.copy(priorA)

# %%
# Run the myopic sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First the myopic sensor manager and associated filter is run
# over 100 iterations, or until it reaches a stopping criteria
# defined as the square root of the covariance trace being below a
# threshold value. It is important to note that this has no bearing
# on the performance of the estimation, just the convergence of
# the particle distribution.

from stonesoup.types.hypothesis import SingleHypothesis

n_iter = 100
stopping_criteria = 3

trackA = Track(priorA)
sensorA_x = [sensor_platformA.position[0]]
sensorA_y = [sensor_platformA.position[1]]
for n in range(n_iter):
    time = (start_time + timedelta(seconds=n+1))
    if n > 0:
        chosen_actions = sensormanagerA.choose_actions({trackA}, time, n_steps=2, step_size=2,
                                                      action_mapping=(0, 1))
        for sensor, actions in chosen_actions[0].items():
            sensor.add_actions(actions)
            sensor.act(time)

    prediction = predictor.predict(priorA, timestamp=time)
    theta.timestamp=time
    detection = (gas_sensorA.measure({theta}, noise=True).pop())
    hypothesis = SingleHypothesis(prediction, detection)
    update = updater.update(hypothesis)
    trackA.append(update)
    priorA = trackA[-1]
    sensorA_x.append(sensor_platformA.position[0])
    sensorA_y.append(sensor_platformA.position[1])

    if np.sqrt(np.trace(trackA.state.covar)) < stopping_criteria:
        print('Converged in {} iterations'.format(n))
        break


# %%
# Plot the myopic result
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The performance of the myopic benchmark approach is illustrated
# with an animation showing the sensor platform trajectory,
# detections along this trajectory and the resulting source estimate
# with the particle distributions.

from matplotlib import animation
import matplotlib
matplotlib.rcParams['animation.html'] = 'jshtml'
matplotlib.rcParams['animation.embed_limit'] = 40000000

# Plot ground truth plume
plotterA = Plotter()
plotterA.ax.set_xlim(left=0, right=50)
plotterA.ax.set_ylim(bottom=0, top=50)
plotterA.ax.set_box_aspect(1)
gas_distributionA = plotterA.ax.pcolor(pos_x, pos_y, intensity.T)
plotterA.fig.colorbar(gas_distributionA, label='Concentration')
plotterA.ax.set_title('Myopic KLD Sensor Management Result')

# Plot platform start location
plotterA.ax.plot(sensor_platformA.movement_controller.states[0].state_vector[0],
                sensor_platformA.movement_controller.states[0].state_vector[1],
                'go',
               label='Start Location')

# Plot initial particles and store line object for setting later
partsA, = plotterA.ax.plot(trackA[0].state_vector[0],
                           trackA[0].state_vector[1],
                           'g.',
                           markersize=0.5,
                           linewidth=0,
                           label='Particles')

# Plot scatter of detection locations, intially with large marker for the legend
detectionsA = plotterA.ax.scatter(sensorA_x,
                                  sensorA_y,
                                  s=np.array([10]*len(sensorA_x)),
                                  c='r',
                                  linewidth=0,
                                  label='Detections')

# Plot first sensor position for constructing trajectory, storing the line object for setting later
trajectoryA, = plotterA.ax.plot(sensor_platformA.movement_controller.states[0].state_vector[0],
                                sensor_platformA.movement_controller.states[0].state_vector[0],
                                'r-',
                                label='Sensor Path')

plotterA.ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))

# Reset the marker sizes after creating legend
detectionsA.set_sizes(np.array([0]*len(sensorA_x)))

def anim_funcA(i):
    # Update particle line object according to latest track
    partsA.set_data(trackA[i].state_vector[0], trackA[i].state_vector[1])

    states = np.array([[sensorA_x[0]], [sensorA_y[0]], [0]])
    detection_marker_sizes = np.zeros(len(trackA))
    for n in range(i):
        # Collect states upto timestep i
        states = np.append(states, np.array([[sensorA_x[n+1]], [sensorA_y[n+1]], [0]]), axis=1)

        # Collect detections upto timestep i
        if hasattr(trackA[n+1], 'hypothesis'):
            detection_marker_sizes[n+1] = (
                    trackA[n+1].hypothesis.measurement.state_vector[0][0]*3000)

    # Update trajectory data and detection scatter marker sizes
    trajectoryA.set_data(states[0], states[1])
    detectionsA.set_sizes(detection_marker_sizes)

animation.FuncAnimation(plotterA.fig, anim_funcA, interval=100, frames=len(trackA))

# %%
# Run the MCTS sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The MCTS sensor manager and associated filters are run in the same
# way as regular sensor manager in Stone Soup. Again, the process is
# limited to 100 iterations or meeting a stopping criteria, whichever
# comes first.

n_iter = 100
stopping_criteria = 3

trackB = Track(priorB)
sensorB_x = [sensor_platformB.position[0]]
sensorB_y = [sensor_platformB.position[1]]
for n in range(n_iter):
    time = (start_time + timedelta(seconds=n+1))
    if n > 0:
        chosen_actions = sensormanagerB.choose_actions({trackB}, time, n_steps=2, step_size=2,
                                                      action_mapping=(0, 1))
        for sensor, actions in chosen_actions[0].items():
            sensor.add_actions(actions)
            sensor.act(time)

    prediction = predictor.predict(priorB, timestamp=time)
    theta.timestamp=time
    detection = (gas_sensorB.measure({theta}, noise=True).pop())
    hypothesis = SingleHypothesis(prediction, detection)
    update = updater.update(hypothesis)
    trackB.append(update)
    priorB = trackB[-1]
    sensorB_x.append(sensor_platformB.position[0])
    sensorB_y.append(sensor_platformB.position[1])

    if np.sqrt(np.trace(trackB.state.covar)) < stopping_criteria:
        print('Converged in {} iterations'.format(n))
        break


# %%
# Plot the MCTS result
# ^^^^^^^^^^^^^^^^^^^^
#
# The same animation is generated for the MCTS algorithm. Notice
# that the algorithm converged before reaching the iteration limit
# suggesting that it converged faster than the myopic approach in
# this case.


# Plot ground truth plume
plotterB = Plotter()
plotterB.ax.set_xlim(left=0, right=50)
plotterB.ax.set_ylim(bottom=0, top=50)
plotterB.ax.set_box_aspect(1)
gas_distributionB = plotterB.ax.pcolor(pos_x, pos_y, intensity.T)
plotterB.fig.colorbar(gas_distributionB, label='Concentration')
plotterB.ax.set_title('MCTS KLD Sensor Management Result')

# Plot platform start location
plotterB.ax.plot(sensor_platformB.movement_controller.states[0].state_vector[0],
                sensor_platformB.movement_controller.states[0].state_vector[1],
                'go',
               label='Start Location')

# Plot initial particles and store line object for setting later
partsB, = plotterB.ax.plot(trackB[0].state_vector[0],
                           trackB[0].state_vector[1],
                           'g.',
                           markersize=0.5,
                           linewidth=0,
                           label='Particles')

# Plot scatter of detection locations, intially with large marker for the legend
detectionsB = plotterB.ax.scatter(sensorB_x,
                                  sensorB_y,
                                  s=np.array([10]*len(sensorB_x)),
                                  c='r',
                                  linewidth=0,
                                  label='Detections')

# Plot first sensor position for constructing trajectory, storing the line object for setting later
trajectoryB, = plotterB.ax.plot(sensor_platformB.movement_controller.states[0].state_vector[0],
                                sensor_platformB.movement_controller.states[0].state_vector[0],
                                'r-',
                                label='Sensor Path')

plotterB.ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))

# Reset the marker sizes after creating legend
detectionsB.set_sizes(np.array([0]*len(sensorB_x)))

def anim_funcB(i):
    # Update particle line object according to latest track
    partsB.set_data(trackB[i].state_vector[0], trackB[i].state_vector[1])

    states = np.array([[sensorB_x[0]], [sensorB_y[0]], [0]])
    detection_marker_sizes = np.zeros(len(trackB))
    for n in range(i):
        # Collect states upto timestep i
        states = np.append(states, np.array([[sensorB_x[n+1]], [sensorB_y[n+1]], [0]]), axis=1)

        # Collect detections upto timestep i
        if hasattr(trackB[n+1], 'hypothesis'):
            detection_marker_sizes[n+1] = (
                    trackB[n+1].hypothesis.measurement.state_vector[0][0]*3000)

    # Update trajectory data and detection scatter marker sizes
    trajectoryB.set_data(states[0], states[1])
    detectionsB.set_sizes(detection_marker_sizes)

animation.FuncAnimation(plotterB.fig, anim_funcB, interval=250, frames=len(trackB))

# sphinx_gallery_thumbnail_number = 3

# %%
# Comparing the performance of both approaches here, it is clear that
# the MCTS algorithm was able to converge to a better source term
# estimate and did so in less iterations than the myopic benchmark. Considering
# non-myopic actions in this scenario allows for more robust handling of
# unreliable measurements, a common problem in STE that is caused by low
# quality sensors or turbulent environment conditions.

# %%
# References
# ----------
#
# .. [#] Kochenderfer, Mykel J. & Wheeler, Tim A. & Wray, Kyle H. "Algorithms for
#        decision making", MIT Press, 2022 (https://algorithmsbook.com/)
# .. [#] Glover, Timothy & Nanavati, Rohit V. & Coombes, Matthew & Liu, Cunjia &
#        Chen, Wen-Hua & Perree, Nicola & Hiscocks, Steven. "A Monte Carlo Tree Search
#        Framework for Autonomous Source Term Estimation in Stone Soup, 2024 27th
#        International Conference on Information Fusion (FUSION), 1-8, 2024. Accepted,
#        awaiting publication
#
