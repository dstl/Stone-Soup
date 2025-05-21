#!/usr/bin/env python

"""
======================================
Probabilistic Multi-Hypothesis Tracker
======================================
"""

# %%
# The Probabilistic Multi-Hypothesis Tracker (PMHT) is an advanced tracking algorithm designed to
# handle multiple hypotheses in a tracking scenario. Originally introduced by Streit and
# Luginbuhl [#]_, [#]_ and [#]_, the PMHT is particularly useful when dealing with complex
# scenarios.
# Such scenarios include the presence of multiple potential targets and uncertainties in
# measurements.
#
# In PMHT, the objective is to track multiple targets evolving independently over time.
# Each target's movement is modelled probabilistically, typically using a Markov transition model.
# In each time step, received measurements can correspond to either one of the target models or
# clutter. Clutter refers to measurements that do not correspond to any actual target.
#
# One of the key challenges in multi-target tracking is data association, i.e., correctly
# associating measurements with their respective targets. PMHT addresses this by considering
# multiple hypotheses for  data association.
# This enables different associations between measurements and targets.
#
# Let the motion of target :math:`m` from time step :math:`t-1` to :math:`t` be given by
# the following Markov transition model:
#
# .. math::
#           p(x^m_t | x^m_{t-1}) = \phi^m_t(x^m_t | x^m_{t-1})
#
# Let :math:`k^t_r` be the target which generates measurement :math:`r` at time step :math:`t`.
# Then the measurement :math:`z^t_r` conditional on the assignment is assumed to be given by
# the following measurement model:
#
# .. math::
#          p(z^t_r | x^{m}_t, k^t_r = m) = \begin{cases} 1/V & \text{if }
#          k^t_r = 0 \\ \zeta(z^t_r | x_m) & \text{otherwise.} \end{cases}
#
# We denote by :math:`X` the joint states of all targets over time steps 1 to :math:`T`, and
# similarly let :math:`Z` and :math:`K` denote the joint measurement and association variables.
# Unlike data association algorithms, such as the JPDA Filter, the PMHT ignores the constraint
# that each target can generate at most one detection.
# This simplification allows us to write down the joint probability as:
#
# .. math::
#          p(Z, X, K | \Pi) = \left(\prod_{m=1}^M \phi^m_0(x^m_0)\right)\prod_{t=1}^T\left\{
#          \left(\prod_{m=1}^M \phi^m_t(x^m_t | x^m_{t-1}) \right)
#          \left(\prod_{r=1}^{n_t}\pi_t^{k^t_r}
#          \zeta\left(z^t_r | x^{k^t_r}_t\right) \right) \right\}
#
# where :math:`\pi^m_t` is the prior probability of a measurement at time step :math:`t` being
# assigned to target :math:`m` (or clutter if :math:`m=0`), and :math:`\Pi` is the joint variable
# over all target hypotheses and time steps.
# From this, the algorithm proceeds in a manner similar to the Expectation-Maximisation algorithm.
# Given an initial estimate :math:`X` of the target states and :math:`\Pi` of the association
# probabilities, we choose new values :math:`X` and :math:`\Pi` which maximise
#
# .. math::
#         Q(\Pi, X | \Pi^\prime, X^\prime) = \sum_K \left\{\log p(Z, X, K | \Pi)p(K | Z, X^\prime,
#         \Pi^\prime)\right\}
#
# This is repeated over a number of iterations. In this implementation, the prior means are used
# for the initial estimates of the target states for the first batch of measurements.
# For subsequent batches, we keep an overlap of a configurable number of time steps and predict
# forward using the prior means.

# %%
# Nearly-constant velocity example
# --------------------------------
#
# General imports
# ^^^^^^^^^^^^^^^
# Import the necessary libraries

import numpy as np
import datetime

np.random.seed(1000)

# %%
# Initialise ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^
# Here are some configurable parameters associated with the ground truth. We specify a fixed
# number of initial targets with no births and deaths.

# The simulator requires a distribution for birth targets, so we specify a dummy one.
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
initial_state_mean = StateVector([[0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.eye(4))
start_time = datetime.datetime.now().replace(microsecond=0)
initial_state = GaussianState(initial_state_mean, initial_state_covariance, start_time)
timestep_size = datetime.timedelta(seconds=1)
number_of_steps = 50

# Initial truth states for fixed number of targets
preexisting_states = [[-20, 5, 0, 10], [20, -5, 0, 10]]

# %%
# Create the transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
q_x = 1.0
q_y = 1.0
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

# %%
# Put this all together in a multi-target simulator.
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,  # Transition Model used as propagator for track.
    initial_state=initial_state,  # Initial state to use to generate states
    preexisting_states=preexisting_states,  # State vectors at time 0 for ground truths
                                            # which should exist at the start of simulation.
    timestep=timestep_size,  # Time step between each state. Default one second.
    number_steps=number_of_steps,  # Number of time steps to run for
    birth_rate=0.0,  # Rate at which tracks are born. Expected number of
                     # occurrences (λ) in Poisson distribution. Default 1.0.
    death_probability=0.0  # Probability of track dying in each time step. Default 0.1.
)

# %%
# Initialise the measurement models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The simulated ground truth will then be passed to a simple detection simulator. This has a
# number of configurable parameters, e.g. where, at what rate or detection probability a clutter
# is generated.
# This implements similar logic to the code in tutorial 9's section
# :ref:`auto_tutorials/09_Initiators_&_Deleters:Generate Detections and Clutter`.
# Note that, currently, s very low clutter rate is used as PMHT seems to struggle with high clutter
# rate.
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[1, 0],  # Covariance matrix for Gaussian PDF
                          [0, 1]])
    )

# probability of detection
detection_probability = 0.5

# clutter will be generated uniformly in this are around the target
meas_range = np.array([[-1, 1], [-1, 1]])*1000

# Clutter rate is in mean number of clutter points per scan
clutter_rate = 1.0e-3

# The detection simulator
from stonesoup.simulator.simple import SimpleDetectionSimulator
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=detection_probability,
    meas_range=meas_range,
    clutter_rate=clutter_rate
)

# %%
# Create the tracker components
# -----------------------------
# In this example a Kalman filter is used with global nearest neighbour (GNN) associator. Other
# options are, of course, available. Note that PMHT algorithm currently assumes a fixed number of
# tracks, with data association built in. Therefore, the usage of an initiator or deleter component
# are optional but could be added if needed.

# %%
# Predictor
# ^^^^^^^^^
# Initialise the predictor using the same transition model as generated the ground truth. Note you
# don't have to use the same model.
# We also need to specify a smoother since PMHT does smoothing on batches of measurements
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.smoother.kalman import KalmanSmoother
predictor = KalmanPredictor(transition_model)
smoother = KalmanSmoother(transition_model)

# %%
# Updater
# ^^^^^^^
# Initialise the updater using the same measurement model as generated the simulated detections.
# Note, again, you don't have to use the same model (noise covariance).
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Data associator
# ^^^^^^^^^^^^^^^

# Initial estimate for tracks
init_means = preexisting_states
init_cov = np.diag([1.0, 1.0, 1.0, 1.0])
init_priors = [GaussianState(StateVector(init_mean), init_cov, timestamp=start_time)
               for init_mean in init_means]

# %%
# Run the Tracker
# ---------------
# The PMHT algorithm is a batch-based approach that processes multiple scans simultaneously.
# It uses a set of scans from the previous batch to initialise the new batch for improved
# performance. The algorithm runs a set number of iterations per batch for convergence, with a
# maximum iteration limit.
# Though a convergence test might stop iterations early, the current setup runs all iterations.
# The algorithm can maintain and update prior probabilities of data association, affecting
# performance based on the application.

# Number of measurement scans to run over for each batch
batch_len = 10

# Number of scans to overlap between batches
overlap_len = 5

# Maximum number of iterations to run each batch over
max_num_iterations = 10

# Whether to update the prior data association values during iterations (True or False)
update_log_pi = True

from stonesoup.tracker.pmht import PMHTTracker

pmht = PMHTTracker(
    detector=detection_sim,
    predictor=predictor,
    smoother=smoother,
    updater=updater,
    meas_range=meas_range,
    clutter_rate=clutter_rate,
    detection_probability=detection_probability,
    batch_len=batch_len,
    overlap_len=overlap_len,
    init_priors=init_priors,
    max_num_iterations=max_num_iterations,
    update_log_pi=update_log_pi)

# %%
# Plot ground truth detections against tracking estimates.

from stonesoup.plotter import AnimatedPlotterly

groundtruth = set()
detections = set()
tracks = set()

timestamps = []
for time, ctracks in pmht:
    timestamps.append(time)
    tracks |= ctracks
    detections |= detection_sim.detections
    groundtruth |= groundtruth_sim.groundtruth_paths

plotter = AnimatedPlotterly(timestamps, tail_length=0.3)

plotter.plot_ground_truths(groundtruth, [0, 2])
plotter.plot_tracks(tracks, [0, 2])
plotter.plot_measurements(detections, [0, 2], measurement_model=measurement_model)
plotter.fig
# %%
# Key points
# ----------
# 1. PMHT uses a probabilistic framework and Bayesian recursion to update beliefs about target
#    states.
#    It accounts for uncertainties in target motion and measurement noise.
# 2. An essential aspect of PMHT is the association of predicted target states with measurements.
#    This association step, called data association, involves matching measurements to targets.
#    PMHT uses algorithms or methods for hypothesis generation and evaluation.
#    This helps in effectively associating measurements with predicted target states.
#
# References
# ----------
# .. [#] R. Streit and T. Luginbuhl, “Maximum likelihood method for probabilistic multihypothesis
#        tracking”, Proceedings of SPIE, 1994.
# .. [#] R. Streit and T. Luginbuhl”, Probabilistic Multi-Hypothesis Tracking”, Technical
#        Report, Naval Undersea Warfare Division, 1995 problems, IEE Proc., Radar Sonar
#        Navigation, 146:2–7
# .. [#] S. Davey, D. Gray and R. Streit, “Tracking, association, and classification: a combined
#        PMHT approach”, Digital Signal Processing 12, 372–382, 2002.
