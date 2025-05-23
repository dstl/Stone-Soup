#!/usr/bin/env python

"""
==================================================================
Using linearised ODEs from non-linear dynamic models in Stone Soup
==================================================================
"""

# %%
# In real world applications targets are subject to much more complex dynamics than the simpler
# approximations we have considered in other examples. However, working with non-linear
# and time-variant dynamics can be challenging and computationally expensive, hence the need for
# approximate methods to ease the simulation and tracking tasks.
#
# In this example we present a simple method to adopt when you have to linearise a non-linear
# dynamical model and such linearised method in a Stone Soup application acting as transition model.
#
# Specifically, in this example we will show how to use the Van Loan method [1]_ to linearise a
# nearly constant heading transition model [2]_.
#
# We can use Stone Soup components to create a linearised model and use standard components to
# perform the tracking. This method can be used in other contexts such as the space domain (i.e.,
# linearisation of gravitational forces acting on a satellite).
#
# The (nearly) constant heading model is a dynamical model that acts on a 4D :class:`~.State` vector
# defined as :math:`[x \ y \ s \ \theta]^T` with :math:`s` being the speed and the :math:`\theta`
# being the heading of the target. This model describes the motion of targets on a 2-dimensional
# Cartesian plane (x-y).
# The speed is represented as follows :math:`s = \sqrt{\dot{x}^{2} + \dot{y}^{2}}`, with
# :math:`\dot{x}` which describes the component of the velocity on the x-axis.
#
# The constant heading model assumes that the velocity and the heading of the target follow a
# Random walk, meaning that the absolute acceleration and the turn rate of such model are modelled
# as white noise components that evolve following independent Brownian motions (noted as
# :math:`dw_{k}` and :math:`db_{k}` in the following equations).
#
# The system is described by this Stochastic Differential Equation (SDE), at a timestamp :math:`k`:
#
# .. math::
#           dx_{k} &= s_{k}\cos\theta dt;\\
#           dy_{k} &= s_{k}\sin\theta dt;\\
#           ds_{k} &= \sigma_{s}dw_{k};\\
#           d\theta_{k} &= \sigma_{k}db_{k},\\
#
# where we note :math:`\sigma` as the uncertainty on the speed and heading, :math:`dt` (or in
# discretised form as :math:`\Delta t`) the time interval between :math:`k` and :math:`k-1`.
# The full formulation of the evolution of the model can be found in the paper by Kountouriotis
# and Maskell [2]_. However, we can still present the time evolution of the system as follows:
#
# .. math::
#           x_{k|k-1} &= f_{k|k-1}(x_{k-1}, q_{k}) &= \begin{bmatrix}
#                                                       x_{k-1} + s_{k-1}\cos\theta_{k-1}\Delta t\\
#                                                       y_{k-1} + s_{k-1}\sin\theta_{k-1}\Delta t\\
#                                                       s_{k-1}\\
#                                                       \theta_{k-1}\\
#                                                      \end{bmatrix} \\
#
# and the noise as :math:`\mathcal{Q}_{CH} =
# diag\{0, 0, \sigma^2_{s}\Delta t, \sigma^{2}_{\theta} \Delta t\}`.
# Our implementation uses a 5-dimensional :class:`~.State` which specifies the velocity components
# in the x and y directions.
#
# We linearise the dynamical functions, Ordinary Differential Equations (ODEs), using the automatic
# differentiation functions over the variables, in this specific case we employ
# https://pypi.org/project/torch/.
#
# To run this example, in a clean environment, do  ``pip install stonesoup[ode]``,
# in this way all the new dependencies will be installed (torch).
#
# This example follows the structure:
#
#   1. Create the target trajectory and detections;
#   2. Create the linearised function;
#   3. Instantiate the tracker components;
#   4. Run the tracker using the linearised function and visualise the final tracks;

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
import torch
from functools import lru_cache
from collections.abc import Callable
from scipy.linalg import expm

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

# linearisation
from stonesoup.types.array import CovarianceMatrix
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.base import TimeVariantModel
from stonesoup.base import Property

# Detections and measurement model
from stonesoup.types.state import GaussianState, StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian

# Ground truths generation
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, KnownTurnRate, RandomWalk)

# Tracking components
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis


# Simulation parameters
np.random.seed(1908)
start_time = datetime.now().replace(microsecond=0)
number_steps = 100
timestep_size = timedelta(seconds=5)  # seconds
start_x = -5.  # x starting location
start_y = -2.  # y starting location
speed = 1.  # target speed
theta = 0.  # starting heading (degrees)

# %%
# 1. Create the target trajectory and detections;
# -----------------------------------------------
# Following the example presented in Kountouriotis and Maskell [2]_, we consider a target
# which moves with a constant heading trajectory and at specific times performs a turn, modifying
# its course.
# We model the trajectory by using an existing transition model present in Stone Soup, in
# particular using the transition model :class:`~.KnownTurnRate`. We consider the transition models
# without any process noise to keep the trajectory as close as possible to the one presented in the
# paper.
# We model the process noise on the :math:`\theta` as a Random walk.
#
# To generate detections we employ a simple :class:`~.LinearGaussian` measurement model.
# In this example, we consider a negligible clutter noise.

# initialise the transition models the ground truth will use
constant_velocity = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.00), ConstantVelocity(0.00), RandomWalk(0.0)])
turn_left = CombinedLinearGaussianTransitionModel(
    [KnownTurnRate([0.0, 0.0], np.radians(90)), RandomWalk(0.0)])
turn_right = CombinedLinearGaussianTransitionModel(
    [KnownTurnRate([0.0, 0.0], np.radians(-90)), RandomWalk(0.0)])

# generate the ground truths in the same way
truth = GroundTruthPath([GroundTruthState(StateVector(
    np.array([start_x, speed*np.cos(theta), start_y, speed*np.sin(theta), theta])),
    timestamp=start_time)])

for k in range(1, number_steps+1):
    if k == 33:  # model the turn left
        new_state = turn_left.function(truth[k-1], time_interval=timestep_size, noise=False)
        truth.append(GroundTruthState(
            new_state,
            timestamp=start_time + timedelta(seconds=timestep_size.total_seconds()*k)))

    elif k == 66:  # model the turn right
        new_state = turn_right.function(truth[k - 1], time_interval=timestep_size, noise=False)
        truth.append(GroundTruthState(
            new_state,
            timestamp=start_time + timedelta(seconds=timestep_size.total_seconds()*k)))

    else:  # straight trajectory
        new_state = constant_velocity.function(
            truth[k - 1], time_interval=timestep_size, noise=False)
        truth.append(GroundTruthState(
            new_state,
            timestamp=start_time + timedelta(seconds=timestep_size.total_seconds()*k)))

# Instantiate the measurement model
measurement_model = LinearGaussian(ndim_state=5,
                                   mapping=(0, 2),
                                   noise_covar=np.array([[10, 0],
                                                         [0, 10]]))

# Collect the measurements and timestamps
measurements, times = [], []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(TrueDetection(
        state_vector=measurement,
        timestamp=state.timestamp,
        groundtruth_path=truth,
        measurement_model=measurement_model))

    times.append(state.timestamp)

# %%
# Let's visualise the target ground truth and detections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# 2. Create the linearised function;
# ----------------------------------
# We have created a series of detections from a target moving with peculiar dynamics as can be seen
# in the figure.
# We can build the constant heading function and the class for the linearisation that would generate
# a linearised transition model.
#
# We create here a copy of a standard Stone Soup transition model class, in which we perform the
# automatic differentiation to obtain the ``jacobian`` matrix and perform the linearisation. In
# addition, as can be seen in other transition models, we instantiate the methods to create a new
# state given a previous one (``function``) and evaluate the covariance matrix (``covar``).


# Create the constant heading function
def constant_heading(state, **kwargs):
    """
        Function that describes the constant heading dynamics;
        The states are a bit odd
    """

    (x, vx, y, vy, theta) = state
    # generate the s component
    s = torch.sqrt(vx*vx + vy*vy)
    return s * torch.cos(theta), 0. * s, s * torch.sin(theta), 0. * s, 0. * theta


# Create the linearisation class
class LinearisedModel(GaussianTransitionModel, TimeVariantModel):
    """
        Class linearisation

        Specify the noise coefficients and the differential equation
    """
    linear_noise_coeffs: np.ndarray = Property(
        doc=r"Noise diffusion coefficients")
    differential_equation: Callable = Property(
        doc=r"Differential equation")

    @property
    def ndim_state(self):
        """ Function to obtain the state dimension """
        return 5

    @lru_cache
    def _get_jacobian(self, function, input_state, **kwargs):
        """function to linearise the dynamic model

            function : function = ODE
            input_state : StateVector = position where to differentiate the ODE
        """

        # Specify the timestamp, state and dimension
        timestamp = input_state.timestamp
        state = input_state.state_vector
        nx = self.ndim_state

        # evaluate the jacobian at a specific timestamp
        d_acc = lambda a: function(a, timestamp=timestamp)

        # Initialise the matrix
        A = np.zeros([nx, nx])
        istate = [i for i in state]

        # use autograd to create the matrix components
        jac_rows = torch.autograd.functional.jacobian(d_acc, torch.tensor(istate))

        for i, r in enumerate(jac_rows):
            A[i] = r

        return A

    def _do_linearise(self, da, state, dt):
        """ Function that linearises the ODE.

            da : function = ODE to be linearised
            state : StateVector/np.array = state where differentiate the ODE
            dt : timedelta = time interval where differentiate the ODE
        """

        # Define the timestamp and dimension
        timestamp = state.timestamp
        nx = self.ndim_state

        # Obtain the jacobian of the ODE
        dA = self._get_jacobian(da, state, timestamp=timestamp)

        # Make the jacobian available for all the class at specific time and state
        self.dA = dA

        # Compute the integral of the Jacobian
        # Get \int e^{dA*s}\,ds
        int_eA = expm(dt * np.block([[dA, np.identity(nx)], [np.zeros([nx, 2 * nx])]]))[:nx, nx:]

        # Get new value of x
        x = [i for i in state.state_vector]

        newx = x + int_eA @ da(torch.tensor(x))

        return newx

    def jacobian(self, state, **kwargs):
        """ Function for the jacobian"""

        da = self.differential_equation
        timestamp = state.timestamp
        dt = kwargs['time_interval'].total_seconds()

        dA = self._get_jacobian(da, state, timestamp=timestamp)
        # Share the jacobian across the class, might not be needed
        self.dA = dA
        A_d = expm(dA * dt)

        return A_d

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """ Transition function that uses the linearised function """

        # use the differential equation and time delta
        da = self.differential_equation
        dt = kwargs['time_interval'].total_seconds()

        # # create a new state using the linearised ODE
        new_state = self._do_linearise(da, state, dt)

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(prior=state, **kwargs)
            else:
                noise = 0

        return StateVector(new_state) + noise

    def covar(self, time_interval, **kwargs):
        """ Function to create the covariance matrix

            We use the jacobian obtained previously using
            self.dA
        """

        # define the dimension and timedelta
        nx = self.ndim_state
        dt = time_interval.total_seconds()
        # Get Q
        q_v, q_theta = self.linear_noise_coeffs
        dQ = np.diag([0, q_v, 0, q_v, q_theta])

        G = expm(dt * np.block([[-self.dA, dQ],
                                [np.zeros([nx, nx]),
                                 np.transpose(self.dA)]]))
        Q = np.transpose(G[nx:, nx:]) @ (G[:nx, :nx])
        Q = (Q + np.transpose(Q)) / 2.
        return CovarianceMatrix(Q)


# Instantiate the transition model
transition_model = LinearisedModel(
    differential_equation=constant_heading,
    linear_noise_coeffs=np.array([5, np.radians(0.5)]))

# %%
# 3. Instantiate the tracker components;
# --------------------------------------
#
# We have the linearised model, we can now prepare the tracking components.
# Given this simple example we do not require the use of a data associator, initiator or deleter
# because we consider the case of a single target scenario. For this example, we employ the
# :class:`~.ExtendedKalmanPredictor` and :class:`~.ExtendedKalmanUpdater` components, and we create
# the prior on the first known position of the target simulation.
#
# As soon as we have instantiated the track we loop over the detections contained in the scans, and
# we perform the tracking.

# Create the predictor and updater
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# prior and track
prior = GaussianState(state_vector=StateVector(np.array([start_x, speed*np.cos(theta),
                                                         start_y, speed*np.sin(theta), theta])),
                      covar=np.diag([1, 1, 1, 1, 1]),
                      timestamp=start_time)
track = Track([prior])

# Tracking
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Visualise the final result
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig

# sphinx_gallery_thumbnail_number = 2

# %%
# Conclusion
# ----------
# In this example we have presented a method that shows how to linearise a differential equation
# and how to use such model to propagate the transition model of a target and perform the tracking
# in a navigation scenario (constant heading).
# This example shows how a non-linear dynamics can be approached and modelled simply in Stone Soup
# and how it can be adapted over various applications with limited changes in the linearised class
# structure.

# %%
# References
# ----------
# .. [1] C. Van Loan, “Computing integrals involving the matrix exponential”,
#        IEEE Transactions on Automatic 460 Control, Vol. 23, No. 3, pp 395—404, 1978.
# .. [2] Kountouriotis, Panagiotis-Aristidis, and Simon Maskell. "Maneuvering target tracking using
#        an unbiased nearly constant heading model." 2012 15th International Conference on
#        Information Fusion. IEEE, 2012.
