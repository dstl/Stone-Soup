#!/usr/bin/env python

"""
==================================================================
Using linearised ODEs from non-linear dynamic models in Stone Soup
==================================================================
"""

# %%
# In real world applications objects are subject to much more complex dynamics than the simpler
# approximations we have considered in other examples. However, working with non-linear
# and time-variant dynamics can be challenging and computationally expensive, hence the needs of
# approximation methods to ease the simulation and the tracking performance.
#
# In this example we present a simple method to adopt when you have to linearise
# a non-linear dynamical model and use such linearised method in a
# Stone soup application acting as transition model.
#
# Specifically in this example we will show how to use the Van Loan method [#]_
# to linearise the gravitational forces acting on a satellite orbiting Earth
# and create a transition model to model the target trajectory, then making use of Stone Soup components
# perform the tracking. This method can be used in other context such as navigation.
#
# To linearise the dynamical functions, Ordinary Differential Equations (ODEs), we
# make use of libraries that allow to automatically differentiate the function over
# the variables, in this specific case we employ https://pypi.org/project/torch/.
#
# To run this example, in a clean environment, do  ``pip install stonesoup``,
# followed by ``pip install torch``.
#
# This example follows the structure:
#   1. Create the linearised function;
#   2. Use the linearised function to create the track and measurements;
#   3. Instantiate the tracker components;
#   4. Run the tracker and visualise the final tracks;
#

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
from stonesoup.types.array import StateVector, CovarianceMatrix, Matrix
from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.models.base import TimeVariantModel
from stonesoup.base import Property
from stonesoup.types.state import GaussianState, StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


# Simulation parameters and gravitaional constants
np.random.seed(2000)
start_time = datetime.now().replace(microsecond=0)
time_interval = timedelta(seconds=360)
simulation_step = 250
Grav0 = 3.9860 * 1e5  # gravitational constant in km units

# %%
# 1. Create the linearised function;
# ----------------------------------
# In this first part we show how to transform an ODE to
# a linearised model that can be used to create tracks of the targets.
# To do so we create a class that behaves in a similar way as the
# Stone Soup transition model classes, with explicit definition of
# ``function`` to generate new data point using the transition matrix and
# ``covar`` for the covariance matrix.
#
# The approach is a general formuation and can be used for multiple functions.
#

# Create the Class for linearisation
class LinearisedModel(GaussianTransitionModel, TimeVariantModel):
    """
        Class linearisation

        Specify the noise coefficients and the differential equation
    """
    linear_noise_coeffs : np.ndarray = Property(
        doc=r"Noise diffusion coefficients")
    differential_equation : Callable = Property(
        doc=r"Differential equation")

    @property
    def ndim_state(self):
        """ Function to obtain the state dimension """
        return 6


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
        d_acc = lambda a:function(a, timestamp=timestamp)

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

        # create a new state using the linearised ODE
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
        q_xdot, q_ydot, q_zdot = self.linear_noise_coeffs
        dQ = np.diag([0., q_xdot, 0., q_ydot, 0., q_zdot])

        G = expm(dt * np.block([[-self.dA, dQ],
                                [np.zeros([nx, nx]),
                                 np.transpose(self.dA)]]))
        Q = np.transpose(G[nx:, nx:]) @ (G[:nx, :nx])
        Q = (Q + np.transpose(Q)) / 2.

        return CovarianceMatrix(Q)


# %%
# We have specified a class that requires  a differential equation, e.g. a force,
# that can be linearised and the process noise as inputs. These imputs will generate the transition method.
# We need to define an ODE, linearise it and use the transition matrix.

# create the ODE function
def acceleration_function(state, constants=None, **kwargs):
    """ Gravitational acceleration"""

    if constants is None:
        constants = 3.9860 * 1e5

    # specify the (x, y, z) and (vx, vy, vz) in this way since they are coming from
    # torch tensors
    x, y, z = state[0], state[2], state[4]
    vx, vy, vz = state[1], state[3], state[5]

    range = torch.linalg.vector_norm(torch.tensor((x, y, z)))

    r_pow_3 = torch.float_power(range, 3)

    # this is the final state containing Velocities and accelerations
    return (vx, -constants * x / r_pow_3,
            vy, -constants * y / r_pow_3,
            vz, -constants * z / r_pow_3)

# %%
# 2. Use the linearised function to create the track and measurements
# -------------------------------------------------------------------
# We have the transition model, we can create a single target
# subject to such forces and model the dynamics accordingly. Then,
# we can specify the measurement model and obtain the measurements
# from the ground truths available.
#
# We employ a non-linear 3D measurement model using :class:`~.CartesianToElevationBearingRange`.

# Create the linearised transition model
transition_model = LinearisedModel(
    differential_equation=acceleration_function,
    linear_noise_coeffs=np.array([10, 10, 10])
)

# Create an initial state, from a known orbital state
initial_state = np.array([20757, -2.676,
                          36700, 1.513,
                          -279.938, 0.0213]) # The states are in km

# Define the initial covariance
initial_covariance = CovarianceMatrix(np.diag([5000, 100, 5000,
                                               100, 5000, 100]))

# Define the trajectory prior
prior= GaussianState(state_vector=StateVector(initial_state),
                     covar=initial_covariance,
                     timestamp=start_time)

# %%
# Create the gound truths

# initialise the ground truth
truth = GroundTruthPath([
    GroundTruthState(initial_state, timestamp=start_time)])

# Generate all the ground truths detections, we set the noise to 0 in this case,
# please note the timesteps
for k in range(1, simulation_step + 1):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1],
                                  noise=False,   # if True the orbit will not be circular anymore
                                  time_interval=time_interval),
        timestamp=start_time + timedelta(seconds=k*time_interval.total_seconds())))

# Prepare the measurement model and true detections
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.detection import TrueDetection

# define the measurement model
measurement_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar= np.diag([50e-6, 50e-6, 5000])  # angles in microradians
)

# Create the measurements
measurements = []
for state in truth:  # loop over the ground truths
    measurement = measurement_model.function(state=state, noise=True)
    measurements.append(TrueDetection(
        state_vector=measurement,
        timestamp=state.timestamp,
        groundtruth_path=truth,
        measurement_model=measurement_model))

# %%
# Let's visualise the results. We add the center of Earth as a
# fixed platform.

from stonesoup.platform.base import FixedPlatform

Center_earth = FixedPlatform(
    states=GaussianState(
        np.array([0, 0, 0, 0, 0, 0]),
        np.diag([1, 1, 1, 1, 1, 1])),
    position_mapping=(0, 2, 4))

from stonesoup.plotter import Plotter, Dimension

plotter = Plotter(Dimension.THREE)
plotter.plot_ground_truths(truth, [0, 2, 4])
plotter.plot_measurements(measurements, [0, 2, 4])
plotter.plot_sensors(Center_earth, mapping=[0, 1, 2], sensor_label='Center Earth')
plotter.fig

# %%
# 3. Instantiate the tracker components and run the tracker
# ---------------------------------------------------------
# We employ a :class:`~.ExtendedKalmanUpdater` and
# :class:`~.ExtendedKalmanPredictor` to perform the tracking.
# Since we are dealing with a single target tracker we don't specify
# any track deleter and initiator. As well, in this simpler case
# we don't consider any misdetection or clutter.

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# Prepare the tracking
track = Track([prior])

for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# Now visualise the results
plotter.plot_tracks(track, [0, 2, 4], track_label='Track')
plotter.fig

# %%
# Conclusion
# ----------
# In this example we have presented a method that shows how to linearise
# a differential equation and how to use such model to propagate the transition
# model of a target and perform the tracking in a orbital space scenario.

# %%
# References
# ----------
# .. [#] C. Van Loan, “Computing integrals involving the matrix exponential”,
#        IEEE Transactions on Automatic 460 Control, Vol. 23, No. 3, pp 395—404, 1978.
