#!/usr/bin/env python

"""
Control Models: Inference with Linear Gaussian Control Model
============================================================
"""

# %% [markdown]
# This example demonstrates the use of a simple control model which can be used to improve
# inference when the control input of the target system is known. This allows manoeuvres
# to be accounted for without relying on high process noise to enable deviations from the
# motion model. This does however rely on **a**) knowing the underlying control model of
# the system, **b**) knowing the inputs that the system is making and when it is making
# them. This limits the application of this technique, usually, to scenarios where the
# tracker and target both belong to the same owner.
#
# The example will proceed as follows:
#   1. Import some required modules
#   2. Define a simple linear control model and its class
#   3. Generate ground truth target trajectory
#   4. Generate measurements
#   5. Define and run baseline Kalman filter
#   6. Define and run Kalman filter with control
#   7. Compare performance
#
# Before proceeding with the example, the reader is reminded that this is illustrative
# of how it is possible to create a control model, rather than the preferred way. There are
# a number of possible techniques for implementing such models depending on how a problem
# or system is defined. There are a number of choices made in this example which do not
# define the only way to create the demonstrated behaviours. The reader may wish to consider
# alternative approaches when implementing for their own system or problem at hand.

# %%
# Standard external module imports and Environment Setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First some packages used throughout the example are imported and random numbers are
# seeded and fixed for repeatability.

# General imports
import numpy as np
from datetime import datetime, timedelta

# Start simulation time
start_time = datetime.now()
# Use a non-unit time increment (and vary it) to check the various matrices are
# calculated correctly
time_interval = timedelta(seconds=0.5)

# Seed random number generation for repeatability
np.random.seed(1991)

# %%
# Define Control Model
# ^^^^^^^^^^^^^^^^^^^^
#
# Control models invoke a second term in the state-space motion model representation.
# Omitting any noise terms, the motion model is now given by
#
# .. math::
#     \mathbf{x}_{k} = F_{k}\mathbf{x}_{k-1} + B_{k}\mathbf{u}_k
#
# where :math:`B_{k}` is the control matrix and :math:`\mathbf{u}_{k}` is the control input both
# at time :math:`k`. It is clear that the first term above is applying the standard transition
# according to the current state and transition matrix while the second term is
# manipulating this according to an applied action. This could be an acceleration for
# example, which gets mapped and applied to the state vector through :math:`B_k`.
#
# In this example, we define a linear constant acceleration control model which applies
# an acceleration input across the interval between :math:`k-1` and :math:`k`. The state vector
# will include position and velocity. Therefore, by the equations of motion, this
# makes the control matrix
#
# .. math::
#     B_k = \left[\begin{array}{c}
#         \frac{\Delta t^2}{2} \\
#         \Delta t
#     \end{array}\right]
#
# for each dimension. This can be concatenated over multiple dimensions, in a similar
# way to the transition matrix, as
#
# .. math::
#     B_k^D = \left[\begin{array}{ccc}
#         B_k^1 & \cdots & \mathbf{0} \\
#         \vdots & \ddots & \vdots \\
#         \mathbf{0} & \cdots & B_k^D
#     \end{array}\right]
#
# for :math:`D` dimensions.

# %%
# Defining the Control Model Class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To define the control model class, we will create a variant of :class:`~.LinearControlModel` and
# overwrite the :attr:`~.LinearControlModel.matrix` attribute according to the above expression.


from stonesoup.models.control.linear import LinearControlModel


class ConstantAccelerationLinearControlModel(LinearControlModel):
    """A model that applies a constant acceleration over a specified time period.
    This is just a :class:`~.LinearControlModel` which accepts a time_interval input
    to matrix to compute the control matrix.

    A location and velocity state vector with location and velocity interleaved is assumed.
    """

    def matrix(self, time_interval, **kwargs) -> np.ndarray:
        r"""

        Parameters
        ----------
        time_interval : :class:`datetime.timedelta`
            A time interval. Note the units used are :math:`s` so accelerations are implicitly
            per second squared.

        Returns
        -------
        : :class:`numpy.ndarray`
            the control-input model matrix, :math:`B_k`
        """

        # Calculate time interval
        deltat = time_interval.total_seconds()
        # Define B for one dimension
        onedm = np.array([[(deltat**2)/2.0], [deltat]])
        # Construct control matrix for each dimension
        control_matrix = self.control_matrix
        for i in range(0, self.ndim):
            control_matrix[2*i:2*i+2, i:i+1] = onedm

        self.control_matrix = control_matrix

        return self.control_matrix


# %%
# Generate Ground Truth
# ^^^^^^^^^^^^^^^^^^^^^
# Now that the control model is defined, it can be used to create the ground truth
# for the target. There are two main differences compared to generating ground truth
# without a control model. The first is deciding on the control input which is required; it
# should be stored for use later in the filtering process. The second is having to apply
# the transition model function and the control model function when progressing states.
#
# In this example, the target will undergo two manoeuvres. The first starts at 40 seconds,
# lasts 50 seconds and will be a turn to the right; the second manoeuvre
# at 120 seconds, again lasting 50 seconds, will be a turn to the left. Nominally, the
# target will traverse according to nearly constant velocity when not manoeuvering.
#
# First we define the models and populate the required parameters. Note that the first
# input to the `ConstantAccelerationLinearControlModel` class is a control matrix. This
# ensures the number of control dimensions is set correctly and will not be used by the
# model. This is evident as the `control_matrix` property is getting overwritten above
# when calling `matrix` which is dependant on the possibly time interval where the action
# is applied.


# Transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0005),
                                                          ConstantVelocity(0.0005)])

# Control model
control_model = ConstantAccelerationLinearControlModel(np.array([[1., 0],
                                                                 [1., 0],
                                                                 [0, 1.],
                                                                 [0, 1.]]),
                                                       control_noise=np.diag([0.005,
                                                                              0.005]))

# %%
# We can then calculate the control inputs and subsequent ground truth states.


from stonesoup.types.state import State
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.array import Matrix

# Define rotation matrix to calculate acceleration input required.
# Arbitrary turn rate set to 7 degrees/deltat
theta = np.radians(7)
costheta = np.cos(theta)
sintheta = np.sin(theta)
# left turn matrix
left = Matrix([[costheta, -sintheta], [sintheta, costheta]])
# right turn matrix
right = Matrix([[costheta, sintheta], [-sintheta, costheta]])

# Initialise truth object
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# Record these for later use when filtering.
controlinputs = []
for k in range(1, 181):

    velocity = truth[-1].state_vector[[1, 3]]

    if 40 < k < 90:
        # Rotate it to the right
        newvelocity = right @ velocity
    elif 120 < k < 170:
        # Rotate it to the left
        newvelocity = left @ velocity
    else:
        newvelocity = velocity

    # Calculate acceleration based on velocity change
    resultant = (newvelocity - velocity)/time_interval.total_seconds()
    u = State(resultant)
    controlinputs.append(u)

    truth.append(GroundTruthState(
        transition_model.function(truth[-1], noise=True, time_interval=time_interval) +
        control_model.function(u, time_interval=time_interval, noise=True),
        timestamp=start_time+timedelta(seconds=k*time_interval.total_seconds())))

# %%
# And plot the resultant path


from stonesoup.plotter import Plotterly
plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Generate Measurements
# ^^^^^^^^^^^^^^^^^^^^^
# Now some measurements of the target throughout the simulation are generated using a
# range-bearing sensor. This will require inference via an :class:`~.ExtendedKalmanUpdater`
# but is more realisic than adopting a linear measurement model.


from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.detection import Detection

sensor_x = -100  # Placing the sensor off-centre
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.02), 0.5]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
    # sensor in cartesian.
)

measurements = []
for state in truth[1:]:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# %%
# Plot the measurements

plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Initialise Baseline Kalman Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now the baseline, ignorant of control input, Kalman filtering components are created.
# Since the transition model is linear we can adopt a standard Kalman predictor. However,
# since the measurement model is not linear, a suitable Kalman variant should be adopted.
# Here we use a :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`.


from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.state import GaussianState

predictor = KalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# Set the prior
prior = GaussianState([[0], [1], [0], [1]], np.diag([10, 1, 10, 1]), timestamp=start_time)

# %%
# Run Baseline Filter
# ^^^^^^^^^^^^^^^^^^^


from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plot the baseline tracks


plotter.plot_tracks(track, [0, 2], uncertainty=True, label="No Control")
plotter.fig

# %%
# Initialise Kalman Filter with Control Model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With the baseline simulation complete, we now initialise the filter which considers
# control input. Since this change only impacts the prediction process, the updater
# created earlier can be used. The only difference here is we now pass two inputs to
# the predictor: :class:`~.KalmanPredictor.measurement_model` and
# :class:`~.KalmanPredictor.control_model`


predictor_with_ctrl = KalmanPredictor(transition_model, control_model)

# Set the prior
covar = Matrix(np.diag([10, 1, 10, 1]))
prior = GaussianState([[0], [1], [0], [1]], covar, timestamp=start_time)

# %%
# Run Filter with Control
# ^^^^^^^^^^^^^^^^^^^^^^^


track_with_ctrl = Track()
for measurement, c_input in zip(measurements, controlinputs):
    prediction = predictor_with_ctrl.predict(prior,
                                             control_input=c_input,
                                             timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track_with_ctrl.append(post)
    prior = track_with_ctrl[-1]

# %%
# Plot and compare performance


plotter.plot_tracks(track_with_ctrl, [0, 2], uncertainty=True, label="With Control")
plotter.fig
