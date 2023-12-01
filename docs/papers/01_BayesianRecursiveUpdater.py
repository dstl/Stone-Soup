#!/usr/bin/env python

"""
==============================
1 - Bayesian Recursive Updater
==============================
"""

# %%
# The Paper
# ---------
#
# This entry of Papers in Stone Soup looks at the Bayesian Recursive Update Filter (BRUF). The BRUF
# is an extension of the Extended Kalman Filter (EKF), that recursively iterates over the update
# step. The BRUF is implemented in Stone Soup as the :class:`~.BayesianRecursiveUpdater`, and is
# based on algorithm 1 from 'RECURSIVE UPDATE FILTERING: A NEW APPROACH'. This paper, authored
# by Kristen A. Michaelson, Andrey A. Popov, and Renato Zanetti is available at
# TODO add paper location
#
# Psuedo code for the BRUF is detailed as follows:
#
# Given the prior state estimate, :math:`\hat{\textbf{x}}^-`; the prior covariance, :math:`P`; a
# measurement, :math:`y`; the measurement covariance, :math:`R`; and the number of steps,
# :math:`N`.
#
# .. math::
#       1& \quad \textbf{x}_0 \leftarrow \hat{\textbf{x}}^- \\
#       2& \quad P_0 \leftarrow P \\
#       3& \quad \textbf{for} \ i=1, ..., N \\
#       4& \qquad H_i =\left. \frac{d\textbf{y}}{d\textbf{x}}
#           \right|_{\textbf{x}=\hat{\textbf{x}}_{i-1}} \\
#       5& \qquad K_i = P_{i-1}H_{i}^T(H_iP_{i-1}H_i^T + NR)^{-1}\\
#       6& \qquad \hat{\textbf{x}}_{i} \leftarrow \hat{\textbf{x}}_{i-1} +
#           K_i(\textbf{y} - h(\hat{\textbf{x}}_{i-1}))\\
#       7& \qquad P_i \leftarrow (I - K_iH_i)P_{i-1}(I-K_iH_i)^T + K_i(NR)K_i^T \\
#       8& \quad \textbf{end for} \\
#       9& \quad \textbf{return} \ \hat{\textbf{x}}_{N}, \ P_N
#
#
# The Code
# --------
#
# The Stone Soup source code for the BRUF algorithm is available at
# :class:`~.BayesianRecursiveUpdater`. This section provides an example of using the
# :class:`~.BayesianRecursiveUpdater` in a tracking scenario. Similar to other Stone Soup examples,
# we begin with scenario generation.

# %%
# Ground Truth and Measurement Generation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.detection import Detection
from stonesoup.plotter import Plotterly

start_time = datetime.now().replace(microsecond=0)
np.random.seed(1991)

# Ground Truth simulation
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])
timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

# Sensor Model
sensor_x = 50  # Placing the sensor off-centre
sensor_y = 0
measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
    # sensor in cartesian.
)

# Simulate measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# Plot the Ground Truth and Measurements
plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Elements of the Bayesian Recursive Updater Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The BRUF uses the same predictor as the EKF, but implements a new updater. The
# :class:`~.BayesianRecursiveUpdater` takes the same parameters as the EKF, with an additional new
# parameter: `number_steps`. This parameter dictates how many iterations the updater will complete
# before returning an updated state.

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)

from stonesoup.updater.recursive import BayesianRecursiveUpdater
updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=10)

# %%
# Run the BRUF
# ^^^^^^^^^^^^

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig
