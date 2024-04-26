"""
3D plotting example
===================
This example demonstrates the 3D plotting functionality available in the :class:`~.Plotterly`
class in Stone Soup by creating data that is difficult to visualise using 2D
:class:`~.Plotterly` and :class:`~.AnimatedPlotterly` plots. We will use the standard Stone Soup
components to generate ground truth, detections and tracks for two targets moving in 3D space
where one spirals around the other. We then show how 2D plotters can be used to try and
visualise the data, and compare that with the 3D plotter. The data generation mirrors the first
Stone Soup tutorial so can be skipped. """

# %%
# First, include some standard imports and initialise the start time:
from datetime import datetime, timedelta
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    KnownTurnRate, ConstantVelocity
from stonesoup.plotter import AnimatedPlotterly, Plotterly
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

start_time = datetime.now().replace(microsecond=0)

# %%
# Create ground truth for the two targets
# ---------------------------------------
# One target is initialised at the origin and goes in a straight line in the positive-z direction.
# A second target is initialised to spiral around the first target.
timesteps = [start_time]
transition_model1 = CombinedLinearGaussianTransitionModel([
    KnownTurnRate(np.array([0, 0]), 0.5),
    ConstantVelocity(0)])
truth1 = GroundTruthPath([GroundTruthState([0, 1, -2, 0, 0, 0.3], timestamp=start_time)])

transition_model2 = CombinedLinearGaussianTransitionModel([ConstantVelocity(0),
                                                           ConstantVelocity(0),
                                                           ConstantVelocity(0)])
truth2 = GroundTruthPath([GroundTruthState([0, 0, 0, 0, 0, 0.3], timestamp=start_time)])

for k in range(1, 40):
    timesteps.append(start_time+timedelta(seconds=k))
    truth1.append(GroundTruthState(
        transition_model1.function(truth1[k-1], time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
    truth2.append(GroundTruthState(
        transition_model2.function(truth2[k-1], time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
truths = [truth1, truth2]

# %%
# Create detections on the targets
# --------------------------------
# For simplicity, we use the method detailed in the first Stone Soup tutorial.

measurement_model = LinearGaussian(
    ndim_state=6,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2, 4),  # Mapping measurement vector index to state index
    noise_covar=np.array([[0.1, 0, 0],  # Covariance matrix for Gaussian PDF
                          [0, 0.1, 0],
                          [0, 0, 0.02]])
    )

measurements1 = []
measurements2 = []

for state in truth1:
    measurement = measurement_model.function(state, noise=True)
    measurements1.append(Detection(measurement,
                                   timestamp=state.timestamp,
                                   measurement_model=measurement_model))
for state in truth2:
    measurement = measurement_model.function(state, noise=True)
    measurements2.append(Detection(measurement,
                                   timestamp=state.timestamp,
                                   measurement_model=measurement_model))

measurements = measurements1 + measurements2

# %%
# Track the targets
# -----------------
# We do this individually on each target to avoid data association issues. This is unrealistic,
# but we only need data for visualisation purposes.
predictor1 = KalmanPredictor(transition_model1)
predictor2 = KalmanPredictor(transition_model2)
updater = KalmanUpdater(measurement_model)
prior1 = GaussianState([0, 1, -2, 0, 0, 0.3], np.diag([1, 0.1, 1, 0.1, 1, 0.1]),
                       timestamp=start_time)
prior2 = GaussianState([0, 0, 0, 0, 0, 0.3], np.diag([1, 0.1, 1, 0.1, 1, 0.1]),
                       timestamp=start_time)

track1 = Track()
track2 = Track()
for measurement in measurements1:
    prediction = predictor1.predict(prior1, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track1.append(post)
    prior = track1[-1]
for measurement in measurements2:
    prediction = predictor2.predict(prior2, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track2.append(post)
    prior = track2[-1]
tracks = {track1, track2}

# %%
# 2D plotting
# -----------
# We first use the 2D :class:`~.Plotterly` and :class:`~.AnimatedPlotterly` plotters to display
# our data. We will see that it's almost impossible to understand the data without using
# multiple 2D plotters.
#
# The below plot shows a 2D animation of the xy plane. Note that it doesn't show the central
# ground truth path as its xy position does not change for the whole simulation. It is also very
# unclear as to what the 3D picture looks like. Technically, one could read the metadata for
# each point, but it's more practical to also plot the xz plane.

fig_ani = AnimatedPlotterly(timesteps)
fig_ani.plot_ground_truths(truths, [0, 2])
fig_ani.plot_measurements(measurements, [0, 2])
fig_ani.plot_tracks(tracks, [0, 2])
fig_ani.fig

# %%
# To compliment the xy plot, the below figure shows a static plot of the yz plane. Due to
# rotational symmetry, this is very similar to the view of the xz plane. Again, it is unclear
# that one target is spiralling around the other. Due to the way the plotter plots one trace at
# a time, the rotating target appears to always be in front of the other. The user
# must mentally combine these two plots to gather a full picture of target movement.
fig = Plotterly(axis_labels=["y", "z"])
fig.plot_ground_truths(truths, [2, 4])
fig.plot_measurements(measurements, [2, 4])
fig.plot_tracks(tracks, [2, 4], uncertainty=True)
fig.fig

# %%
# 3D plotting
# -----------
#
# We now compare this to the 3-dimensional Plotterly functionality in Stone Soup. This plotter
# is static (does not show progression over time) but is highly interactive. Ground truth,
# measurements and tracks can be toggled on and off, and the plot itself can be rotated and
# enlarged. Metadata is available by hovering over each datum.
#
# All the Plotly-based Plotters have been designed to be initialised using the same syntax. The
# only differences from the 2D plotter when initialising the 3D one is that, because the default
# "dimension" value is 2, one needs to specify "dimension=3". Furthermore, one must specify a
# 3-dimensional mapping.
plt = Plotterly(dimension=3)
plt.plot_ground_truths(truths, [0, 2, 4])
plt.plot_measurements(measurements, [0, 2, 4])
plt.plot_tracks(tracks, [0, 2, 4])
plt.fig

# %%
#
# It may also be desirable to change the aspect ratio of the plotter. Plotly enables this
# through updating the ``scene_aspectmode`` parameter. By default, the plotter scales
# proportionally to the data using the input string ``data``, but the input strings ``cube``,
# ``auto``, and ``manual`` are also available. We demonstrate the ``cube`` option below.
plt.fig.update_layout(scene_aspectmode='cube')
plt.fig

# %%
# Conclusion
# ----------
#
# This example shows how the 3D Plotterly plotter offers enhanced visualisation over the
# available 2D Plotterly plotters for certain use cases. Complex 3D target behaviours can be
# difficult to visualise in 2D, and the fact that the syntax is almost identical for the 3D
# plotter makes it an attractive visualisation tool. However, it should be noted that
# uncertainty and particle representation are not supported for 3D unlike in the 2D plotters
# which may affect its desirability.

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/Plotter3D.PNG'
