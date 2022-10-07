#!/usr/bin/env python
# coding: utf-8

"""
===========================================================
Accumulated States Densities - Out-of-Sequence measurements
===========================================================
"""
# %%
# Smoothing a filtered trajectory is an important task in live systems. Using
# Rauch–Tung–Striebel retrodiction after the normal filtering has a great effect on
# the filtered trajectories but it is not optimal because one has to calculate the
# retrodiction in an own step. In this point the Accumulated-State-Densities (ASDs) can help.
# In the ASDs the retrodiction is calculated in the prediction and update step.
# We use a multistate over time which can be pruned for better performance. Another advantage
# is the possibility to calculate Out-of-Sequence measurements in an optimal way.
# A more detailed introduction and the derivation of the formulas can be found in [#]_.
#

# %%
# First of all we plot the ground truth of one target moving on the Cartesian 2D plane.
# The target moves in a cubic function.

# %%
from datetime import timedelta
from datetime import datetime
import numpy as np
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

truth = GroundTruthPath()
start_time = datetime.now()
for n in range(1, 202, 2):
    x = n -100
    y = 1e-4 * (n-100)**3
    varxy = np.array([[0.1, 0.], [0., 0.1]])
    xy = np.random.multivariate_normal(np.array([x, y]), varxy)
    truth.append(GroundTruthState(np.array([[xy[0]], [xy[1]]]),
                                  timestamp=start_time + timedelta(seconds=n)))

# Plot the result
_ = ax.plot([state.state_vector[0, 0] for state in truth],
        [state.state_vector[1, 0] for state in truth],
        linestyle="--")

# %%
# Following we plot the measurements made of the ground truth. The measurements have
# an error matrix of variance 5 in both dimensions.

from scipy.stats import multivariate_normal
from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    x, y = multivariate_normal.rvs(
        state.state_vector.ravel(), cov=np.diag([5., 5.]))
    measurements.append(Detection(
        [x, y], timestamp=state.timestamp))

# Plot the result
ax.scatter([state.state_vector[0, 0] for state in measurements],
           [state.state_vector[1, 0] for state in measurements],
           color='b')
fig

# %%
# Now we have to setup a transition model for the prediction and the :class:`~.ASDPredictor`.

from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.asd import ASDKalmanPredictor

transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(0.2), ConstantVelocity(0.2)))
predictor = ASDKalmanPredictor(transition_model)

# %%
# We have to do the same for the measurement model and the :class:`~.ASDKalmanUpdater`.

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.updater.asd import ASDKalmanUpdater

measurement_model = LinearGaussian(
    4,  # Number of state dimensions (position and velocity in 2D)
    (0, 2),  # Mapping measurement vector index to state index
    np.array([[5., 0.],  # Covariance matrix for Gaussian PDF
              [0., 5.]])
)
updater = ASDKalmanUpdater(measurement_model)

# %%
# We set up the state at position (-100, -100) with velocity 0. We set max_nstep
# to 30.

from stonesoup.types.state import ASDGaussianState

prior = ASDGaussianState(multi_state_vector=[[-100.], [0.], [-100.], [0.]],
                         timestamps=start_time,
                         multi_covar=np.diag([1., 1., 1., 1.]),
                         max_nstep=30)

# %%
# Last but not least we set up a track and execute the filtering. The first and last 10 steps
# are processed in sequence. All other measurements are divided in groups of 10 following in time.
# The latest one is processed first and the other 9 are used for filtering. In the end we plot the
# filtered trajectory. The animated plot will show the changing state estimate across `max_nstep`
# set above.
import matplotlib
from matplotlib import animation
matplotlib.rcParams['animation.html'] = 'jshtml'

from stonesoup.plotter import Plotter
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

plotter = Plotter()
frames = []
artists = []

track = Track()  # For ASD track
track2 = Track()  # For Gaussian state equivalent without ASD
processed_measurements = set()
for i in range(0, len(measurements)):
    if i > 10:
        if i % 10 != 0:  # or i%10==3:
            m = measurements[i]
            prediction = predictor.predict(prior, timestamp=m.timestamp)
            track2.append(prediction.state)  # This track will ignore OoS measurements
        else:
            # prediction and update of the newest measurement
            m = measurements[i]
            processed_measurements.add(m)
            prediction = predictor.predict(prior, timestamp=m.timestamp)
            hypothesis = SingleHypothesis(prediction, m)
            # Used to group a prediction and measurement together
            post = updater.update(hypothesis)
            track.append(post)
            track2.append(post.state)
            prior = track[-1]

            artists.extend(plotter.plot_tracks(Track(track[-1].states), [0, 2], color='r'))
            artists.extend(
                plotter.plot_measurements(processed_measurements, [0, 2], measurement_model))
            frames.append(artists); artists =[]
            for j in range(9, 0, -1):
                # prediction and update for all OOS measurement. Beginning with the latest one.
                m = measurements[i - j]
                processed_measurements.add(m)
                prediction = predictor.predict(prior, timestamp=m.timestamp)
                hypothesis = SingleHypothesis(prediction, m)
                # Used to group a prediction and measurement together
                post = updater.update(hypothesis)
                track.append(post)
                prior = track[-1]

                artists.extend(plotter.plot_tracks(Track(track[-1].states), [0, 2], color='r'))
                artists.extend(
                    plotter.plot_measurements(processed_measurements, [0, 2], measurement_model))
                frames.append(artists); artists = []
    else:
        # the first 10 steps are for beginning of the ASD so that it is numerically stable
        m = measurements[i]
        processed_measurements.add(m)
        prediction = predictor.predict(prior, timestamp=m.timestamp)
        hypothesis = SingleHypothesis(prediction, m)
        # Used to group a prediction and measurement together
        post = updater.update(hypothesis)
        track.append(post)
        track2.append(post.state)
        prior = track[-1]

        artists.extend(plotter.plot_tracks(Track(track[-1].states), [0, 2], color='r'))
        artists.extend(
            plotter.plot_measurements(processed_measurements, [0, 2], measurement_model))
        frames.append(artists); artists = []

animation.ArtistAnimation(plotter.fig, frames)

# %%
# For comparision, the plot below shows a approximately equivalent track if
# at each step the prediction was stored, and out of sequence measurements were ignored.

# sphinx_gallery_thumbnail_number = 4
from operator import attrgetter

plotter = Plotter()
asd_states = []
for state in reversed(list(track.last_timestamp_generator())):
    if state.timestamp not in (asd_state.timestamp for asd_state in asd_states):
        asd_states.extend(state.states)
asd_states = sorted(asd_states, key=attrgetter('timestamp'))

plotter.plot_tracks({track2}, [0, 2], uncertainty=True, track_label="Equivalent track without ASD")
_ = plotter.plot_tracks({Track(asd_states)}, [0, 2], color='r', track_label="ASD Track")

# %%
# References
# ----------
# .. [#] W. Koch and F. Govaers, On Accumulated State Densities with Applications to
#        Out-of-Sequence Measurement Processing in IEEE Transactions on Aerospace and Electronic Systems,
#        vol. 47, no. 4, pp. 2766-2778, OCTOBER 2011, doi: 10.1109/TAES.2011.6034663.

