#!/usr/bin/env python

"""
=================================================
Kernel methods: the adaptive kernel Kalman filter
=================================================
"""

# %%
# The AKKF is a model-based Bayesian filter designed for non-linear and non-Gaussian systems.
# It can avoid problematic resampling in most particle filters and reduce computational complexity.
# We aim to realise adaptive updates of the empirical kernel mean embeddings (KMEs) for posterior
# probability density functions (PDFs) using the AKKF, which is executed in the state space,
# measurement space, and reproducing kernel Hilbert space (RKHS).
#
# - In the state space, we generate the proposal state particles and propagate them through the
#   motion model to obtain the state particles.
#
# - In the measurement space, the measurement particles are achieved by propagating the state
#   particles through the measurement model.
#
# - All these particles are mapped into RKHSs as feature mappings and linearly predict and update
#   the corresponding kernel weight mean vector and covariance matrix to approximate the empirical
#   KMEs of the posterior PDFs in the RKHS.
#
# Background and notation
# -----------------------
#
# Kernel mean embedding (KME)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The KME approach represents a PDF :math:`p(\mathbf{x})` as an element in the RKHS using the
# following equation:
#
# .. math::
#     \mu_X := \mathbb{E}_X \left[\phi_\mathbf{x}(X)\right] =
#     \int_{\mathcal{X}} \phi_\mathbf{x}(\mathbf{x}) p(\mathbf{x})d \mathbf{x}.
#
# For joint PDFs of two or more variables, e.g., :math:`p(\mathbf{x}, \mathbf{y})`, they can be
# embedded into a tensor product feature space
# :math:`\mathcal{H}_\mathbf{x} \otimes \mathcal{H}_\mathbf{y}` as an uncentred covariance
# operator:
#
# .. math::
#     \mathcal{C}_{XY} := \mathbb{E}_{XY} \left[\phi_\mathbf{x}(X)\otimes \phi_\mathbf{y}(Y)\right]
#     = \int_{\mathcal{X}\times \mathcal{Y}} \phi_\mathbf{x}(\mathbf{x}) \otimes
#     \phi_\mathbf{y}(\mathbf{y})p(\mathbf{x},\mathbf{y})  d \mathbf{x} d\mathbf{y}.
#
# Similar to the KME of :math:`p(\mathbf{x})`, the KME of a conditional PDF :math:`p(X|\mathbf{y})`
# is defined as
#
# .. math::
#     \mu_{X|\mathbf{y}} := \mathbb{E}_{X|\mathbf{y}} \left[\phi_\mathbf{x}(X)\right] =
#     \int_{\mathcal{X}} \phi_\mathbf{x}(\mathbf{x}) p(\mathbf{x}|\mathbf{y}) d\mathbf{x}.
#
# The difference between the KME  :math:`\mu_{X|\mathbf{y}}` and :math:`\mu_{X}` is that
# :math:`\mu_{X}` is a single element in the RKHS, while :math:`\mu_{X|\mathbf{y}}` is a family of
# points, each indexed by fixing :math:`Y` to a particular value :math:`\mathbf{y}`.
# A conditional operator :math:`\mathcal{C}_{X|Y}` is defined under certain conditions as the
# linear operator, which takes the feature mapping of a fixed value :math:`\mathbf{y}` as the
# input and outputs the corresponding conditional KME:
#
# .. math::
#     \mu_{X|\mathbf{y}} = \mathcal{C}_{X|Y} \phi_\mathbf{y}(\mathbf{y}) =
#     \mathcal{C}_{XY}\left(\mathcal{C}_{YY} + {\kappa} I\right)^{-1} \phi_\mathbf{y}(\mathbf{y}).
#
# Here, :math:`\kappa` is a regularisation parameter to ensure that the inverse is well-defined,
# and :math:`I` is the identity operator matrix.
#
# Empirical KME of conditional distribution
# """""""""""""""""""""""""""""""""""""""""
#
# As it is rare to have access to the actual underlying PDF mentioned above, we can alternatively
# estimate the KMEs using a finite number of samples/particles drawn from the corresponding PDF.
# The :math:`M` sample pairs
# :math:`\mathcal{D}_{XY} = \{ (\mathbf{x}^{\{1\}}, \mathbf{y}^{\{1\}}), \dots,
# (\mathbf{x}^{\{M\}},\mathbf{y}^{\{M\}})\}` are drawn independently and identically distributed
# (i.i.d.) from :math:`p(\mathbf{x},\mathbf{y})` with the feature mappings
# :math:`\Phi := \left[\phi_{\mathbf{x}}(\mathbf{x}^{\{1\}}),\dots,
# \phi_{\mathbf{x}}(\mathbf{x}^{\{M\}})\right]` and
# :math:`\Upsilon := \left[\phi_{\mathbf{y}}(\mathbf{y}^{\{1\}}),\dots,
# \phi_{\mathbf{y}}(\mathbf{y}^{\{M\}})\right]`.
# The estimate of the conditional embedding operator :math:`\hat{\mathcal{C}}_{X|Y}` is obtained
# as a linear regression in the RKHS, as illustrated in figure 1.
# Then, the empirical KME of the conditional distribution, i.e.,
# :math:`p(\mathbf{x}\mid\mathbf{y})\xrightarrow{\text{KME}} \hat{\mu}_{X|\mathbf{y}}`,
# is calculated by a linear algebra operation as:
#
# .. math::
#       p(\mathbf{x}\mid\mathbf{y}) \xrightarrow{\text{KME}}\hat{\mu}_{X|\mathbf{y}} =
#       \hat{\mathcal{C}}_{X|Y}\phi_\mathbf{y}(\mathbf{y})=
#       \Phi\left(G_{\mathbf{y}\mathbf{y}} + \kappa I\right)^{-1}
#       \Upsilon^{\rm{T}}\phi_{\mathbf{y}}(\mathbf{y}) {\equiv} \Phi\mathbf{w}.
#
# Here, :math:`\phi_{\mathbf{y}}(\mathbf{y})` represents the feature mapping of the observation.
# The estimated empirical conditional operator :math:`\hat{\mathcal{C}}_{X|Y}` is regarded as a
# linear regression in the RKHS. :math:`G_{\mathbf{y}\mathbf{y}}  = \Upsilon^{\rm{T}} \Upsilon`
# is the Gram matrix for the samples from the observed variable :math:`Y`.
#
# The empirical KME :math:`\hat{\mu}_{X|\mathbf{y}}` is a weighted sum of the feature mappings of
# the training samples. The weight vector includes :math:`M` non-uniform weights, i.e.,
# :math:`{\mathbf{w}} = \left[w^{\{1\}}, \dots, w^{\{M\}} \right]^{\rm{T}}`, and is calculated as:
#
# .. math::
#      \mathbf{w} =  \left(G_{\mathbf{y}\mathbf{y}} + \kappa I\right)^{-1} G_{:,\mathbf{y}}.
#
# Here, the vector of kernel functions
# :math:`G_{:,\mathbf{y}} = \left[ k_{\mathbf{y}}(\mathbf{y}^{\{1\}}, \mathbf{y}), \dots,
# k_{\mathbf{y}}(\mathbf{y}^{\{M\}}, \mathbf{y}) \right]^{\rm{T}}`.
# The kernel weight vector :math:`\mathbf{w}` is the solution to a set of linear equations in the
# RKHS. Unlike the particle filter, there are no non-negativity or normalisation constraints.
#
# .. image:: ../../_static/KME.png
#   :width: 800
#   :alt: Illustration of Kernel Mean Embedding from data space to kernel feature space
#
# Figure 1: This figure represents the KME of the conditional distribution :math:`p(X|\mathbf{Y})`
# is embedded as a point in kernel feature space as
# :math:`\mu_{X|y} = \int_{\mathcal{X}}\phi_x(x) d P(x|y)`.
# Given the training data sampled from :math:`P(X, Y)`, the empirical KME of :math:`P(X|y)` is
# approximated as a linear operation in RKHS, i.e.,
# :math:`\hat{\mu}_{X|y}=\hat{C}_{X|Y}\phi_y(y)=\Phi\mathbf{w}`.

# %%
# AKKF based on KME
# ^^^^^^^^^^^^^^^^^
#
# The AKKF draws inspiration from the concepts of KME and Kalman Filter to address tracking
# problems in nonlinear systems. Unlike the particle filter, the AKKF approximates the posterior
# PDF using particles and weights, but not in the state space.
# Instead, it employs an empirical KME in an RKHS representation.
# The AKKF aims to estimate the hidden state :math:`\mathbf{x}_k` at the :math:`k`-th time slot
# given the corresponding observation :math:`\mathbf{y}_k`.
# This is achieved by embedding the posterior PDF into the RKHS as an empirical KME:
#
# .. math::
#       p(\mathbf{x}_k \mid \mathbf{y}_{1:k}) \rightarrow \hat{\mu}_{\mathbf{x}_k}^{+} =
#       \Phi_k {\mathbf{w}}^{+}_{k}.
#
# The feature mappings of particles :math:`\mathbf{x}_k^{\{1:M\}}` are represented as
# :math:`\Phi_k`, i.e., :math:`\Phi_k = \left[\phi_{\mathbf{x}}(\mathbf{x}_k^{\{1\}}),\dots,
# \phi_{\mathbf{x}}(\mathbf{x}_k^{\{M\}})\right]`, and the weight vector
# :math:`{\mathbf{w}}^{+}_{k}` includes :math:`M` non-uniform weights.
# The KME :math:`\hat{\mu}_{\mathbf{x}_k}^{+}` is an element in the RKHS that captures the
# feature value of the distribution.
#
# To obtain the empirical KME of the hidden state's posterior PDF, the AKKF requires a set of
# generated particles' feature mappings and their corresponding kernel weights.
# The particles are updated and propagated in the data space using a parametric dynamical state
# space model to capture the diversity of nonlinearities.
# The kernel weight mean vector :math:`\mathbf{w}_k^{\pm}` and covariance matrix
# :math:`S_k^{\pm}` are predicted and updated by matching (or approximating in a least squares
# manner in the feature space) with the state KME.
# This process ensures that the AKKF can effectively estimate the hidden state in nonlinear and
# non-Gaussian systems, leading to improved tracking performance.

# %%
# Implement the AKKF
# ^^^^^^^^^^^^^^^^^^
#
# The AKKF consists of three modules, as depicted in figure 2: a predictor that utilises both prior
# and proposal information at time :math:`k-1` to update the prior state particles and predict the
# kernel weight mean and covariance at time :math:`k`, an updater that employs the predicted
# values to update the kernel weight and covariance, and an updater that generates the proposal
# state particles.
#
# Predictor takes prior and proposal
# """"""""""""""""""""""""""""""""""
#
# :math:`{\color{blue}\blacksquare}` In the state space, at time :math:`k`, the prior state
# particles are generated by passing the proposal particles at time :math:`k-1`, i.e.,
# :math:`\tilde{\mathbf{x}}_{k-1}^{\{i=1:M\}}`, through the motion model as
#
# .. math::
#       \mathbf{x}_k^{\{i\}} =  \mathtt{f}\left(\tilde{\mathbf{x}}_{k-1}^{\{i\}},
#       \mathbf{u}_{k}^{\{i\}} \right).
#
# :math:`{\color{orange}\blacksquare}` In the RKHS, :math:`{\mathbf{x}}_{k}^{\{i=1:M_{\text{A}}\}}`
# are mapped as feature mappings :math:`\Phi_k`.
# Then, the predictive kernel weight vector :math:`\mathbf{w}^{-}_{k}`, and covariance matrix
# :math:`{S}_{k}^{-}`, are calculated as
#
# .. math::
#      \mathbf{w}^{-}_{k} &= \Gamma_{k}  \mathbf{w}^{+}_{k-1}\\
#      {S}_{k}^{-} &= \Gamma_{k} {S}^{+}_{k-1} \Gamma_{k} ^{\mathrm{T}} +V_{k}.
#
# Here, :math:`\mathbf{w}^{+}_{k-1}` and :math:`{S}_{k-1}^{+}` are the posterior kernel weight
# mean vector and covariance matrix at time :math:`k-1`, respectively.
# The transition matrix :math:`\Gamma_{k}` represents the change of sample representation, and
# :math:`{V}_{k}` represents the finite matrix representation of the transition residual matrix.
#
# Updater uses prediction
# """""""""""""""""""""""
#
# :math:`{\color{green}\blacksquare}` In the measurement space, the measurement particles are
# generated according to the measurement model as:
#
#  .. math::
#      \mathbf{y}_k^{\{i\}} =  \mathtt{h}\left({\mathbf{x}}_{k}^{\{i\}},
#      \mathbf{v}_{k}^{\{i\}} \right).
#
# :math:`{\color{orange}\blacksquare}` In the RKHS, :math:`{\mathbf{y}}_{k}^{\{i=1:M_{\text{A}}\}}`
# are mapped as feature mappings :math:`\Upsilon_k`. The posterior kernel weight vector and
# covariance matrix are updated as:
#
# .. math::
#    \mathbf{w}^{+}_{k} &= \mathbf{w}^{-}_{k} +
#    Q_k\left(\mathbf{g}_{\mathbf{yy}_k} - G_\mathbf{yy}\mathbf{w}^{-}_{k}\right)\\
#    S_k^{+} &= S_k^{-} - Q_k G_\mathbf{yy}S_k^{-}
#
#
# Here, :math:`G_\mathbf{yy} = \Upsilon^\mathrm{T}_k\Upsilon_k`,
# :math:`\mathbf{g}_{\mathbf{yy}_k}=\Upsilon_k^\mathrm{T}\phi_\mathbf{y}(\mathbf{y}_k)`
# represents the kernel vector of the measurement at time :math:`k`, :math:`Q_k` is the kernel
# Kalman gain operator.
#
# Proposal generated in updater
# """""""""""""""""""""""""""""
#
# :math:`{\color{blue}\blacksquare}` The AKKF replaces :math:`X_{k} = \mathbf{x}_k^{\{i=1:M\}}`
# with new weighted proposal particles :math:`\tilde{X}_{k} = \tilde{\mathbf{x}}_k^{\{i=1:M\}}`,
# where
#
# .. math::
#     \tilde{\mathbf{x}}_k^{\{i=1:M_{\text{A}}\}} &\sim
#     \mathcal{N}\left(\mathbb{E}\left(X_{k}\right), \mathrm{Cov}\left(X_{k}\right)\right)\\
#     \mathbb{E}\left(X_{k}\right) &= X_{k}\mathbf{w}^{+}_{k}\\
#     \mathrm{Cov}\left(X_{k}\right) &= X_{k}S^{+}_{k} X_{k}^{\rm{T}}.
#
# .. image:: ../../_static/AKKF_flow_diagram.png
#   :width: 800
#   :alt: Flow diagram of the AKKF
#
# Figure 2: Flow diagram of the AKKF

# %%
# Nearly-constant velocity & Bearing-only  Tracking (BOT) example
# ---------------------------------------------------------------
#
# We consider the BOT problem with one object moving in a 2-D space. The moving trajectory
# follows a nearly-constant velocity motion mode, as in the previous tutorials.
# The measurement is the actual bearing with an additional Gaussian error term that is non-linear.
#
# Ground truth
# ^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta

np.random.seed(50)

# %%
# Initialise Stone Soup ground truth and transition models.

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

start_time = datetime.now().replace(microsecond=0)

q_x = 1 * 10 ** (-4)
q_y = 1 * 10 ** (-4)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])

# %%
# Create the ground truth path.

truth = GroundTruthPath([GroundTruthState([-0.3, 0.001, 0.7, -0.055], timestamp=start_time)])

num_steps = 30
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))

# %%
# Plot the ground truth trajectory.

from stonesoup.plotter import Plotter
plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Initialise the bearing using the appropriate measurement model.

from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing
measurement_model = Cartesian2DToBearing(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.005 ** 2]])
    )

# %%
# Generate the set of bearing only measurements

from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement,
                                  timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# %%
# Set up the AKKF
# ^^^^^^^^^^^^^^^
#
# We have designed the key new components required in Stone Soup for the AKKF to run, including
# defining different component types, such as the :class:`KernelParticleState`, :class:`Kernel`,
# :class:`AdaptiveKernelKalmanPredictor` and :class:`AdaptiveKernelKalmanUpdater`. Stone Soup
# provides the design choice of the :class:`Kernel` class to enable the use of different kernels
# and will permit the AKKF to be used for a wide variety of dynamic and measurement models, as
# well as future extensions for joint tracking and parameter estimation problems.

# %%
# :class:`KernelParticleState`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`KernelParticleState` inherits the functionality of the :class:`ParticleState` and
# adds the kernel weight covariance matrix property, i.e., :meth:`kernel_covar`.
#
# :class:`Kernel`
# ^^^^^^^^^^^^^^^
#
# The :class:`Kernel` class provides a transformation from state space into the RKHS.
# The kernel can be either a polynomial or a Gaussian kernel.
# The polynomial kernels, :class:`QuadraticKernel` and :class:`QuarticKernel`, have the following
# properties:
#
# - :attr:`c`: a parameter that trades off the influence of higher - order versus lower - order
#   terms in the polynomial.
# - :attr:`ialpha`: the inverse of :math:`\alpha` and is the slope parameter that controls the
#   influence of the dot product on the kernel value.
#
# The Gaussian kernel, :class:`GaussianKernel` has the following property
#
# - :attr:`variance`: the variance parameter of the Gaussian kernel.
#
# :class:`AdaptiveKernelKalmanPredictor`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`AdaptiveKernelKalmanPredictor` is a subclass of :class:`KalmanPredictor` and inherits
# the methods and properties of the :class:`KalmanPredictor`.
# The :class:`AdaptiveKernelKalmanPredictor` includes the following properties:
#
# - :attr:`lambda_predictor`: :math:`\lambda_{\tilde{K}}`, is a regularisation parameter to
#   stabilise the inverse of the Gram matrix of state particles' feature mappings.
#   It typically falls within the proper range of :math:`\left[10^{-4}, 10^{-2}\right]`.
# - :attr:`kernel`: the :class:`Kernel` class which is chosen to be used to map the state space
#   into the RKHS.
#
# :class:`AdaptiveKernelKalmanUpdater`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`AdaptiveKernelKalmanUpdater` is a subclass of :class:`KalmanUpdater` and inherits the
# methods and properties of the :class:`KalmanUpdater`. The :class:`AdaptiveKernelKalmanUpdater`
# includes the following new properties:
#
# - :attr:`lambda_updater`: :math:`\kappa` is a regularisation parameter to ensure the inverse of
#   :math:`G_{\mathbf{y}\mathbf{y}} S_k ^ {-}` is well - defined.
# - :attr:`kernel`: the `Kernel` class which is chosen to be used to map the measurement space into
#   the kernel space.

# %%
# Initialise a prior
# ------------------
#
# To begin, we initialise a prior estimate for the tracking problem.
# This is achieved by :class:`KernelParticleState` that describes the state as a distribution of
# particles, each associated with StateVectors and corresponding weights.
# The prior state is sampled from a Gaussian distribution.

from scipy.stats import multivariate_normal

from stonesoup.types.state import KernelParticleState
from stonesoup.types.array import StateVectors

number_particles = 30

# Sample from the prior Gaussian distribution
np.random.seed(50)
samples = multivariate_normal.rvs(np.squeeze(truth[0].state_vector),
                                  np.diag([0.1, 0.005, 0.1, 0.01])**2,
                                  size=number_particles)

# Create prior particle state.
prior = KernelParticleState(state_vector=StateVectors(samples.T),
                            weight=np.array([1/number_particles]*number_particles),
                            timestamp=start_time,
                            )

from stonesoup.kernel import QuadraticKernel
from stonesoup.predictor.kernel import AdaptiveKernelKalmanPredictor
predictor = AdaptiveKernelKalmanPredictor(transition_model=transition_model,
                                          kernel=QuadraticKernel(c=1, ialpha=1))

from stonesoup.updater.kernel import AdaptiveKernelKalmanUpdater
updater = AdaptiveKernelKalmanUpdater(measurement_model=measurement_model,
                                      kernel=QuadraticKernel(c=1, ialpha=1))

# %%
# Run the tracker
# ---------------
#
# Now, we execute the predict and update steps of the tracker. The Predictor module takes both
# the prior and proposal information to propagate the state particles, and it calculates the
# predictive kernel weight vector and covariance matrix.
#
# Next, the Updater module uses the predictions to generate the measurement particles and updates
# the posterior kernel weight vector and covariance matrix accordingly.
# Additionally, the Updater generates new proposal particles at every step to refine the state
# estimate.

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plot the resulting track with the sample points at each iteration.

plotter = Plotter()
plotter.plot_ground_truths(truth, [0, 2], linewidth=3.0, color='black')
plotter.plot_tracks(track, [0, 2], label='AKKF - quadratic', color='royalblue')
plotter.fig

# %%

# sphinx_gallery_thumbnail_number = -1
plotter.plot_tracks(track, [0, 2], particle=True)
plotter.fig
