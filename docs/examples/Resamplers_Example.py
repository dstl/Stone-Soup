#!/usr/bin/env python
# coding: utf-8

"""
===================================
Particle Filter Resamplers: Example
===================================
"""

# %%
# Introduction
# ------------
# The Stone-Soup package comes with a number of resamplers, for the Particle Filter, that can be
# used straight out of the box. This example explains each of the resamplers and compares the
# results of using each one.
#
#
# **Resamplers currently available in Stone-Soup:**
#
# - Systematic resampler
# - Multinomial resampler
# - Stratified resampler

# %%
# Plotter for this notebook
# """""""""""""""""""""""""

import matplotlib.pyplot as plt


def plot(norm_weights, u_j=None, stratified=False):
    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 1.5)
    for i, particle_weight in enumerate(norm_weights):
        left = 0
        for j in range(i):
            left += norm_weights[j]
        ax.barh(['CDF'], [particle_weight], left=left)
    if u_j is not None:
        for point in u_j:
            plt.plot([u_j, u_j], [-0.4, 0.4], 'k-', lw=4)
    if stratified is not False:
        n_parts = len(norm_weights)
        s_lbs = np.arange(n_parts) * (1 / n_parts)
        for lb in s_lbs:
            plt.plot([lb, lb], [-0.4, 0.4], 'w:', lw=2)

# %%
# Cumulative Distribution Function
# --------------------------------
#
# The Systematic, Multinomial, and Stratified resamplers all use a cumulative distribution
# function (CDF) to randomly pick points from. The CDF is calculated from the cumulative sum
# of the normalised weights of the existing particles.
#
# **Example**
#
# Given the following set of particles and their weights:
#
# .. list-table:: Particle Weights
#    :widths: 25 25 50
#    :header-rows: 1
#
#    * - Particle
#      - Weight
#      - Normalised Weight
#    * - 1
#      - 1
#      - 0.1
#    * - 2
#      - 5
#      - 0.5
#    * - 3
#      - 0.5
#      - 0.05
#    * - 4
#      - 1.5
#      - 0.15
#    * - 5
#      - 2
#      - 0.2
#
# We would calculate the following CDF:


particle_weights = [1, 5, 1.5, 0.5, 2]
n_particles = len(particle_weights)
normalised_weights = [particle/sum(particle_weights) for particle in particle_weights]
plot(normalised_weights)

# %%
# Each of the resamplers then use a different method to pick points in the CDF. For example,
# assume the following points are picked: 0.15, 0.41, 0.57, 0.63, and 0.89. These points are
# shown on the CDF below.


u_j = [0.15, 0.41, 0.57, 0.63, 0.89]
plot(normalised_weights, u_j)

# %%
# As shown above, the points 0.15, 0.41, and 0.57, all fall within the section of the CDF
# corresponding to particle 2, while the points 0.63 and 0.89 fall within the sections of
# the CDF corresponding to particles 3 and 5 respectively.
#
# Hence, for this example, the number of times each particle is resampled is as follows:
#
# .. list-table:: Particle Weights
#    :widths: 25 25 50
#    :header-rows: 1
#
#    * - Particle
#      - No. of times resampled
#      - Weight of new resampled particles
#    * - 1
#      - 0
#      - 0.2
#    * - 2
#      - 3
#      - 0.2
#    * - 3
#      - 1
#      - 0.2
#    * - 4
#      - 0
#      - 0.2
#    * - 5
#      - 1
#      - 0.2
#
# As shown in the table, the resampler assigns a new weight to each sample - by default, each of
# the resamplers included in Stone-Soup give all particles an equal weight of :math:`1/N` where
# :math:`N` = no. of resampled particles.

# %%
# Resamplers
# ----------

# %%
# Multinomial Resampler
# """""""""""""""""""""
#
# The Multinomial resampler calculates :math:`N` independent random numbers from the uniform
# distribution :math:`u \sim U(0,1]`, where :math:`N` is the target number of particles to
# be resampled. In most cases, we use :math:`N = M`, where :math:`M` is the number of existing
# particles to be resampled. However, the Multinomial resampler can upsample (:math:`N > M`) or
# downsample (:math:`N < M`) to a value :math:`N \neq M \in \mathbb{N}`. The Multinomial resampler
# has a computational complexity of :math:`O(MN)` where :math:`N` and :math:`M` are the number of
# resampled and existing particles respectively.
#
# An example of how the Multinomial resampler picks points from the CDF is shown below. Black
# lines represent the chosen points.


import numpy as np

# Pick :math:`N` random points in the uniform distribution :math:`u~U(0,1]`
u_j = np.random.rand(5)

plot(normalised_weights, u_j)

# %%
# Systematic Resampler
# """"""""""""""""""""
#
# Unlike the Multinomial resampler, the Systematic resampler doesn't calculate all points
# independently. Instead, a single, random starting point is chosen. :math:`N` points are then
# calculated at equidistant intervals along the CDF, so that there is a gap of :math:`1/N`
# between any two consecutive points. The Systematic resampler has a computational complexity
# of :math:`O(N)` where :math:`N` is the number of resampled particles.
#
# An example of how the Systematic resampler picks points from the CDF is shown below. Black
# lines represent the chosen points.


# Pick a starting point
s = np.random.uniform(0, 1/n_particles)

# Calculate N equidistant points from the starting point
u_j = s + (1 / n_particles) * np.arange(n_particles)

plot(normalised_weights, u_j)

# %%
# Stratified Resampler
# """"""""""""""""""""
#
# The Stratified resampler splits the whole CDF into :math:`N` evenly sized strata
# (subpopulations). A random point is then chosen, independently, from each stratum. This results
# in the gap between any two consecutive points will be in the range :math:`(0, 2/N]`. The
# Stratified resampler has a computational complexity of :math:`O(N)` where :math:`N` is the
# number of resampled particles.
#
# An example of how the Stratified resampler picks points from the CDF is shown below. Black
# lines represent the chosen points, with white dashed lines representing the strata boundaries.


# Calculate lower bound of each stratum
s_lb = np.arange(n_particles) * (1 / n_particles)

# Independently pick a point from each stratum
u_j = np.random.uniform(s_lb, s_lb + (1 / n_particles))

plot(normalised_weights, u_j, s_lb)

# %%
# Example in Stone-Soup
# ---------------------
#
# Generate some particles
# """""""""""""""""""""""
#
# An example of resampling particles using the Stone-Soup package. We generate some particles
# using the Particle class from Stone-Soup. In this example, we give every particle an equal
# weight - each particle will have the same likelihood of being resampled. The resample
# function returns a ParticleState object, within which ParticleState.state_vector contains a
# list of particles to be resampled.


from stonesoup.types.particle import Particle

particles = [Particle(np.array([[i]]), weight=1/5) for i in range(5)]

# %%
# Specify and run resampler
# """""""""""""""""""""""""


from stonesoup.resampler.particle import MultinomialResampler

# Resampler
resampler = MultinomialResampler()

# Resample particles
resampled_particles = resampler.resample(particles)

print(resampled_particles.state_vector)
print(resampled_particles.weight)

# %%
# Here we can see which particles were chosen to be resampled, and the weights of the new
# particles.
