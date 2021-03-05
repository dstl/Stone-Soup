#!/usr/bin/env python
# coding: utf-8

"""
Using a Hidden Markov Model for classification
==============================================
An example of using a hidden markov model to classify a target using Stone Soup components.
This particular forward algorithm implementation is motivation for introducing classifiers in to
Stone Soup.
"""

# %%
# Modelling the state space
# -------------------------
# We model a target that can be one of three categories: Bicycle, Car or Bus.
# In this simple instance, the target does not switch between these categories (ie. if it is a bus,
# it stays a bus).
# This is modelled in the state space by 3 state vector elements. The first component indicates
# the probability that the target is a bicycle, the second that it is a car, and the third that it
# is a bus.
# For this model, it will remain a bus, therefore the 3rd state vector component will be
# :math:`1` throughout, with the other components remaining :math:`0`.

import numpy as np

from stonesoup.types.state import State, StateVector

np.random.seed(1991)

gt = []
nsteps = 100

for step in range(nsteps):
    gt.append(State(StateVector([0, 0, 1])))  # bicycle, car, bus

# %%
[np.argmax(x.state_vector) for x in gt]

# %%
# Modelling state transition
# --------------------------
# We require a model of the state transition in order to track the target's classification.
# This will be a simple linear map, represented by a :math:`3\times 3` matrix.
# As we understand that our target does not change class, we use the identity matrix in this model.
# Transition uncertainty will be accounted for by additive noise. Transitioned states are
# normalised.
#
# .. math::
#       T = \begin{bmatrix}
#  		P(Bicycle_{t}|Bicycle_{t-1}) & P(Bicycle_{t}|Car_{t-1}) & P(Bicycle_{t}|Bus_{t-1}) \\
#  		P(Car_{t}|Bicycle_{t-1}) & P(Car_{t}|Car_{t-1}) & P(Car_{t}|Bus_{t-1}) \\
#  		P(Bus_{t}|Bicycle_{t-1}) & P(Bus_{t}|Car_{t-1}) & P(Bus_{t}|Bus_{t-1})
#  	    \end{bmatrix}
#
# The transitioned state is given by
# :math:`f(X_{t-1})_{j} = T_{ij}{X_{t-1}}_{i} = T^T_{ji}{X_{t-1}}_{i}`, where
# :math:`T_{ij} = P(class(j)_{t} | class(i)_{t-1})`.
#
# Therefore, we take the transpose :math:`f(X_{t-1}) = T^{T}X_{t-1}` when transitioning the state.
#
# Consider:
#
# .. math::
#       P(Bus_{t}) &= P(Bus_{t}|Bicycle_{t-1})P(Bicycle_{t-1}) + \\
#            &\quad  P(Bus_{t}|Car_{t-1})P(Car_{t-1}) + \\
#            &\quad  P(Bus_{t}|Bus_{t-1})P(Bus_{t-1}) \\
#            &= (P(Bus_{t}|Bicycle_{t-1}), P(Bus_{t}|Car_{t-1}), P(Bus_{t}|Bus_{t-1}))
#            \begin{pmatrix}P(Bicycle_{t-1}) \\P(Car_{t-1}) \\P(Bus_{t-1})\end{pmatrix} \\
#            &= T_{i2}X_{t-1} \\
#            &= T_{2i}^{T}X_{t-1}
#
# Noise will be modelled by a small probability that a target may switch category.

F = np.eye(3)

transit_noise = np.array([[0.01, 0.02, 0.02],
                          [0.02, 0.01, 0.02],
                          [0.02, 0.02, 0.01]])


# Off-diagonal elements are larger, leading to slight increases in alternate category
# probabilities.


def transit(state, noise=False):
    x = F.T @ state.state_vector

    if noise:
        row = transit_noise @ x

        x = x + StateVector(row)

        x = x / np.sum(x)

    return x


# %%
# Modelling observations
# ----------------------
# We do not observe the state directly (the hidden state). We observe 'emissions' governed by a
# matrix of probabilities :math:`E`.
# Observations of the target may be incorrect, and affected by various input from a sensor.
# We simply model an observer of the target's size, with an understanding of the underlying
# distribution of bicycle, car and bus targets that are/could be small or large.
# This is defined in the emission matrix, utilised by the observer model to determine a measurement
# which says whether the target is small or large. Similar to the state space categories, this is
# represented by a detection state vector with 2 components, either 0 or 1.
# To determine which classification is observed, we randomly sample from the multinomial
# distribution defined by the row of the emission matrix corresponding to the most likely state of
# the target. We will make the emission matrix time-invariant.
#
# .. math::
#       E_{t} = E = \begin{bmatrix}
#         		P(Small_{t} | Bicycle_{t}) & P(Large_{t} | Bicycle_{t})\\
#         		P(Small_{t} | Car_{t}) & P(Large_{t} | Car_{t})\\
#         		P(Small_{t} | Bus_{t}) & P(Large_{t} | Bus_{t})
#         	    \end{bmatrix}
#
# where a measurement is sampled from the distribution given by:
#
# .. math::
#       y_{t} &= \begin{pmatrix}P(Small_{t})\\P(Large_{t})\end{pmatrix} \\
#             &= \begin{pmatrix}
#             P(Small_{t}|Bicycle_{t})P(Bicycle_{t})+
#             P(Small_{t}|Car_{t})P(Car_{t})+
#             P(Small_{t}|Bus_{t})P(Bus_{t}) \\
#             P(Large_{t}|Bicycle_{t})P(Bicycle_{t})+
#             P(Large_{t}|Car_{t})P(Car_{t})+
#             P(Large_{t}|Bus_{t})P(Bus_{t})
#             \end{pmatrix} \\
#             &= E^{T}X_{t}
#
# The sensor is modelled to return a definitive category. I.E. 'The target is small' or 'the target
# is large', and not probabilities of each. Therefore we sample the resulting multinomial
# distribution.


E = np.array([[0.89, 0.11],
              [0.3, 0.7],
              [0.1, 0.9]])

# ie. bicycles are measured to be small 89% of the time, and large 11% of the time.
# Perhaps these values could be more similar in value to model a less certain / noisier sensor?
# Additive noise could also be considered.

import scipy


def _sample(row):
    rv = scipy.stats.multinomial(n=1, p=row)
    return rv.rvs(size=1, random_state=None)


def observe(state):
    y = E.T @ state.state_vector

    y = y / np.sum(y)

    sample = _sample(y.flatten())

    return StateVector(sample)


# %%

from stonesoup.types.detection import Detection

observations = []
for i in range(0, nsteps):
    observations.append(Detection(observe(gt[i])))

# %%
# Tracking classification
# -----------------------
# Posterior state is given by :math:`X_{t} = EZ_{t} * T^T X_{t-1}`, where :math:`X_{t}` is the
# state estimate at time :math:`t`, :math:`E` the emission matrix, :math:`Z_{t}` an observation at
# time :math:`t` and :math:`T` the transition model matrix, where :math:`*` notates piecewise
# vector multiplication.
# We begin with a prior state guess, whereby we have no knowledge of the target's classifcaiton,
# and therefore appoint equal probability to each category.

prior = State(StateVector([1 / 3, 1 / 3, 1 / 3]))

# %%
# Next, we carry-out the tracking loop, making sure to normalise the posterior estimate at each
# iteration, as track state vector components represent a categorical distribution (they must
# therefore sum to 1).
# A similarity to the traditional tracking loop of prediction and update can be seen: whereby the
# prior state is predicted forwards via application of a transition model :math:`TX`, and the
# resultant prediction is updated using an observation mapped to the state space :math:`EY` via
# pointwise product of the two vectors.
#
# .. math::
#       posterior_=^{i} &= P(class^{i}_{t})\\
# 			  &= P(obs^{j}_{t})P(obs^{j}_{t}|class^{i}_{t})
#                P(class^{i}_{t}|class^{k}_{t-1})P(class^{k}_{t-1})\\
# 			  &= Y^{j}_{t-1}E^{ij}_{t-1}T^{ki}_{t-1}X^{k}_{t-1}\\
# 			  &= (EY)^{i}(T^{T}X)^{i}

from stonesoup.types.track import Track

track = Track()
for observation in observations:
    TX = transit(prior)  # 'Predict'
    EY = E @ observation.state_vector

    prenormalise = np.multiply(TX, EY)  # 'Update'

    normalise = prenormalise / np.sum(prenormalise)

    track.append(State(normalise))
    prior = track[-1]

# %%
# Plotting
# --------
# We plot the track classification as a stacked bar graph. Green indicates bicycle, white car, and
# red bus. The larger a particular bar, the greater the probability that the track at that time
# is of the corresponding classification.
# Observations are plotted underneath this graph. Light blue indicates an observation that the
# target is small, blue that it is medium, and dark blue that it is large.

import matplotlib.pyplot as plt

track_bicycle, track_car, track_bus = np.array([list(state.state_vector) for state in track]).T

d_colours = []
for s in observations:
    sv = s.state_vector
    if sv[0]:
        d_colours.append('lightblue')
    elif sv[1]:
        d_colours.append('blue')
    else:
        d_colours.append('darkblue')
d = len(track_bicycle) * [-0.05]

ind = np.arange(nsteps)
width = 1
p1 = plt.bar(ind, track_bicycle, width, color='g')
p2 = plt.bar(ind, track_car, width, bottom=track_bicycle, color='w')
p3 = plt.bar(ind, track_bus, width, bottom=[track_bicycle[i] + track_car[i]
                                            for i in range(len(track_bicycle))], color='r')
p4 = plt.bar(ind, d, width, color=d_colours)

truth_index = np.argmax(gt[0].state_vector)
category = ['bicycle', 'car', 'bus'][truth_index]
emit = str(E[truth_index])
title = 'GT: ' + category + ', E: ' + emit

plt.title(title)
plt.legend((p1[0], p2[0], p3[0]), ('Bicycle', 'Car', 'Bus'))
