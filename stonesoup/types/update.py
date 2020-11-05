# -*- coding: utf-8 -*-
from ..base import Property
from .base import Type
from .hypothesis import Hypothesis
from .state import State, GaussianState, ParticleState, SqrtGaussianState
from .mixture import GaussianMixture


class Update(Type):
    """ Update type

    The base update class. Updates are returned by :class:'~.Updater' objects
    and contain the information that was used to perform the updating"""

    hypothesis: Hypothesis = Property(doc="Hypothesis used for updating")


class StateUpdate(Update, State):
    """ StateUpdate type

    Most simple state update type, where everything only has time
    and a state vector. Requires a prior state that was updated,
    and the hypothesis used to update the prior.
    """


class GaussianStateUpdate(Update, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class SqrtGaussianStateUpdate(Update, SqrtGaussianState):
    """ SqrtGaussianStateUpdate type

    This is equivalent to a Gaussian state update object, but with the
    covariance of the Gaussian distribution stored in matrix square root
    form.
    """


class GaussianMixtureUpdate(Update, GaussianMixture):
    """ GaussianMixtureUpdate type

    This is a Gaussian mixture update object, which, as the name
    suggests, is described by a Gaussian mixture.
    """


class ParticleStateUpdate(Update, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state update object.
    """
