# -*- coding: utf-8 -*-
import datetime
from collections import abc
from typing import MutableSequence, List

import numpy as np
import uuid

from ..base import Property
from ..functions import gm_reduce_single
from .array import StateVector, CovarianceMatrix, StateVectors
from .base import Type
from .particle import Particles
from .numeric import Probability


class State(Type):
    """State type.

    Most simple state type, which only has time and a state vector."""
    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the state. Default None.")
    state_vector: StateVector = Property(doc='State vector.')

    def __init__(self, state_vector, *args, **kwargs):
        # Don't cast away subtype of state_vector if not necessary
        if state_vector is not None \
                and not isinstance(state_vector, StateVector):
            state_vector = StateVector(state_vector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]


class StateMutableSequence(Type, abc.MutableSequence):
    """A mutable sequence for :class:`~.State` instances

    This sequence acts like a regular list object for States, as well as
    proxying state attributes to the last state in the sequence. This sequence
    can also be indexed/sliced by :class:`datetime.datetime` instances.

    Example
    -------
    >>> t0 = datetime.datetime(2018, 1, 1, 14, 00)
    >>> t1 = t0 + datetime.timedelta(minutes=1)
    >>> state0 = State([[0]], t0)
    >>> sequence = StateMutableSequence([state0])
    >>> print(sequence.state_vector, sequence.timestamp)
    [[0]] 2018-01-01 14:00:00
    >>> sequence.append(State([[1]], t1))
    >>> for state in sequence[t1:]:
    ...     print(state.state_vector, state.timestamp)
    [[1]] 2018-01-01 14:01:00
    """

    states: MutableSequence[State] = Property(
        default=None,
        doc="The initial list of states. Default `None` which initialises with empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        elif not isinstance(states, abc.Sequence):
            # Ensure states is a list
            states = [states]
        super().__init__(states, *args, **kwargs)

    def __len__(self):
        return self.states.__len__()

    def __setitem__(self, index, value):
        return self.states.__setitem__(index, value)

    def __delitem__(self, index):
        return self.states.__delitem__(index)

    def __getitem__(self, index):
        if isinstance(index, slice) and (
                isinstance(index.start, datetime.datetime)
                or isinstance(index.stop, datetime.datetime)):
            items = []
            for state in self.states:
                try:
                    if index.start and state.timestamp < index.start:
                        continue
                    if index.stop and state.timestamp >= index.stop:
                        continue
                except TypeError as exc:
                    raise TypeError(
                        'both indices must be `datetime.datetime` objects for'
                        'time slice') from exc
                items.append(state)
            return StateMutableSequence(items[::index.step])
        elif isinstance(index, datetime.datetime):
            for state in reversed(self.states):
                if state.timestamp == index:
                    return state
            else:
                raise IndexError('timestamp not found in states')
        elif isinstance(index, slice):
            return StateMutableSequence(self.states.__getitem__(index))
        else:
            return self.states.__getitem__(index)

    def __getattribute__(self, name):
        # This method is called if we try to access an attribute of self. First we try to get the
        # attribute directly, but if that fails, we want to try getting the same attribute from
        # self.state instead. If that, in turn,  fails we want to return the error message that
        # would have originally been raised, rather than an error message that the State has no
        # such attribute.
        #
        # An alternative mechanism using __getattr__ seems simpler (as it skips the first few lines
        # of code, but __getattr__ has no mechanism to capture the originally raised error.
        try:
            # This tries first to get the attribute from self.
            return Type.__getattribute__(self, name)
        except AttributeError as original_error:
            if name.startswith("_"):
                # Don't proxy special/private attributes to `state`, just raise the original error
                raise original_error
            else:
                # For non _ attributes, try to get the attribute from self.state instead of self.
                try:
                    my_state = Type.__getattribute__(self, 'state')
                    return getattr(my_state, name)
                except AttributeError:
                    # If we get the error about 'State' not having the attribute, then we want to
                    # raise the original error instead
                    raise original_error

    def insert(self, index, value):
        return self.states.insert(index, value)

    @property
    def state(self):
        return self.states[-1]


class GaussianState(State):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.
    """
    covar: CovarianceMatrix = Property(doc='Covariance matrix of state.')

    def __init__(self, state_vector, covar, *args, **kwargs):
        # Don't cast away subtype of covar if not necessary
        if not isinstance(covar, CovarianceMatrix):
            covar = CovarianceMatrix(covar)
        super().__init__(state_vector, covar, *args, **kwargs)
        if self.state_vector.shape[0] != self.covar.shape[0]:
            raise ValueError(
                "state vector and covariance should have same dimensions")

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector


class SqrtGaussianState(State):
    """A Gaussian State type where the covariance matrix is stored in a form :math:`W` such that
    :math:`P = WW^T`

    For :math:`P` in general, :math:`W` is not unique and the user may choose the form to their
    taste. No checks are undertaken to ensure that a sensible square root form has been chosen.

    """
    sqrt_covar: CovarianceMatrix = Property(doc="A square root form of the Gaussian covariance "
                                                "matrix.")

    def __init__(self, state_vector, sqrt_covar, *args, **kwargs):
        sqrt_covar = CovarianceMatrix(sqrt_covar)
        super().__init__(state_vector, sqrt_covar, *args, **kwargs)

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector

    @property
    def covar(self):
        """The full covariance matrix.

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The covariance matrix calculated via :math:`W W^T`, where :math:`W` is a
            :class:`~.SqrtCovarianceMatrix`

        """
        return self.sqrt_covar @ self.sqrt_covar.T
GaussianState.register(SqrtGaussianState)  # noqa: E305


class WeightedGaussianState(GaussianState):
    """Weighted Gaussian State Type

    Gaussian State object with an associated weight.  Used as components
    for a GaussianMixtureState.
    """
    weight: Probability = Property(default=0, doc="Weight of the Gaussian State.")

    @property
    def gaussian_state(self):
        """The Gaussian state."""
        return GaussianState(self.state_vector,
                             self.covar,
                             timestamp=self.timestamp)

    @classmethod
    def from_gaussian_state(cls, gaussian_state, *args, copy=True, **kwargs):
        r"""
        Returns a WeightedGaussianState instance based on the gaussian_state.

        Parameters
        ----------
        gaussian_state : :class:`~.GaussianState`
            The guassian_state used to create the new WeightedGaussianState.
        \*args : See main :class:`~.WeightedGaussianState`
            args are passed to :class:`~.WeightedGaussianState` __init__()
        copy : Boolean, optional
            If True, the WeightedGaussianState is created with copies of the elements
            of gaussian_state. The default is True.
        \*\*kwargs : See main :class:`~.WeightedGaussianState`
            kwargs are passed to :class:`~.WeightedGaussianState` __init__()

        Returns
        -------
        :class:`~.WeightedGaussianState`
            Instance of WeightedGaussianState.
        """
        state_vector = gaussian_state.state_vector
        covar = gaussian_state.covar
        timestamp = gaussian_state.timestamp
        if copy:
            state_vector = state_vector.copy()
            covar = covar.copy()
        return cls(
            state_vector=state_vector,
            covar=covar,
            timestamp=timestamp,
            *args, **kwargs
        )


class TaggedWeightedGaussianState(WeightedGaussianState):
    """Tagged Weighted Gaussian State Type

    Gaussian State object with an associated weight and tag. Used as components
    for a GaussianMixtureState.
    """
    tag: str = Property(default=None, doc="Unique tag of the Gaussian State.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tag is None:
            self.tag = str(uuid.uuid4())


class GaussianMixtureState(Type):
    """Gaussian Mixture state"""
    components: List[WeightedGaussianState] = Property(
                    doc='Gaussian Mixture components')

    @property
    def ndim(self):
        return self.components[0].ndim

    @property
    def timestamp(self):
        return self.components[0].timestamp

    @property
    def means(self):
        means = [component.mean for component in self.components]
        return np.concatenate((means),1)

    @property
    def covars(self):
        covars= [component.covar for component in self.components]
        return np.array(covars)

    @property
    def weights(self):
        return np.array([[component.weight] for component in self.components])

    @property
    def mean(self):
        means = self.means
        weights = self.weights
        return means@weights

    @property
    def covar(self):
        covars = np.moveaxis(self.covars,0,2)
        means = self.means.view(StateVectors)
        #_, covar = gm_reduce_single(self.means.T, self.covars,
        #                            self.weights.ravel())
        _, covar = gm_reduce_single(means, covars, self.weights.ravel())
        return covar

    @property
    def state_vector(self):
        return self.mean

class ParticleState(Type):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles"""

    particles: Particles = Property(doc='All particles.')
    fixed_covar: CovarianceMatrix = Property(default=None,
                                             doc='Fixed covariance value. Default `None`, where'
                                                 'weighted sample covariance is then used.')
    timestamp: datetime.datetime = Property(default=None,
                                            doc="Timestamp of the state. Default None.")

    def __init__(self, particles, *args, **kwargs):
        if particles is not None and not isinstance(particles, Particles):
            particles = Particles(particle_list=particles)
        super().__init__(particles, *args, **kwargs)

    @property
    def ndim(self):
        return self.particles.ndim

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        result = np.average(self.particles.state_vector,
                            axis=1,
                            weights=self.particles.weight)
        # Convert type as may have type of weights
        return result

    @property
    def state_vector(self):
        """The mean value of the particle states"""
        return self.mean

    @property
    def covar(self):
        if self.fixed_covar is not None:
            return self.fixed_covar
        cov = np.cov(self.particles.state_vector, ddof=0, aweights=np.array(self.particles.weight))
        # Fix one dimensional covariances being returned with zero dimension
        return cov
State.register(ParticleState)  # noqa: E305
