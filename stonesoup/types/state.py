# -*- coding: utf-8 -*-
import datetime
import uuid
from collections import abc
from typing import MutableSequence, Any, Optional, Sequence
import typing

import numpy as np

from ..base import Property
from .array import StateVector, CovarianceMatrix, PrecisionMatrix, StateVectors
from .base import Type
from .particle import Particle
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
                and not isinstance(state_vector, (StateVector, StateVectors)):
            state_vector = StateVector(state_vector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]

    @staticmethod
    def from_state(state: 'State', *args: Any, target_type: Optional[typing.Type] = None,
                   **kwargs: Any) -> 'State':
        """Class utility function to create a new state (or compatible type) from an existing
        state. The type and properties of this new state are defined by `state` except for any
        explicitly overwritten via `args` and `kwargs`.

        It acts similarly in feel to a copy constructor, with the optional over-writing of
        properties.

        Parameters
        ----------
        state: State
            :class:`~.State` to use existing properties from, and identify new state-type from.
        \\*args: Sequence
            Arguments to pass to newly created state, replacing those with same name in `state`.
        target_type: Type,  optional
            Optional argument specifying the type of of object to be created. This need not
            necessarily be :class:`~.State` subclass. Any arguments that match between the input
            `state` and the target type will be copied from the old to the new object (except those
            explicitly specified in `args` and `kwargs`.
        \\*\\*kwargs: Mapping
            New property names and associate value for use in newly created state, replacing those
            on the `state` parameter.
        """
        # Handle being initialised with state sequence
        if isinstance(state, StateMutableSequence):
            state = state.state

        if target_type is None:
            target_type = type(state)

        args_property_names = {
            name for n, name in enumerate(target_type.properties) if n < len(args)}

        new_kwargs = {
            name: getattr(state, name)
            for name in type(state).properties.keys() & target_type.properties.keys()
            if name not in args_property_names}

        new_kwargs.update(kwargs)

        return target_type(*args, **new_kwargs)


class CreatableFromState:
    class_mapping = {}

    def __init_subclass__(cls, **kwargs):
        bases = cls.__bases__
        if CreatableFromState in bases:
            # Direct subclasses should not be added to the class mapping, only subclasses of
            # subclasses
            return
        if len(bases) != 2:
            raise TypeError('A CreatableFromState subclass must have exactly two superclasses')
        base_class, state_type = cls.__bases__
        if not issubclass(base_class, CreatableFromState):
            raise TypeError('The first superclass of a CreatableFromState subclass must be a '
                            'CreatableFromState (or a subclass)')
        if not issubclass(state_type, State):
            # Non-state subclasses do not need adding to the class mapping, as they should not
            # be created from States
            return
        if base_class not in CreatableFromState.class_mapping:
            CreatableFromState.class_mapping[base_class] = {}
        CreatableFromState.class_mapping[base_class][state_type] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_state(
            cls,
            state: State,
            *args: Any,
            target_type: Optional[type] = None,
            **kwargs: Any) -> 'State':
        """
        Return new object instance of suitable type from an existing `state`.
        The type returned can be explicitly specified using the `target_type` argument, otherwise
        it is chosen by introspection of the created subclasses of this type: see below for an
        example. Any compatible properties are copied from the input `state` to the returned
        object, except for those specified by `args` and `kwargs`, which take precedence over those
        from the input `state`.

        This method is primarily concerned with type selection, with actual copying performed by
        the static :meth:`~.State.from_state` method. As an example of the type selection
        algorithm, consider the case of the class
        `GaussianStatePrediction(Prediction, GaussianState)`. This is subclass of `Prediction`,
        and `GaussianState` and so the `class_mapping` property will have an entry added (when
        `GaussianStatePrediction` is defined) such that
        `class_mapping[Prediction][GaussianState] = GaussianStatePrediction`. If this method is
        then called like below

        >>>> gaussian_state = GaussianState(some_arguments)
        >>>> new_prediction = Prediction.from_state(gaussian_state, *args, **kwargs)

        then the `from_state` method will look up the class mapping and see that
        `Prediction.from_state()` called with a `GaussianState` input should return a
        `GaussianStatePrediction` object, and therefore the type of `new_prediction` will be
        `GaussianStatePrediction`

        The functionality is currently used by :class:`~.Prediction` and :class:`~.Updater`
        objects.

        Parameters
        ----------
        state: State
            :class:`~.State` to use existing properties from, and identify prediction type from
        \\*args: Sequence
            Arguments to pass to newly created prediction, replacing those with same name on
            ``state`` parameter.
        target_type: Type, optional
            Type to use for prediction, overriding one from :attr:`class_mapping`.
        \\*\\*kwargs: Mapping
            New property names and associate value for use in newly created prediction, replacing
            those on the ``state`` parameter.
        """
        # Handle being initialised with state sequence
        if isinstance(state, StateMutableSequence):
            state = state.state
        try:
            state_type = next(type_ for type_ in type(state).mro()
                              if type_ in CreatableFromState.class_mapping[cls])
        except StopIteration:
            raise TypeError(f'{cls.__name__} type not defined for {type(state).__name__}')
        if target_type is None:
            target_type = CreatableFromState.class_mapping[cls][state_type]

        return State.from_state(state, *args, **kwargs, target_type=target_type)


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

    def last_timestamp_generator(self):
        """Generator yielding the last state for each timestamp

        This provides a method of iterating over a sequence of states,
        such that when multiple states for the same timestamp exist,
        only the last state is yielded. This is particularly useful in
        cases where you may have multiple :class:`~.Update` states for
        a single timestamp e.g. multi-sensor tracking example.

        Yields
        ------
        State
            A state for each timestamp present in the sequence.
        """
        state_iter = iter(self)
        current_state = next(state_iter)
        for next_state in state_iter:
            if next_state.timestamp > current_state.timestamp:
                yield current_state
            current_state = next_state
        yield current_state


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


class InformationState(State):
    r"""Information State Type

    The information state class carries the :attr:`state_vector`,
    :math:`\mathbf{y}_k = Y_k \mathbf{x}_k` and the precision or information matrix
    :math:`Y_k = P_k^{-1}`, where :math:`\mathbf{x}_k` and :math:`P_k` are the mean and
    covariance, respectively, of a Gaussian state.

    """
    precision: PrecisionMatrix = Property(doc='precision matrix of state.')


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

    BIRTH = 'birth'
    '''Tag value used to signify birth component'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tag is None:
            self.tag = str(uuid.uuid4())


class ParticleState(State):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles"""

    state_vector: StateVectors = Property(doc='State vectors.')
    weight: MutableSequence[Probability] = Property(default=None, doc='Weights of particles')
    parent: 'ParticleState' = Property(default=None, doc='Parent particles')
    particle_list: MutableSequence[Particle] = Property(default=None,
                                                        doc='List of Particle objects')
    fixed_covar: CovarianceMatrix = Property(default=None,
                                             doc='Fixed covariance value. Default `None`, where'
                                                 'weighted sample covariance is then used.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (self.particle_list is not None) and \
                (self.state_vector is not None or self.weight is not None):
            raise ValueError("Use either a list of Particle objects or StateVectors and weights,"
                             " but not both.")

        if self.particle_list and isinstance(self.particle_list, list):
            self.state_vector = \
                StateVectors([particle.state_vector for particle in self.particle_list])
            self.weight = \
                np.array([Probability(particle.weight) for particle in self.particle_list])
            parent_list = [particle.parent for particle in self.particle_list]

            if parent_list.count(None) == 0:
                self.parent = ParticleState(None, particle_list=parent_list)
            elif 0 < parent_list.count(None) < len(parent_list):
                raise ValueError("Either all particles should have"
                                 " parents or none of them should.")

        if self.parent:
            self.parent.parent = None  # Removed to avoid using significant memory

        if self.state_vector is not None and not isinstance(self.state_vector, StateVectors):
            self.state_vector = StateVectors(self.state_vector)
        if self.weight is not None and not isinstance(self.weight, np.ndarray):
            self.weight = np.array(self.weight)

    def __getitem__(self, item):
        if self.parent:
            p = self.parent[item]
        else:
            p = None

        particle = Particle(state_vector=self.state_vector[:, item],
                            weight=self.weight[item],
                            parent=p)
        return particle

    @property
    def particles(self):
        return [particle for particle in self]

    def __len__(self):
        return self.state_vector.shape[1]

    @property
    def ndim(self):
        return self.state_vector.shape[0]

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        result = np.average(self.state_vector,
                            axis=1,
                            weights=self.weight)
        # Convert type as may have type of weights
        return result

    @property
    def covar(self):
        if self.fixed_covar is not None:
            return self.fixed_covar
        cov = np.cov(self.state_vector, ddof=0, aweights=np.array(self.weight))
        # Fix one dimensional covariances being returned with zero dimension
        return cov


State.register(ParticleState)  # noqa: E305


class EnsembleState(Type):
    r"""Ensemble State type

    This is an Ensemble state object which describes the system state as a
    ensemble of state vectors for use in Ensemble based filters.

    This approach is functionally identical to the Particle state type except
    it doesn't use any weighting for any of the "particles" or ensemble members.
    All "particles" or state vectors in the ensemble are equally weighted.

    .. math::

        \mathbf{X} = [x_1, x_2, ..., x_M]

    """

    state_vector: StateVectors = Property(doc="An ensemble of state vectors which represent the "
                                              "state")
    timestamp: datetime.datetime = Property(
        default=None, doc="Timestamp of the state. Default None.")

    @classmethod
    def from_gaussian_state(self, gaussian_state, num_vectors):
        """
        Returns an EnsembleState instance, from a given
        GaussianState object.

        Parameters
        ----------
        gaussian_state : :class:`~.GaussianState`
            The GaussianState used to create the new EnsembleState.
        num_vectors : int
            The number of desired column vectors present in the ensemble.
        Returns
        -------
        :class:`~.EnsembleState`
            Instance of EnsembleState.
        """
        mean = gaussian_state.state_vector.reshape((gaussian_state.ndim,))
        covar = gaussian_state.covar
        timestamp = gaussian_state.timestamp

        return EnsembleState(state_vector=self.generate_ensemble(mean, covar, num_vectors),
                             timestamp=timestamp)

    @classmethod
    def generate_ensemble(self, mean, covar, num_vectors):
        """
        Returns a StateVectors wrapped ensemble of state vectors, from a given
        mean and covariance matrix.

        Parameters
        ----------
        mean : :class:`~.numpy.ndarray`
            The mean value of the distribution being sampled to generate
            ensemble.
        covar : :class:`~.numpy.ndarray`
            The covariance matrix of the distribution being sampled to
            generate ensemble.
        num_vectors : int
            The number of desired column vectors present in the ensemble,
            or the number of "samples".
        Returns
        -------
        :class:`~.EnsembleState`
            Instance of EnsembleState.
        """
        # This check is necessary, because the StateVector wrapper does
        # funny things with dimension.
        rng = np.random.default_rng()
        if mean.ndim != 1:
            mean = mean.reshape(len(mean))
        try:
            ensemble = StateVectors(
                                    [StateVector((rng.multivariate_normal(mean, covar)))
                                     for n in range(num_vectors)])
        # If covar is univariate, then use the univariate noise generation function.
        except ValueError:
            ensemble = StateVectors(
                [StateVector((rng.normal(mean, covar))) for n in range(num_vectors)])

        return ensemble

    @property
    def ndim(self):
        """Number of dimensions in state vectors"""
        return np.shape(self.state_vector)[0]

    @property
    def num_vectors(self):
        """Number of columns in state ensemble"""
        return np.shape(self.state_vector)[1]

    @property
    def mean(self):
        """The state mean, numerically equivalent to state vector"""
        return np.average(self.state_vector, axis=1)

    @property
    def covar(self):
        """Sample covariance matrix for ensemble"""
        return np.cov(self.state_vector)

    @property
    def sqrt_covar(self):
        """sqrt of sample covariance matrix for ensemble, useful for
        some EnKF algorithms"""
        return ((self.state_vector-np.tile(self.mean, self.num_vectors))
                / np.sqrt(self.num_vectors - 1))


State.register(EnsembleState)  # noqa: E305


class CategoricalState(State):
    r"""CategoricalState type.

    State object representing an object in a categorical state space. A state vector
    :math:`\mathbf{\alpha}_t^i = P(\phi_t^i)` defines a categorical distribution over a finite set
    of discrete categories :math:`\Phi = \{\phi^m|m\in \mathbf{N}, m\le M\}` for some finite
    :math:`M`."""

    categories: Sequence[float] = Property(doc="Category names. Defaults to a list of integers.",
                                           default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_vector = self.state_vector / np.sum(self.state_vector)  # normalise state vector

        if self.categories is None:
            self.categories = list(map(str, range(self.ndim)))

        if len(self.categories) != self.ndim:
            raise ValueError(
                f"ndim of {self.ndim} does not match number of categories {len(self.categories)}"
            )

    def __str__(self):
        strings = [f"P({category}) = {p}"
                   for category, p in zip(self.categories, self.state_vector)]
        string = ',\n'.join(strings)
        return string

    @property
    def category(self):
        """Return the name of the most likely category."""
        return self.categories[np.argmax(self.state_vector)]


class CompositeState(Type):
    """Composite state type.

    A composition of ordered sub-states (:class:`State`) existing at the same timestamp,
    representing an object with a state for (potentially) multiple, distinct state spaces.
    """

    sub_states: Sequence[State] = Property(
        doc="Sequence of sub-states comprising the composite state. All sub-states must have "
            "matching timestamp. Must not be empty.")
    default_timestamp: datetime.datetime = Property(
        default=None,
        doc="Default timestamp if no sub-states exist to attain timestamp from. Defaults to "
            "`None`, whereby sub-states will be required to have timestamps.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.sub_states) == 0:
            raise ValueError("Cannot create an empty composite state")

        self._check_timestamp()  # validate timestamps of sub-states

    @property
    def timestamp(self):
        return self.default_timestamp

    def _check_timestamp(self):
        """Check all timestamps are equal. Replace empty sub-state timestamps with validated
        timestamp."""

        self._timestamp = None

        sub_timestamps = {sub_state.timestamp
                          for sub_state in self.sub_states
                          if sub_state.timestamp}

        if len(sub_timestamps) > 1:
            raise ValueError("All sub-states must share the same timestamp if defined")

        if (sub_timestamps and self.default_timestamp
                and not sub_timestamps == {self.default_timestamp}):
            raise ValueError("Sub-state timestamps and default timestamp must be the same if "
                             "defined")

        if sub_timestamps:
            self.default_timestamp = sub_timestamps.pop()

        for sub_state in self.sub_states:
            sub_state.timestamp = self.default_timestamp

    @property
    def state_vectors(self):
        return [state.state_vector for state in self.sub_states]

    @property
    def state_vector(self):
        """A combination of the component states' state vectors."""
        return StateVector(np.concatenate(self.state_vectors))

    def __contains__(self, item):

        return self.sub_states.__contains__(item)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self.sub_states.__getitem__(index))
        return self.sub_states.__getitem__(index)

    def __iter__(self):
        return self.sub_states.__iter__()

    def __len__(self):
        return self.sub_states.__len__()


State.register(CompositeState)  # noqa: E305
