import copy
import datetime
import uuid
from collections import abc
from numbers import Integral
from typing import MutableSequence, Any, Optional, Sequence, MutableMapping
import typing

import numpy as np

from ..base import Property, clearable_cached_property
from .array import StateVector, CovarianceMatrix, PrecisionMatrix, StateVectors
from .base import Type
from .particle import Particle, MultiModelParticle, RaoBlackwellisedParticle
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
            if name not in args_property_names and name not in kwargs}

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

        return target_type.from_state(state, *args, **kwargs, target_type=target_type)


class ASDState(Type):
    """ASD State type

    For the use of Accumulated State Densities.
    """

    multi_state_vector: StateVector = Property(
        doc="State vector of all timestamps")
    timestamps: Sequence[datetime.datetime] = Property(
        doc="List of all timestamps which have a state in the ASDState")
    max_nstep: int = Property(
        doc="Decides when the state is pruned in a prediction step. If 0 then there is no pruning")

    def __init__(self, multi_state_vector, timestamps,
                 max_nstep=0, *args, **kwargs):
        if multi_state_vector is not None and timestamps is not None:
            multi_state_vector = StateVector(multi_state_vector)
            if not isinstance(timestamps, Sequence):
                timestamps = list([timestamps])
            self.max_nstep = max_nstep
        super().__init__(multi_state_vector, timestamps, max_nstep, *args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, Integral):
            ndim = self.ndim
            start = item * ndim
            end = None if item == -1 else (item+1) * ndim
            state_slice = slice(start, end)
            state_vector = StateVector(self.multi_state_vector[state_slice])
            timestamp = self.timestamps[item]
            return State(state_vector=state_vector, timestamp=timestamp)
        else:
            raise TypeError(f'{type(self).__name__!r} only subscriptable by int')

    @property
    def state_vector(self):
        """The State vector of the newest timestamp"""
        return self.multi_state_vector[0:self.ndim]

    @property
    def timestamp(self):
        """The newest timestamp"""
        return self.timestamps[0]

    @property
    def ndim(self):
        """Dimension of one State"""
        return int(self.multi_state_vector.shape[0] / len(self.timestamps))

    @property
    def nstep(self):
        """Number of timesteps which are in the ASDState"""
        return len(self.timestamps)

    @clearable_cached_property('multi_state_vector', 'timestamps')
    def state(self):
        """A :class:`~.State` object representing latest timestamp"""
        return self[0]

    @clearable_cached_property('multi_state_vector', 'timestamps')
    def states(self):
        return [self[i] for i in range(self.nstep)]


State.register(ASDState)


class StateMutableSequence(Type, abc.MutableSequence):
    """A mutable sequence for :class:`~.State` instances

    This sequence acts like a regular list object for States, as well as
    proxying state attributes to the last state in the sequence. This sequence
    can also be indexed/sliced by :class:`datetime.datetime` instances.

    Notes
    -----
    If shallow copying, similar to a list, it is safe to add/remove states
    without affecting the original sequence.

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

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        property_name = self.__class__.states._property_name
        inst.__dict__[property_name] = copy.copy(self.__dict__[property_name])
        return inst

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

    @clearable_cached_property('sqrt_covar')
    def covar(self):
        """The full covariance matrix."""
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

    @clearable_cached_property('state_vector', 'precision')
    def gaussian_state(self):
        """The Gaussian state."""

        return GaussianState(self.mean,
                             self.covar,
                             self.timestamp)

    @clearable_cached_property('precision')
    def covar(self):
        """Covariance matrix, inverse of :attr:`precision` matrix."""
        return np.linalg.inv(self.precision)

    @clearable_cached_property('state_vector', 'precision')
    def mean(self):
        """Equivalent Gaussian mean"""
        return self.covar @ self.state_vector

    @classmethod
    def from_gaussian_state(cls, gaussian_state, *args, **kwargs):
        r"""
        Returns an InformationState instance based on the gaussian_state.

        Parameters
        ----------
        gaussian_state : :class:`~.GaussianState`
            The guassian_state used to create the new WeightedGaussianState.
        \*args : See main :class:`~.InformationState`
            args are passed to :class:`~.InformationState` __init__()
        \*\*kwargs : See main :class:`~.InformationState`
            kwargs are passed to :class:`~.InformationState` __init__()

        Returns
        -------
        :class:`~.InformationState`
            Instance of InformationState.
        """
        precision = np.linalg.inv(gaussian_state.covar)
        state_vector = precision @ gaussian_state.state_vector
        timestamp = gaussian_state.timestamp

        return cls(
            state_vector=state_vector,
            precision=precision,
            timestamp=timestamp,
            *args, **kwargs
        )


class ASDGaussianState(ASDState):
    """ASDGaussian State type

    This is a simple Accumulated State Density Gaussian state object, which as
    the name suggests is described by a Gaussian state distribution.
    """
    multi_covar: CovarianceMatrix = Property(doc="Covariance of all timesteps")
    correlation_matrices: MutableSequence[MutableMapping[str, np.ndarray]] = Property(
        default=None,
        doc="Sequence of Correlation Matrices, consisting of :math:`P_{l|l}`, :math:`P_{l|l+1}` "
            "and :math:`F_{l+1|l}` built in the Kalman predictor and Kalman updater, aligned to "
            ":attr:`timestamps`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.correlation_matrices is None:
            self.correlation_matrices = []

    def __getitem__(self, item):
        if isinstance(item, Integral):
            ndim = self.ndim
            start = item * ndim
            end = None if item == -1 else (item+1) * ndim
            state_slice = slice(start, end)
            state_vector = StateVector(self.multi_state_vector[state_slice])
            covar = CovarianceMatrix(self.multi_covar[state_slice, state_slice])
            timestamp = self.timestamps[item]
            return GaussianState(state_vector=state_vector, covar=covar, timestamp=timestamp)
        else:
            raise TypeError(f'{type(self).__name__!r} only subscriptable by int')

    @property
    def covar(self):
        return self.multi_covar[:self.ndim, :self.ndim]

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector

    @clearable_cached_property('multi_state_vector', 'multi_covar', 'timestamps')
    def state(self):
        """A :class:`~.GaussianState` object representing latest timestamp"""
        return super().state

    @clearable_cached_property('multi_state_vector', 'multi_covar', 'timestamps')
    def states(self):
        return super().states


class WeightedGaussianState(GaussianState):
    """Weighted Gaussian State Type

    Gaussian State object with an associated weight.  Used as components
    for a GaussianMixtureState.
    """
    weight: Probability = Property(default=0, doc="Weight of the Gaussian State.")

    @clearable_cached_property('state_vector', 'covar')
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


class ASDWeightedGaussianState(ASDGaussianState):
    """ASD Weighted Gaussian State Type

    ASD Gaussian State object with an associated weight.  Used as components
    for a GaussianMixtureState.
    """
    weight: Probability = Property(default=0, doc="Weight of the Gaussian State.")


class ParticleState(State):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles
    """

    state_vector: StateVectors = Property(doc='State vectors.')
    weight: MutableSequence[Probability] = Property(default=None, doc='Weights of particles')
    log_weight: np.ndarray = Property(default=None, doc='Log weights of particles')
    parent: 'ParticleState' = Property(default=None, doc='Parent particles')
    particle_list: MutableSequence[Particle] = Property(default=None,
                                                        doc='List of Particle objects')
    fixed_covar: CovarianceMatrix = Property(default=None,
                                             doc='Fixed covariance value. Default `None`, where'
                                                 'weighted sample covariance is then used.')

    def __init__(self, *args, **kwargs):
        weight = next(
            (val for name, val in zip(type(self).properties, args) if name == 'weight'),
            kwargs.get('weight', None))
        log_weight, idx = next(
            ((val, idx) for idx, (name, val) in enumerate(zip(type(self).properties, args))
             if name == 'log_weight'),
            (kwargs.get('log_weight', None), None))

        if weight is not None and log_weight is not None:
            raise ValueError("Cannot provide both weight and log weight")
        elif log_weight is None and weight is not None:
            log_weight = np.log(np.asfarray(weight))
            if idx is not None:
                args[idx] = log_weight
            else:
                kwargs['log_weight'] = log_weight
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

    def __getitem__(self, item):
        if self.parent is not None:
            parent = self.parent[item]
        else:
            parent = None

        if self.log_weight is not None:
            log_weight = self.log_weight[item]
        else:
            log_weight = None

        if isinstance(item, int):
            result = Particle(state_vector=self.state_vector[:, item],
                              weight=self.weight[item] if self.weight is not None else None,
                              parent=parent)
        else:
            # Allow for Prediction/Update sub-types
            result = type(self).from_state(self,
                                           state_vector=self.state_vector[:, item],
                                           log_weight=log_weight,
                                           parent=parent)
        return result

    @classmethod
    def from_state(cls, state: 'State', *args: Any, target_type: Optional[typing.Type] = None,
                   **kwargs: Any) -> 'State':

        # Handle default presence of both particle_list and weight once class has been created by
        # ignoring particle_list and weight (setting to None) if not provided.
        particle_list, particle_list_idx = next(
            ((val, idx) for idx, (name, val) in enumerate(zip(cls.properties, args))
             if name == 'particle_list'),
            (kwargs.get('particle_list', None), None))
        if particle_list_idx is None:
            kwargs['particle_list'] = particle_list

        weight, weight_idx = next(
            ((val, idx) for idx, (name, val) in enumerate(zip(cls.properties, args))
             if name == 'weight'),
            (kwargs.get('weight', None), None))
        if weight_idx is None:
            kwargs['weight'] = weight

        return super().from_state(state, *args, target_type=target_type, **kwargs)

    @clearable_cached_property('state_vector', 'log_weight')
    def particles(self):
        """Sequence of individual :class:`~.Particle` objects."""
        if self.particle_list is not None:
            return self.particle_list
        return tuple(particle for particle in self)

    def __len__(self):
        return self.state_vector.shape[1]

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]

    @clearable_cached_property('state_vector', 'log_weight')
    def mean(self):
        """Sample mean for particles"""
        if len(self) == 1:  # No need to calculate mean
            return self.state_vector
        return np.average(self.state_vector, axis=1, weights=np.exp(self.log_weight))

    @clearable_cached_property('state_vector', 'log_weight', 'fixed_covar')
    def covar(self):
        """Sample covariance matrix for particles"""
        if self.fixed_covar is not None:
            return self.fixed_covar
        return np.cov(self.state_vector, ddof=0, aweights=np.exp(self.log_weight))

    @weight.setter
    def weight(self, value):
        if value is None:
            self.log_weight = None
        else:
            self.log_weight = np.log(np.asfarray(value))
            self.__dict__['weight'] = np.asanyarray(value)

    @weight.getter
    def weight(self):
        try:
            return self.__dict__['weight']
        except KeyError:
            log_weight = self.log_weight
            if log_weight is None:
                return None
            weight = Probability.from_log_ufunc(log_weight)
            self.__dict__['weight'] = weight
            return weight

State.register(ParticleState)  # noqa: E305
ParticleState.log_weight._clear_cached.add('weight')


class MultiModelParticleState(ParticleState):
    """Multi-Model Particle State type

    This is a particle state object which describes the state as a
    distribution of particles, where each particle has an associated
    dynamics model
    """

    dynamic_model: np.ndarray = Property(
        default=None,
        doc="Array of indices that identify which model is associated with each particle.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.particle_list and isinstance(self.particle_list, list):
            self.dynamic_model = \
                np.array([particle.dynamic_model for particle in self.particle_list])

    def __getitem__(self, item):
        if self.parent is not None:
            parent = self.parent[item]
        else:
            parent = None

        if self.log_weight is not None:
            log_weight = self.log_weight[item]
        else:
            log_weight = None

        if self.dynamic_model is not None:
            dynamic_model = self.dynamic_model[item]
        else:
            dynamic_model = None

        if isinstance(item, int):
            result = MultiModelParticle(
                state_vector=self.state_vector[:, item],
                weight=self.weight[item] if self.weight is not None else None,
                parent=parent,
                dynamic_model=dynamic_model)
        else:
            # Allow for Prediction/Update sub-types
            result = type(self).from_state(self,
                                           state_vector=self.state_vector[:, item],
                                           log_weight=log_weight,
                                           parent=parent,
                                           dynamic_model=dynamic_model)
        return result


class RaoBlackwellisedParticleState(ParticleState):

    model_probabilities: np.ndarray = Property(
        default=None,
        doc="2d NumPy array containing probability of particle belong to particular model. "
            "Shape (n-models, m-particles)."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.particle_list and isinstance(self.particle_list, list):
            self.model_probabilities = \
                np.column_stack([particle.model_probabilities for particle in self.particle_list])

    def __getitem__(self, item):
        if self.parent is not None:
            parent = self.parent[item]
        else:
            parent = None

        if self.log_weight is not None:
            log_weight = self.log_weight[item]
        else:
            log_weight = None

        if self.model_probabilities is not None:
            model_probabilities = self.model_probabilities[:, item]
        else:
            model_probabilities = None

        if isinstance(item, int):
            result = RaoBlackwellisedParticle(
                state_vector=self.state_vector[:, item],
                weight=self.weight[item] if self.weight is not None else None,
                parent=parent,
                model_probabilities=model_probabilities)
        else:
            # Allow for Prediction/Update sub-types
            result = type(self).from_state(self,
                                           state_vector=self.state_vector[:, item],
                                           log_weight=log_weight,
                                           parent=parent,
                                           model_probabilities=model_probabilities)
        return result


class EnsembleState(State):
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
    def from_gaussian_state(cls, gaussian_state, num_vectors):
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

        return EnsembleState(state_vector=cls.generate_ensemble(mean, covar, num_vectors),
                             timestamp=timestamp)

    @staticmethod
    def generate_ensemble(mean, covar, num_vectors):
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
    def num_vectors(self):
        """Number of columns in state ensemble"""
        return np.shape(self.state_vector)[1]

    @clearable_cached_property('state_vector')
    def mean(self):
        """The state mean, numerically equivalent to state vector"""
        return np.average(self.state_vector, axis=1)

    @clearable_cached_property('state_vector')
    def covar(self):
        """Sample covariance matrix for ensemble"""
        return np.cov(self.state_vector)

    @clearable_cached_property('state_vector')
    def sqrt_covar(self):
        """sqrt of sample covariance matrix for ensemble, useful for
        some EnKF algorithms"""
        return ((self.state_vector-np.tile(self.mean, self.num_vectors))
                / np.sqrt(self.num_vectors - 1))


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
