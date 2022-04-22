# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest
import scipy.linalg

from ..angle import Bearing
from ..array import StateVector, StateVectors, CovarianceMatrix
from ..groundtruth import GroundTruthState
from ..numeric import Probability
from ..particle import Particle
from ..state import CreatableFromState
from ..state import State, GaussianState, ParticleState, EnsembleState, \
    StateMutableSequence, WeightedGaussianState, SqrtGaussianState, CategoricalState, \
    CompositeState

from ...base import Property


def test_state():
    with pytest.raises(TypeError):
        State()

    # Test state initiation without timestamp
    state_vector = StateVector([[0], [1]])
    state = State(state_vector)
    assert np.array_equal(state.state_vector, state_vector)

    # Test state initiation with timestamp
    timestamp = datetime.datetime.now()
    state = State(state_vector, timestamp=timestamp)
    assert state.timestamp == timestamp


def test_state_invalid_vector():
    with pytest.raises(ValueError):
        State(StateVector([[[1, 2, 3, 4]]]))


def test_gaussianstate():
    """ GaussianState Type test """

    with pytest.raises(TypeError):
        GaussianState()

    mean = StateVector([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = CovarianceMatrix([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    timestamp = datetime.datetime.now()

    # Test state initiation without timestamp
    state = GaussianState(mean, covar)
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])
    assert(state.timestamp is None)

    # Test state initiation with timestamp
    state = GaussianState(mean, covar, timestamp)
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])
    assert(state.timestamp == timestamp)


def test_gaussianstate_invalid_covar():
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3]))  # 3D
    with pytest.raises(ValueError):
        GaussianState(mean, covar)


def test_sqrtgaussianstate():
    """Test the square root Gaussian Type"""

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0.1, 0.03, 0.01],
                      [0.1, 2.2130, 0.03, 0.02],
                      [0.03, 0.03, 2.123, 0.01],
                      [0.01, 0.02, 0.01, 2.012]]) * 1e3
    timestamp = datetime.datetime.now()

    # Test that a lower triangular matrix returned when 'full' covar is passed
    lower_covar = np.linalg.cholesky(covar)
    state = SqrtGaussianState(mean, lower_covar, timestamp=timestamp)
    assert np.array_equal(state.sqrt_covar, lower_covar)
    assert np.allclose(state.covar, covar, 0, atol=1e-10)
    assert np.allclose(state.sqrt_covar @ state.sqrt_covar.T, covar, 0, atol=1e-10)
    assert np.allclose(state.sqrt_covar @ state.sqrt_covar.T, lower_covar @ lower_covar.T, 0,
                       atol=1e-10)

    # Test that a general square root matrix is also a solution
    general_covar = scipy.linalg.sqrtm(covar)
    another_state = SqrtGaussianState(mean, general_covar, timestamp=timestamp)
    assert np.array_equal(another_state.sqrt_covar, general_covar)
    assert np.allclose(state.covar, covar, 0, atol=1e-10)
    assert not np.allclose(another_state.sqrt_covar, lower_covar, 0, atol=1e-10)


def test_weighted_gaussian_state():
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3]))  # 3D
    weight = 0.3
    timestamp = datetime.datetime.now()
    with pytest.raises(ValueError):
        WeightedGaussianState(mean, covar, timestamp, weight)

    # Test initialization using a GuassianState
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3, 4]))
    weight = 0.3
    gs = GaussianState(mean, covar, timestamp=timestamp)
    wgs = WeightedGaussianState.from_gaussian_state(gaussian_state=gs, weight=weight)
    assert np.array_equal(gs.state_vector, wgs.state_vector)
    assert np.array_equal(gs.covar, wgs.covar)
    assert gs.timestamp == wgs.timestamp
    assert weight == wgs.weight
    assert wgs.state_vector is not gs.state_vector
    assert wgs.covar is not gs.covar

    # Test copy flag
    wgs = WeightedGaussianState.from_gaussian_state(gaussian_state=gs, copy=False)
    assert wgs.state_vector is gs.state_vector
    assert wgs.covar is gs.covar

    # Test gaussian_state property
    gs2 = wgs.gaussian_state
    assert np.array_equal(gs.state_vector, gs2.state_vector)
    assert np.array_equal(gs.covar, gs2.covar)
    assert gs.timestamp == gs2.timestamp


def test_particlestate():
    with pytest.raises(TypeError):
        ParticleState()

    # 1D
    num_particles = 10
    state_vector1 = StateVector([[0.]])
    state_vector2 = StateVector([[100.]])
    weight = Probability(1/num_particles)
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[50]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500]]))

    # Test state with timestamp
    timestamp = datetime.datetime.now()
    state = ParticleState(particles, timestamp=timestamp)
    assert np.allclose(state.state_vector, StateVector([[50]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500]]))
    assert state.timestamp == timestamp

    # 2D
    state_vector1 = StateVector([[0.], [0.]])
    state_vector2 = StateVector([[100.], [200.]])
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    state = ParticleState(particles)
    assert isinstance(state, State)
    assert ParticleState in State.subclasses
    assert np.allclose(state.state_vector, StateVector([[50], [100]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500, 5000], [5000, 10000]]))


def test_particlestate_weighted():
    num_particles = 10

    # Half particles at high weight at 0
    state_vector1 = StateVector([[0.]])
    weight1 = Probability(0.75 / (num_particles / 2))

    # Other half of particles low weight at 100
    state_vector2 = StateVector([[100]])
    weight2 = Probability(0.25 / (num_particles / 2))

    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight1) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight2) for _ in range(num_particles//2))

    # Check particles sum to 1 still
    assert pytest.approx(1) == sum(particle.weight for particle in particles)

    # Test state vector is now weighted towards 0 from 50 (non-weighted mean)
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[25]]))
    assert np.allclose(state.covar, CovarianceMatrix([[1875]]))


def test_particlestate_angle():
    num_particles = 10

    state_vector1 = StateVector([[Bearing(np.pi + 0.1)], [-10.]])
    state_vector2 = StateVector([[Bearing(np.pi - 0.1)], [20.]])
    weight = Probability(1/num_particles)
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[np.pi], [5.]]))
    assert np.allclose(state.covar, CovarianceMatrix([[0.01, -1.5], [-1.5, 225]]))

    
    def test_ensemblestate():

    # 1D
    state_vector1 = StateVector(np.array([1.5]))
    state_vector2 = StateVector(np.array([0.5]))
    list_of_state_vectors = [state_vector1, state_vector2]
    ensemble = StateVectors(list_of_state_vectors)
    
    # Test state without timestamp
    state = EnsembleState(ensemble)
    assert np.allclose(state.state_vector, StateVector([[1]]))
    assert np.allclose(state.covar, CovarianceMatrix([[0.5]]))

    # Test state with timestamp
    timestamp = datetime.datetime(2021, 2, 25, 22, 29, 2)
    state = EnsembleState(ensemble, timestamp=timestamp)
    assert np.allclose(state.state_vector, StateVector([[1]]))
    assert np.allclose(state.covar, CovarianceMatrix([[0.5]]))
    assert state.timestamp == timestamp

    # 2D
    state_vector1 = StateVector(np.array([1.5,0.75]))
    state_vector2 = StateVector(np.array([0.5,1.25]))
    ensemble = StateVectors([state_vector1, state_vector2])

    state = EnsembleState(ensemble)
    assert np.allclose(state.state_vector, StateVector([[1], [1]]))
    assert np.allclose(state.covar, CovarianceMatrix([[0.5, -0.25], [-0.25, 0.125]]))
    assert np.allclose(state.sqrt_covar @ state.sqrt_covar.T, state.covar)
    
def test_ensemblestate_gaussian_init():
    """Test initialising with an existing gaussian state object"""
    
    #Initialize GaussianState
    mean = StateVector([[25], [25], [25], [25]])
    covar = CovarianceMatrix(np.eye(4))
    timestamp = datetime.datetime(2021, 2, 26, 16, 35, 42)
    gaussian_state = GaussianState(mean,covar,timestamp)
    #Generate EnsembleState
    num_vectors = 500
    ensemble_state = EnsembleState.from_gaussian_state(gaussian_state, num_vectors)

    
    assert isinstance(ensemble_state.state_vector,StateVector)
    assert isinstance(ensemble_state.ensemble,StateVectors)
    assert isinstance(ensemble_state.covar,CovarianceMatrix)
    assert isinstance(ensemble_state.timestamp,datetime.datetime)
    assert ensemble_state.timestamp == timestamp
    

def test_state_mutable_sequence_state():
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        [State(state_vector, timestamp=timestamp+delta*n)
         for n in range(10)])

    assert sequence.state is sequence.states[-1]
    assert np.array_equal(sequence.state_vector, state_vector)
    assert sequence.timestamp == timestamp + delta*9

    del sequence[-1]
    assert sequence.timestamp == timestamp + delta*8


def test_state_mutable_sequence_slice():
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        [State(state_vector, timestamp=timestamp+delta*n)
         for n in range(10)])

    assert isinstance(sequence[timestamp:], StateMutableSequence)
    assert isinstance(sequence[5:], StateMutableSequence)
    assert isinstance(sequence[timestamp], State)
    assert isinstance(sequence[5], State)

    assert len(sequence[timestamp:]) == 10
    assert len(sequence[:timestamp]) == 0
    assert len(sequence[timestamp+delta*5:]) == 5
    assert len(sequence[:timestamp+delta*5]) == 5
    assert len(sequence[timestamp+delta*4:timestamp+delta*6]) == 2
    assert len(sequence[timestamp+delta*2:timestamp+delta*8:3]) == 2
    assert len(sequence[timestamp+delta*1:][:timestamp+delta*2]) == 1

    assert sequence[timestamp] == sequence.states[0]

    end_timestamp = sequence.timestamp
    assert sequence[end_timestamp] == sequence.states[-1]
    assert sequence[end_timestamp].state_vector == StateVector([[0]])

    # Add state at same time
    sequence.append(State(state_vector + 1, timestamp=end_timestamp))
    assert sequence[end_timestamp]
    assert sequence[end_timestamp].state_vector == StateVector([[1]])

    assert len(sequence) == 11
    assert len(list(sequence.last_timestamp_generator())) == 10

    with pytest.raises(TypeError):
        sequence[timestamp:1]

    with pytest.raises(IndexError):
        sequence[timestamp-delta]


def test_state_mutable_sequence_sequence_init():
    """Test initialising with an existing sequence"""
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        StateMutableSequence([State(state_vector, timestamp=timestamp + delta * n)
                              for n in range(10)]))

    assert not isinstance(sequence.states, list)

    assert sequence.state is sequence.states[-1]
    assert np.array_equal(sequence.state_vector, state_vector)
    assert sequence.timestamp == timestamp + delta * 9

    del sequence[-1]
    assert sequence.timestamp == timestamp + delta * 8


def test_state_mutable_sequence_error_message():
    """Test that __getattr__ doesn't incorrectly identify the source of a missing attribute"""

    class TestSMS(StateMutableSequence):
        test_property: int = Property(default=3)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_variable = 5

        def test_method(self):
            pass

        @property
        def complicated_attribute(self):
            if self.test_property == 3:
                return self.test_property
            else:
                raise AttributeError('Custom error message')

    timestamp = datetime.datetime.now()
    test_obj = TestSMS(states=State(state_vector=StateVector([1, 2, 3]), timestamp=timestamp))

    # First check no errors on assigned vars
    test_obj.test_method()
    assert test_obj.test_property == 3
    test_obj.test_property = 6
    assert test_obj.test_property == 6
    assert test_obj.test_variable == 5

    # Now check that state variables are proxied correctly
    assert np.array_equal(test_obj.state_vector, StateVector([1, 2, 3]))
    assert test_obj.timestamp == timestamp

    # Now check that the right error messages are raised on missing attributes
    with pytest.raises(AttributeError, match="'TestSMS' object has no attribute 'missing_method'"):
        test_obj.missing_method()

    with pytest.raises(AttributeError, match="'TestSMS' object has no attribute "
                                             "'missing_variable'"):
        _ = test_obj.missing_variable

    # And check custom error messages are not swallowed
    # in the default case (test_property == 3), complicated_attribute works
    test_obj.test_property = 3
    assert test_obj.complicated_attribute == 3

    # when test_property != 3  it raises a custom error.
    test_obj.test_property = 5
    with pytest.raises(AttributeError, match="Custom error message"):
        _ = test_obj.complicated_attribute


def test_from_state():
    start = datetime.datetime.now()
    kwargs = {"state_vector": np.arange(4), "timestamp": start}

    states = [
        State(**kwargs),
        GaussianState(**kwargs, covar=np.eye(4)),
        GroundTruthState(**kwargs, metadata={"colour": "blue"})
    ]

    for use_sequence in (False, True):
        for state in states:

            original_type = type(state)
            if use_sequence:
                state = StateMutableSequence(states=[state])
            # test replacement arg
            new_state = State.from_state(state, np.ones(4))
            assert isinstance(new_state, original_type)
            assert np.array_equal(new_state.state_vector.flatten(), np.ones(4))
            assert new_state.timestamp == start
            if original_type is GaussianState:
                assert np.array_equal(new_state.covar, state.covar)
            elif original_type is GroundTruthState:
                assert new_state.metadata == state.metadata

            # test replacement kwarg
            new_time = start + datetime.timedelta(seconds=5)
            new_state = State.from_state(state, timestamp=new_time)
            assert isinstance(new_state, original_type)
            assert np.array_equal(new_state.state_vector, state.state_vector)
            assert new_state.timestamp == new_time
            if original_type is GaussianState:
                assert np.array_equal(new_state.covar, state.covar)
            elif original_type is GroundTruthState:
                assert new_state.metadata == state.metadata

            # test replacement arg and kwarg
            new_time = start + datetime.timedelta(seconds=5)
            new_state = State.from_state(state, np.ones(4), timestamp=new_time)
            assert isinstance(new_state, original_type)
            assert np.array_equal(new_state.state_vector.flatten(), np.ones(4))
            assert new_state.timestamp == new_time
            if original_type is GaussianState:
                assert np.array_equal(new_state.covar, state.covar)
            elif original_type is GroundTruthState:
                assert new_state.metadata == state.metadata

    # test covar overwrite
    new_state = State.from_state(states[1], covar=2 * np.eye(4))
    assert isinstance(new_state, type(states[1]))
    assert np.array_equal(new_state.state_vector, states[1].state_vector)
    assert new_state.timestamp == states[1].timestamp
    assert np.array_equal(new_state.covar, 2 * np.eye(4))

    # test metadata overwrite
    new_metadata = {"size": "big"}
    new_state = State.from_state(states[2], metadata=new_metadata)
    assert isinstance(new_state, type(states[2]))
    assert np.array_equal(new_state.state_vector, states[2].state_vector)
    assert new_state.timestamp == states[2].timestamp
    assert new_state.metadata == new_metadata


# noinspection PyUnusedLocal
def test_creatable_from_state_error():
    class SubclassCfs(CreatableFromState):
        pass
    with pytest.raises(TypeError,
                       match='The first superclass of a CreatableFromState subclass must be a '
                             'CreatableFromState \\(or a subclass\\)'):
        class SubSubclassCfs(State, SubclassCfs):
            pass


# noinspection PyUnusedLocal
def test_creatable_from_state_multi_base_error():
    class SubclassCfs(CreatableFromState):
        pass
    with pytest.raises(TypeError,
                       match='A CreatableFromState subclass must have exactly two superclasses'):
        class SubSubclassCfs(State, StateMutableSequence, SubclassCfs):
            pass


def test_categorical_state():

    # Test mismatched number of category names
    with pytest.raises(ValueError, match="ndim of 3 does not match number of categories 4"):
        CategoricalState(state_vector=StateVector([50, 60, 90]),
                         categories=['red', 'green', 'blue', 'yellow'])

    state = CategoricalState(state_vector=StateVector([50, 60, 90]))

    # Test normalised
    state.state_vector == [0.25, 0.3, 0.45]

    # Test default category names
    assert state.categories == ['0', '1', '2']

    # Test string
    assert str(state) == "P(0) = 0.25,\nP(1) = 0.3,\nP(2) = 0.45"

    # Test category
    assert state.category == '2'

    
def test_composite_state_timestamp():
    with pytest.raises(ValueError,
                       match="All sub-states must share the same timestamp if defined"):
        CompositeState([State([0], timestamp=1), State([0], timestamp=2)])
    with pytest.raises(ValueError,
                       match="Sub-state timestamps and default timestamp must be the same if "
                             "defined"):
        CompositeState([State([0], timestamp=1)], default_timestamp=2)
    with pytest.raises(ValueError,
                       match="Sub-state timestamps and default timestamp must be the same if "
                             "defined"):
        CompositeState([State([0], timestamp=1), State([0], timestamp=1)], default_timestamp=2)

    for i in range(1, 4):
        assert CompositeState(i * [State([0], timestamp=1)]).timestamp == 1
        assert CompositeState(i * [State([0], timestamp=1)],
                              default_timestamp=1).timestamp == 1
        assert CompositeState(i * [State([0])]).timestamp is None


def test_composite_state():
    # Test error on empty composite
    with pytest.raises(ValueError, match="Cannot create an empty composite state"):
        CompositeState([])

    a = State([0, 1], timestamp=1)
    b = State([2], timestamp=1)
    c = State([3, 4], timestamp=1)
    sub_states = [a, b, c]
    state = CompositeState(sub_states)

    # Test state vectors
    for actual, expected in zip(state.state_vectors,
                                [StateVector([0, 1]), StateVector([2]), StateVector([3, 4])]):
        assert (actual == expected).all()

    # Test state vector
    assert (state.state_vector == StateVector([0, 1, 2, 3, 4])).all()

    # Test contains and getitem
    for index, sub_state in enumerate(sub_states):
        assert sub_state in state
        assert state[index] is sub_state
    assert State([5, 6], timestamp=1) not in state
    assert "a" not in state
    state_slice = state[1:]
    assert isinstance(state_slice, CompositeState)
    assert state_slice.sub_states == sub_states[1:]

    # Test iter
    for exp_sub_state, actual_sub_state in zip(sub_states, state):
        assert exp_sub_state is actual_sub_state

    # Test len
    assert len(state) == 3

