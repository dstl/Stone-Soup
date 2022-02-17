# -*- coding: utf-8 -*-
from datetime import datetime

import numpy as np
import pytest

from ..categorical import SimpleCategoricalInitiator
from ..composite import CompositeUpdateInitiator
from ..simple import SinglePointInitiator, GaussianParticleInitiator
from ...predictor.tests.test_composite import create_state
from ...types.detection import CompositeDetection, Detection, CategoricalDetection
from ...types.hypothesis import CompositeHypothesis
from ...types.update import CompositeUpdate, Update
from ...updater.tests.test_composite import create_measurement_model


def composite_measurements(num_sub_states, ndim_states, timestamp):
    """Create a random number of composite detections, each with `num_sub_states`
    sub-detections.
    """
    measurement_models = [
        create_measurement_model(True, ndim_states[0]),
        create_measurement_model(True, ndim_states[1]),
        create_measurement_model(True, ndim_states[2]),
        create_measurement_model(True, ndim_states[3]),
        create_measurement_model(False, ndim_states[4])
    ]

    measurements = set()

    for _ in range(np.random.randint(3, 6)):
        sub_measurements = [
            Detection(
                state_vector=create_state(True, False, ndim_states[0], timestamp).state_vector,
                timestamp=timestamp,
                measurement_model=measurement_models[0]),
            Detection(
                state_vector=create_state(True, False, ndim_states[1], timestamp).state_vector,
                timestamp=timestamp,
                measurement_model=measurement_models[1]),
            Detection(
                state_vector=create_state(True, False, ndim_states[2], timestamp).state_vector,
                timestamp=timestamp,
                measurement_model=measurement_models[2]),
            Detection(
                state_vector=create_state(True, True, ndim_states[3], timestamp).state_vector,
                timestamp=timestamp,
                measurement_model=measurement_models[3]),
            CategoricalDetection(
                state_vector=create_state(False, False, ndim_states[4], timestamp).state_vector,
                timestamp=timestamp,
                measurement_model=measurement_models[4])
        ]
        measurements.add(CompositeDetection(sub_measurements[:num_sub_states]))

    measurement1 = measurements.pop()

    # Create a measurement with reversed mapping
    sub_states = list(np.flip(measurement1.sub_states))
    mapping = list(np.flip(measurement1.mapping))
    reversed_measurement = CompositeDetection(sub_states, mapping=mapping)
    measurements.add(reversed_measurement)

    if num_sub_states > 1:
        # Create a measurement without one sub-state element
        measurement2 = measurements.pop()
        missed_index = np.random.randint(num_sub_states)
        sub_states = measurement2.sub_states[:missed_index] + measurement2.sub_states[
                                                              missed_index + 1:]
        mapping = measurement2.mapping[:missed_index] + measurement2.mapping[missed_index + 1:]
        incomplete_measurement = CompositeDetection(sub_states, mapping=mapping)
        measurements.add(incomplete_measurement)

    return measurements


def initiators_measurement(num_sub_states, timestamp):
    """ Create a random :class:`~.CompositeUpdateInitiator`, corresponding prior and set of
    appropriate measurements.
    """
    ndim_states = np.random.randint(2, 5, 5)

    sub_priors = [
        create_state(True, False, ndim_states[0], None),
        create_state(True, False, ndim_states[1], None),
        create_state(True, False, ndim_states[2], None),
        create_state(True, False, ndim_states[3], None),  # Particle initiator, Gaussian prior
        create_state(False, False, ndim_states[4], None)]  # Categorical prior

    sub_initiators = [
        SinglePointInitiator(sub_priors[0]),
        SinglePointInitiator(sub_priors[1]),
        SinglePointInitiator(sub_priors[2]),
        GaussianParticleInitiator(SinglePointInitiator(sub_priors[3])),
        SimpleCategoricalInitiator(sub_priors[4])
    ]

    measurements = composite_measurements(num_sub_states, ndim_states, timestamp)

    initiator = CompositeUpdateInitiator(sub_initiators[:num_sub_states])

    return initiator, measurements


@pytest.mark.parametrize('num_initiators', [1, 2, 3, 4, 5])
def test_simple_composite(num_initiators):
    now = datetime.now()

    initiator, measurements = initiators_measurement(num_initiators, now)

    # Test instantiation errors
    with pytest.raises(ValueError, match="Cannot create an empty composite initiator"):
        CompositeUpdateInitiator(list())

    with pytest.raises(ValueError, match="All sub-initiators must be an initiator type"):
        CompositeUpdateInitiator(initiator.sub_initiators + [1, 2, 3])

    # Test prior
    for actual_sub_state, sub_initiator in zip(initiator.prior_state, initiator.sub_initiators):
        assert actual_sub_state == sub_initiator.prior_state

    # Test initiate
    tracks = initiator.initiate(measurements, now)

    assert len(tracks) == len(measurements)
    for track in tracks:
        assert len(track) == 1
        assert isinstance(track.state, CompositeUpdate)
        assert isinstance(track.state.hypothesis, CompositeHypothesis)
        if len(track.hypothesis.measurement) != num_initiators:
            # no update where no detection is given
            assert any({not isinstance(sub_state, Update) for sub_state in track[-1]})

    # Test contains
    for sub_initiator in initiator.sub_initiators:
        assert sub_initiator in initiator
    assert SinglePointInitiator(create_state(True, False, 5, None)) not in initiator
    assert 'a' not in initiator

    # Test get
    for i, expected_hypothesiser in enumerate(initiator.sub_initiators):
        assert initiator[i] == expected_hypothesiser

    # Test get slice
    if num_initiators > 1:
        initiator_slice = initiator[1:]
        assert isinstance(initiator_slice, CompositeUpdateInitiator)
        assert initiator_slice.sub_initiators == initiator.sub_initiators[1:]

    # Test iter
    for i, exp_sub_hypothesiser in enumerate(initiator):
        assert exp_sub_hypothesiser == initiator.sub_initiators[i]

    # Test len
    assert len(initiator) == num_initiators
