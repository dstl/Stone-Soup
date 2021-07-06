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
from ...types.state import CompositeState
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
    measurement2 = measurements.pop()

    # Create a measurement with reversed mapping
    sub_states = list(np.flip(measurement1.sub_states))
    mapping = list(np.flip(measurement1.mapping))
    reversed_measurement = CompositeDetection(sub_states, mapping=mapping)
    measurements.add(reversed_measurement)

    # Create a measurement without one sub-state element
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
        create_state(True, False, ndim_states[0], timestamp),
        create_state(True, False, ndim_states[1], timestamp),
        create_state(True, False, ndim_states[2], timestamp),
        create_state(True, False, ndim_states[3], timestamp),  # Particle initiator, Gaussian prior
        create_state(False, False, ndim_states[4], timestamp)]  # Categorical prior

    sub_initiators = [
        SinglePointInitiator(sub_priors[0]),
        SinglePointInitiator(sub_priors[1]),
        SinglePointInitiator(sub_priors[2]),
        GaussianParticleInitiator(SinglePointInitiator(sub_priors[3])),
        SimpleCategoricalInitiator(sub_priors[4])
    ]

    measurements = composite_measurements(num_sub_states, ndim_states, timestamp)

    prior = CompositeState(sub_priors[:num_sub_states])
    priorless_initiator = CompositeUpdateInitiator(sub_initiators[:num_sub_states])
    prior_initiator = CompositeUpdateInitiator(sub_initiators[:num_sub_states], prior_state=prior)

    return priorless_initiator, prior_initiator, measurements


@pytest.mark.parametrize('num_initiators', [1, 2, 3, 4, 5])
def test_simple_composite(num_initiators):
    now = datetime.now()

    priorless_initiator, prior_initiator, measurements = \
        initiators_measurement(num_initiators, now)

    # Test initiate
    for initiator in (priorless_initiator, prior_initiator):
        tracks = initiator.initiate(measurements, now)

        assert len(tracks) == len(measurements)
        for track in tracks:
            assert len(track) == 1
            assert isinstance(track.state, CompositeUpdate)
            assert isinstance(track.state.hypothesis, CompositeHypothesis)
            if len(track.hypothesis.measurement) != num_initiators:
                # no update where no detection is given
                assert any({not isinstance(sub_state, Update) for sub_state in track[-1]})

    # Test prior error
    with pytest.raises(ValueError,
                       match="If no default prior state is defined, all sub-initiators require a "
                             "defined `prior_state`"):
        sub_initiators = priorless_initiator.sub_initiators
        sub_initiators[0].prior_state = None
        CompositeUpdateInitiator(sub_initiators)
