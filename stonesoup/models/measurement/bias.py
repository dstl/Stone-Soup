import copy
import datetime
from abc import abstractmethod

from . import MeasurementModel
from ..transition import TransitionModel
from ...base import Property
from ...functions import jacobian as compute_jac
from ...types.state import State, StateVectors


class BiasModelWrapper(MeasurementModel):
    """Abstract wrapper that removes bias values from an existing MeasurementModel.
    """
    measurement_model: MeasurementModel = Property(
        doc="Unbiased model being wrapped that bias will be applied to")
    state_mapping: list[int] = Property(
        doc="Mapping to state vector elements relevant to wrapped model")
    bias_mapping: list[int] = Property(doc="Mapping to state vector elements where bias is")

    @property
    def mapping(self):
        return list(self.measurement_model.mapping) + list(self.bias_mapping)

    @property
    def ndim_meas(self):
        return self.measurement_model.ndim_meas

    @abstractmethod
    def function(self, state, noise=False, **kwargs):
        raise NotImplementedError()

    def covar(self, *args, **kwargs):
        return self.measurement_model.covar(*args, **kwargs)

    def jacobian(self, state, **kwargs):
        return compute_jac(self.function, state, **kwargs)

    def pdf(self, *args, **kwargs):
        raise NotImplementedError()

    def rvs(self, *args, **kwargs):
        raise NotImplementedError()


class TimeBiasModelWrapper(BiasModelWrapper):
    """Removes a time-offset bias from the state to the wrapped measurement model.

    The bias elements selected by `bias_mapping` are interpreted as a time offset in seconds.
    For each state vector the wrapper computes the state at the (biased) measurement time by
    applying the `transition_model` with a negative time interval equal to the bias. The
    resulting state is then passed to the wrapped `measurement_model` to produce the
    (time-corrected) state vector.
    """
    transition_model: TransitionModel = Property(
        doc="Transition model applied to state to apply time offset")
    bias_mapping: list[int] = Property(
        default=(-1, ), doc="Mapping to state vector elements where bias is")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.bias_mapping, int):
            self.bias_mapping = (self.bias_mapping, )

    def function(self, state, noise=False, **kwargs):
        predicted_state_vectors = []
        for state_vector in state.state_vector.view(StateVectors):
            delta_t = state_vector[self.bias_mapping[0], 0]
            predicted_state_vectors.append(self.transition_model.function(
                State(state_vector[self.state_mapping, :]),
                time_interval=datetime.timedelta(seconds=-delta_t),
                **kwargs))
        return self.measurement_model.function(
            State(StateVectors(predicted_state_vectors)), noise=noise, **kwargs)


class OrientationBiasModelWrapper(BiasModelWrapper):
    """Removes an orientation (rotation) bias from the state to the wrapped measurement model.

    The wrapper expects `bias_mapping` to select orientation elements (e.g. Euler angles or
    equivalent) stored in the state vector. For each input state the wrapper creates a copy
    of the wrapped measurement model, adjusts its `rotation_offset` by subtracting the
    bias value, and then evaluates the measurement function using the corrected model.

    This allows the wrapped model to remain stateless while the wrapper applies per-state
    orientation corrections.
    """
    bias_mapping: list[int] = Property(
        default=(-3, -2, -1), doc="Mapping to state vector elements where bias is")

    def function(self, state, noise=False, **kwargs):
        state_vectors = []
        for state_vector in state.state_vector.view(StateVectors):
            delta_orient = state_vector[self.bias_mapping, :]
            bias_model = copy.copy(self.measurement_model)
            bias_model.rotation_offset = bias_model.rotation_offset - delta_orient
            state_vectors.append(bias_model.function(
                State(state_vector[self.state_mapping, :]), noise=noise, **kwargs))
        if len(state_vectors) == 1:
            return state_vectors[0]
        else:
            return StateVectors(state_vectors)


class TranslationBiasModelWrapper(BiasModelWrapper):
    """Removes a translation (position) bias from the state to the wrapped measurement model.

    The wrapper expects `bias_mapping` to select translation elements stored in the state
    vector. For each input state the wrapper creates a copy of the wrapped measurement model,
    adjusts its `translation_offset` by subtracting the bias value, and then evaluates the
    measurement function using the corrected model.
    """
    bias_mapping: list[int] = Property(
        default=(-3, -2, -1), doc="Mapping to state vector elements where bias is")

    def function(self, state, noise=False, **kwargs):
        state_vectors = []
        for state_vector in state.state_vector.view(StateVectors):
            delta_trans = state_vector[self.bias_mapping, :]
            bias_model = copy.copy(self.measurement_model)
            bias_model.translation_offset = bias_model.translation_offset - delta_trans
            state_vectors.append(bias_model.function(
                State(state_vector[self.state_mapping, :]), noise=noise, **kwargs))
        if len(state_vectors) == 1:
            return state_vectors[0]
        else:
            return StateVectors(state_vectors)


class OrientationTranslationBiasModelWrapper(BiasModelWrapper):
    """Removes combined orientation and translation biases from the state to the wrapped model.

    `bias_mapping` is expected to contain orientation indices first (3 elements) followed by
    translation indices (2 or 3 elements). For each input state the wrapper copies the
    wrapped measurement model, subtracts the orientation and translation bias values from
    the model's `rotation_offset` and `translation_offset` respectively, and evaluates the
    corrected model on the mapped portion of the state.
    """
    bias_mapping: list[int] = Property(
        default=(-6, -5, -4, -3, -2, -1), doc="Mapping to state vector elements where bias is")

    def function(self, state, noise=False, **kwargs):
        state_vectors = []
        for state_vector in state.state_vector.view(StateVectors):
            delta_orient = state_vector[self.bias_mapping[:3], :]
            delta_trans = state_vector[self.bias_mapping[3:], :]
            bias_model = copy.copy(self.measurement_model)
            bias_model.rotation_offset = bias_model.rotation_offset - delta_orient
            bias_model.translation_offset = bias_model.translation_offset - delta_trans
            state_vectors.append(bias_model.function(
                State(state_vector[self.state_mapping, :]), noise=noise, **kwargs))
        if len(state_vectors) == 1:
            return state_vectors[0]
        else:
            return StateVectors(state_vectors)
