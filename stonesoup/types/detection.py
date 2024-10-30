from collections.abc import MutableMapping, Sequence

from .groundtruth import GroundTruthPath
from .state import CategoricalState, CompositeState
from .state import State, GaussianState, StateVector
from ..base import Property
from ..models.measurement import MeasurementModel


class Detection(State):
    """Detection type"""
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="The measurement model used to generate the detection (the default is ``None``)")

    metadata: MutableMapping = Property(
        default=None, doc='Dictionary of metadata items for Detections.')

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)
        if self.metadata is None:
            self.metadata = {}


class GaussianDetection(Detection, GaussianState):
    """GaussianDetection type"""


class Clutter(Detection):
    """Clutter type for detections classed as clutter

    This is same as :class:`~.Detection`, but can be used to identify clutter
    for metrics and analysis purposes.
    """


class TrueDetection(Detection):
    """TrueDetection type for detections that come from ground truth

    This is same as :class:`~.Detection`, but can be used to identify true
    detections for metrics and analysis purposes.
    """

    groundtruth_path: GroundTruthPath = Property(
        doc="Ground truth path that this detection came from")


class MissedDetection(Detection):
    """Detection type for a missed detection

    This is same as :class:`~.Detection`, but it is used in
    MultipleHypothesis to indicate the null hypothesis (no
    detections are associated with the specified track).
    """

    state_vector: StateVector = Property(default=None, doc="State vector. Default `None`.")

    def __init__(self, state_vector=None, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)

    def __bool__(self):
        return False


class CategoricalDetection(Detection, CategoricalState):
    """Categorical detection type."""


class TrueCategoricalDetection(TrueDetection, CategoricalDetection):
    """TrueCategoricalDetection type for categorical detections that come from ground truth."""


class CompositeDetection(CompositeState):
    """Composite detection type

    Composition of :class:`~.Detection`.
    """

    sub_states: Sequence[Detection] = Property(
        doc="Sequence of sub-detections comprising the composite detection. All sub-detections "
            "must have matching timestamp. Must not be empty.")
    groundtruth_path: GroundTruthPath = Property(
        default=None,
        doc="Ground truth path that this detection came from.")
    mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping of detections to composite state space. Defaults to `None`, where "
            "sub-detections map to sub-state spaces in order.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.mapping and len(self.mapping) != len(self.sub_states):
            raise ValueError("Mappings and sub-detections must have same count")
        elif self.mapping is None:
            self.mapping = list(range(len(self.sub_states)))

    @property
    def metadata(self):
        """Combined metadata of all sub-detections."""
        metadata = dict()
        for sub_detection in self.sub_states:
            metadata.update(sub_detection.metadata)
        return metadata


Detection.register(CompositeDetection)  # noqa: E305
