from typing import Sequence

from .base import Updater
from ..base import Property
from ..types.hypothesis import CompositeHypothesis
from ..types.update import CompositeUpdate


class CompositeUpdater(Updater):
    """Composite updater type

    A composition of sub-updaters (:class:`~.Updater`).
    """

    sub_updaters: Sequence[Updater] = Property(
        doc="Sequence of sub-updaters comprising the composite updater. Must not be empty.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if not isinstance(self.sub_updaters, Sequence):
            raise ValueError(f"Sub-updaters must be defined as an ordered list, not "
                             f"{type(self.sub_updaters)}")

        if len(self.sub_updaters) == 0:
            raise ValueError("Cannot create an empty composite updater")

        if any(not isinstance(sub_updater, Updater) for sub_updater in self.sub_updaters):
            raise ValueError("All sub-updaters must be a Updater type")

    @property
    def measurement_model(self):
        raise NotImplementedError("A composition of updaters has no defined measurement model")

    def predict_measurement(self, *args, **kwargs):
        """To attain measurement predictions, the composite updater will use it's sub-updaters'
        `predict_measurement` methods and leave combining these to the
        :class:`~.CompositeHypothesis` type."""
        raise NotImplementedError("A composite updater has no method to predict a measurement")

    def update(self, hypothesis: CompositeHypothesis, **kwargs):
        r"""Given a hypothesised association between a composite predicted state or composite
        predicted measurement and a composite measurement, calculate the composite
        posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.CompositeHypothesis`
            the prediction-measurement association hypothesis. This hypothesis may carry a
            composite predicted measurement, or a composite predicted state. In the latter case
            a measurement prediction is calculated for each sub-state of the composite hypothesis,
            which will then create its own composite measurement prediction.
        **kwargs : various
            These are passed to the :meth:`predict_measurement` method of each sub-updater

        Returns
        -------
        : :class:`~.CompositeUpdate`
            The posterior composite state update
        """

        sub_updates = []

        if not isinstance(hypothesis, CompositeHypothesis):
            raise ValueError("CompositeUpdater can only update with CompositeHypothesis types")

        if len(hypothesis) != len(self):
            raise ValueError(f"Mismatch in number of sub-hypotheses {len(hypothesis)} and number "
                             f"of sub-updaters {len(self)}")

        for sub_updater, sub_hypothesis in zip(self.sub_updaters, hypothesis.sub_hypotheses):

            sub_pred = sub_hypothesis.prediction
            sub_meas_model = sub_hypothesis.measurement.measurement_model

            if sub_hypothesis.measurement_prediction is None:
                sub_hypothesis.measurement_prediction = \
                    sub_updater.predict_measurement(sub_pred, sub_meas_model)

            if sub_hypothesis:
                sub_update = sub_updater.update(sub_hypothesis, **kwargs)
            else:
                # append predictions where no detection is available for sub-state
                sub_update = sub_hypothesis.prediction
            sub_updates.append(sub_update)

        return CompositeUpdate(sub_states=sub_updates, hypothesis=hypothesis)

    def __contains__(self, item):
        return self.sub_updaters.__contains__(item)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite updater will be
        created from the sub-list of sub-updaters."""
        if isinstance(index, slice):
            return self.__class__(self.sub_updaters.__getitem__(index))
        return self.sub_updaters.__getitem__(index)

    def __iter__(self):
        return self.sub_updaters.__iter__()

    def __len__(self):
        return self.sub_updaters.__len__()
