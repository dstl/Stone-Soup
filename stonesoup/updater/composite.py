# -*- coding: utf-8 -*-
from typing import Sequence

from .base import Updater
from ..base import Property
from ..types.hypothesis import CompositeHypothesis
from ..types.prediction import CompositePrediction
from ..types.update import CompositeUpdate


class CompositeUpdater(Updater):
    """A composition of multiple sub-updaters.

    Updates a :class:`~.CompositeState` composed of a sequence of predictions using a
    :class:`~.CompositeDetection` composed of a sequence of measurements using a sequence of
    sub-updaters.
    """
    sub_updaters: Sequence[Updater] = Property(doc="A sequence of sub-updaters")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.sub_updaters, Sequence):
            raise ValueError("sub-updaters must be defined as an ordered list")

        if any(not isinstance(sub_updater, Updater) for sub_updater in self.sub_updaters):
            raise ValueError("all sub-updaters must be an Updater type")

    @property
    def measurement_model(self):
        raise NotImplementedError("A composition of updaters have no defined measurement model")

    def predict_measurement(self, *args, **kwargs):
        """To attain measurement predictions, the composite updater will use it's sub-updaters'
        `predict_measurement` methods and leave combining these to the CompositeHypothesis type."""
        raise NotImplementedError("A composite updater has no method to predict a measurement")

    def update(self, hypothesis: CompositeHypothesis, **kwargs):
        r"""Given a hypothesised association between a composite predicted state or composite
        predicted measurement and an actual composite measurement, calculate the composite
        posterior state.

        Parameters
        ----------
        hypothesis : :class:`~.CompositeHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a composite predicted measurement, or a composite predicted state. In the
            latter case a composite predicted measurement will be calculated.
        **kwargs : various
            These are passed to :meth:`predict_measurement`

        Returns
        -------
        : :class:`~.CompositeUpdate`
            The posterior composite state update
        """

        sub_updates = []
        is_prediction = True

        if not isinstance(hypothesis, CompositeHypothesis):
            raise ValueError("CompositeUpdater can only be used with CompositeHypothesis types")
        if len(hypothesis.sub_hypotheses) != len(self.sub_updaters):
            raise ValueError("CompositeHypothesis must be composed of same number of"
                             " sub-hypotheses as sub-updaters")

        for sub_updater, sub_hypothesis in zip(self.sub_updaters, hypothesis.sub_hypotheses):

            sub_pred = sub_hypothesis.prediction
            sub_meas_model = sub_hypothesis.measurement.measurement_model

            if sub_hypothesis.measurement_prediction is None:
                sub_hypothesis.measurement_prediction = \
                    sub_updater.predict_measurement(sub_pred, sub_meas_model)

            # This step is usually handled by tracker type
            if sub_hypothesis:
                sub_update = sub_updater.update(sub_hypothesis, **kwargs)
                # If at least one sub-state is updated, consider the track updated
                is_prediction = False
            else:
                sub_update = sub_hypothesis.prediction
            sub_updates.append(sub_update)

        if is_prediction:
            return CompositePrediction(sub_states=sub_updates)

        return CompositeUpdate(sub_states=sub_updates, hypothesis=hypothesis)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite updater will be
        created from the sub-list of sub-updaters."""
        if isinstance(index, slice):
            return self.__class__(self.sub_updaters.__getitem__(index))
        return self.sub_updaters.__getitem__(index)

    def __iter__(self):
        return iter(self.sub_updaters)

    def __len__(self):
        return self.sub_updaters.__len__()

    def __contains__(self, item):
        return self.sub_updaters.__contains__(item)

    def append(self, value):
        """Add value at end of :attr:`updaters`.

        Parameters
        ----------
        value: :class:`~.Updater`
            An Updater to be added at the end of :attr:`sub_updaters`.
        """
        self.sub_updaters.append(value)
