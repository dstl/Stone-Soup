import copy
import datetime
import weakref

from ..base import Property
from ..hypothesiser import Hypothesiser
from ..initiator.simple import SimpleMeasurementInitiator, SinglePointInitiator
from ..predictor import Predictor
from ..predictor.oos import get_hypothesis, get_past_states, get_prior_state
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import Prediction
from ..types.update import Update
from . import Updater


class OOSUpdaterWrapper(Updater):
    """
    OOSUpdaterWrapper is a wrapper for an Updater that enables out-of-sequence (OOS) measurement
    updates. It manages the process of updating state estimates when measurements arrive out of
    chronological order, by reprocessing the affected states and propagating updates forward.

    To enable this, the chain of past states is required. As :class:`~.Prediction` only holds a
    weak reference to states further back in this chain (to limit memory use), this updater holds
    strong references to the prior of each state it creates, for at least :attr:`min_history`.
    """
    measurement_model = None
    updater: Updater = Property(doc="Updater being wrapped to carry out update stage")
    predictor: Predictor = Property(
        doc="Predictor used to predict when reprocessing updates with out of sequence "
            "measurements")
    initiator: SinglePointInitiator | SimpleMeasurementInitiator = Property(
        default=None,
        doc="Initiator to use for creating initial state if hypothesis prior to oldest past "
            "state. Default ``None``, where an error will be raised.")
    hypothesiser: Hypothesiser = Property(
        default=None,
        doc="Hypothesiser to regenerate hypotheses for out of sequence measurements e.g. for "
            "PDA where wish to recalculate association probabilities for each measurement. "
            "Default ``None``, where existing hypotheses are used.")
    min_history: datetime.timedelta = Property(
        default=None,
        doc="Min history to ensure is kept for out of sequence, beyond which references to "
            "prior states are released, enabling garbage collection of old states. Default "
            "``None``, where history is kept indefinitely.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Strong references from each state to its prior state
        self._priors = weakref.WeakKeyDictionary()

    @staticmethod
    def _update_states(latest_state, prior):
        states = []
        for state in get_past_states(latest_state):
            if state is prior:
                break
            states.append(state)
        return states

    def _update_hypotheses(self, hypothesis, prediction, repredict=False):
        hypothesis = copy.copy(hypothesis)
        if isinstance(hypothesis, MultipleHypothesis):
            hypothesis.single_hypotheses = [
                self._update_hypotheses(hyp, prediction, repredict) for hyp in hypothesis]
            return hypothesis
        if repredict:
            hypothesis.prediction = self._repredict(
                prediction, hypothesis.prediction.timestamp)
        else:
            hypothesis.prediction = prediction
        hypothesis.measurement_prediction = None
        return hypothesis

    @classmethod
    def _get_detections(cls, hypothesis):
        if not hypothesis:
            return set()
        elif isinstance(hypothesis, MultipleHypothesis):
            return {hyp.measurement for hyp in hypothesis if hyp}
        else:
            return {hypothesis.measurement}

    def _initiate_state(self, hypothesis, **kwargs):
        return self.initiator.initiate(
                    {hypothesis.measurement}, hypothesis.prediction.timestamp, **kwargs).pop()[-1]

    def _hold_prior(self, state):
        """Hold strong reference to prior of state, so that it is not garbage collected"""
        prior = get_prior_state(state)
        if prior is not None:
            self._priors[state] = prior

    @staticmethod
    def _release_prior(state):
        """Replace reference to prior of state with weakref, so it can be garbage collected"""
        if isinstance(state, Prediction):
            predictions = [state]
        elif isinstance(state, Update):
            if isinstance(state.hypothesis, MultipleHypothesis):
                predictions = [hyp.prediction for hyp in state.hypothesis]
            else:
                predictions = [state.hypothesis.prediction]
        else:
            predictions = []
        for prediction in predictions:
            if isinstance(prediction, Prediction) and prediction.prior is not None:
                prediction.prior = weakref.ref(prediction.prior)

    def _prune_history(self, state):
        """Release references to states older than :attr:`min_history`"""
        cutoff = state.timestamp - self.min_history
        # Find the most recent state at or before the cutoff, which must be kept to cover
        # the full min history
        for state in get_past_states(state):
            if state.timestamp <= cutoff:
                break
        else:
            return
        while state is not None:
            self._release_prior(state)
            state = self._priors.pop(state, None)

    def _repredict(self, prediction, timestamp, **kwargs):
        return self.predictor.predict(prediction.prior, timestamp, **kwargs)

    def predict_measurement(
            self, predicted_state, measurement_model=None, measurement_noise=True, **kwargs):
        return self.updater.predict_measurement(
            predicted_state, measurement_model, measurement_noise, **kwargs)

    def update(self, hypothesis, **kwargs):
        prediction = get_hypothesis(hypothesis, lambda x: not x).prediction
        latest_state = prediction.prior.orig_prior
        prior = prediction.prior

        if prior.timestamp > prediction.timestamp:
            hyp = get_hypothesis(hypothesis, bool)
            if not hyp:
                # No new information
                post = prior = latest_state
            elif self.initiator:
                # Ideally shouldn't use one hypothesis here
                post = self._initiate_state(hyp, **kwargs)
                prior = None
            else:  # not self.initiator
                raise RuntimeError("Hypothesis prior to earliest state and no initiator")
        else:
            if latest_state is not prior:
                # Could have been smoothed: predict again
                hypothesis = self._update_hypotheses(hypothesis, prediction, repredict=True)
            if not hypothesis:
                post = hypothesis.prediction
            else:
                post = self.updater.update(hypothesis, **kwargs)
            self._hold_prior(post)

        superseded_states = self._update_states(latest_state, prior)
        for state in reversed(superseded_states):
            pred = self.predictor.predict(post, state.timestamp, **kwargs)
            if isinstance(state, Prediction):
                post = pred
            elif isinstance(state, Update):
                if self.hypothesiser:
                    hyp = self.hypothesiser.hypothesise(
                        post, self._get_detections(state.hypothesis), pred.timestamp, **kwargs)
                else:
                    hyp = self._update_hypotheses(state.hypothesis, pred)
                post = self.updater.update(hyp, **kwargs)
            else:
                raise TypeError(f"Unexpected state type: {type(state)!r}")
            self._hold_prior(post)

        # Superseded states have been replaced by reprocessed states, so no longer need history
        for state in superseded_states:
            self._priors.pop(state, None)

        if self.min_history:
            self._prune_history(post)

        return post
