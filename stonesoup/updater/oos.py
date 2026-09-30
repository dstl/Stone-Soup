import copy
import datetime
import weakref

from ..base import Property
from ..hypothesiser import Hypothesiser
from ..initiator.simple import SimpleMeasurementInitiator, SinglePointInitiator
from ..predictor import Predictor
from ..predictor.oos import get_hypothesis, get_past_states
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import Prediction
from ..types.update import Update
from . import Updater


class OOSUpdaterWrapper(Updater):
    """
    OOSUpdaterWrapper is a wrapper for an Updater that enables out-of-sequence (OOS) measurement
    updates. It manages the process of updating state estimates when measurements arrive out of
    chronological order, by reprocessing the affected states and propagating updates forward.
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
        doc="Min history to ensure is used for out of sequence, where weakref to prior beyond "
            "this time will be set to enable garbage collection of old states. Default ``None``, "
            "where history is kept indefinitely.")

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
        else:
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

        for state in reversed(self._update_states(latest_state, prior)):
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

        if self.min_history:
            for state in get_past_states(post):
                if isinstance(state, Prediction) and state.prior \
                        and state.prior.timestamp < post.timestamp - self.min_history:
                    state.prior = weakref.ref(state.prior)
                    break

        return post
