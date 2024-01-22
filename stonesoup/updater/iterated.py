import copy
import warnings

from . import Updater
from ..base import Property
from ..measures import Measure, KLDivergence
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..predictor import Predictor
from .kalman import ExtendedKalmanUpdater
from ..predictor.kalman import ExtendedKalmanPredictor
from ..smoother import Smoother
from ..smoother.kalman import ExtendedKalmanSmoother
from ..types.prediction import Prediction
from ..types.track import Track


class DynamicallyIteratedUpdater(Updater):
    """
    Wrapper around a :class:`~.Predictor`, :class:`~.Updater` and :class:`~.Smoother`. This
    updater takes a :class:`~.Prediction`, and updates as usual by calling its updater property.
    The updated state is then used to smooth the prior state, completing the first iteration.
    The second iteration begins from predicting using the smoothed prior. Iterates until
    convergence, or a maximum number of iterations is reached.

    Implementation of algorithm 2: Dynamically iterated filter, from "Iterated Filters for
    Nonlinear Transition Models"

    References
    ----------

    1. Anton Kullberg, Isaac Skog, Gustaf Hendeby,
    "Iterated Filters for Nonlinear Transition Models"
    """
    measurement_model = None
    predictor: Predictor = Property(doc="Predictor to use for iterating over the predict step. "
                                        "Probably should be the same predictor used for the "
                                        "initial predict step")
    updater: Updater = Property(doc="Updater to use for iterating over update step")
    smoother: Smoother = Property(doc="Smoother used to smooth the prior ")
    tolerance: float = Property(
        default=1e-6,
        doc="The value of the difference in the measure used as a stopping criterion.")
    measure: Measure = Property(
        default=KLDivergence(),
        doc="The measure to use to test the iteration stopping criterion. Defaults to the "
            "Euclidean distance between current and prior posterior state estimate.")
    max_iterations: int = Property(
        default=1000,
        doc="Number of iterations before while loop is exited and a non-convergence warning is "
            "returned")

    def predict_measurement(self, *args, **kwargs):
        return self.updater.predict_measurement(*args, **kwargs)

    def update(self, hypothesis, **kwargs):

        # Get last update step for smoothing
        prior_state = hypothesis.prediction.prior
        while isinstance(prior_state, Prediction) and getattr(prior_state, 'prior', None):
            prior_state = prior_state.prior

        # 1) Compute X^0_{k|k-1}, P^0_{k|k-1} via eqtn 2 (predict step)
        # Step 1 is completed by predict step, provided in hypothesis.prediction

        # 2) Compute X^0_{k|k}, P^0_{k|k} via eqtn 3 (update step)
        posterior_state = self.updater.update(hypothesis, **kwargs)

        # 3) Compute X^0_{k-1|k}, P^0_{k-1|k} via eqtn 4 (smooth)
        track_to_smooth = Track(states=[prior_state, posterior_state])
        # Feed posterior and prior update into the smoother
        smoothed_track = self.smoother.smooth(track_to_smooth)
        # Extract smoothed prior state
        smoothed_state = smoothed_track[0]

        nhypothesis = copy.deepcopy(hypothesis)
        iterations = 0

        old_posterior = None
        new_posterior = posterior_state

        # While not converged, loop through predict, update, smooth steps:
        while iterations == 0 or self.measure(old_posterior, new_posterior) > self.tolerance:

            # Break out of loop if iteration limit is reached before convergence
            if iterations > self.max_iterations:
                warnings.warn("Iterated Kalman update did not converge")
                break

            # Update time: New posterior from previous iteration becomes old posterior
            old_posterior = new_posterior

            # The rest of the loop is equivalent to the following steps:
            # 4) Calculate (A_f, b_f, Ω_f) via linearisation of f about X^i_{k-1|k}, P^i_{k-1|k}
            # 5) Compute X^{i+1}_{k|k-1}, P^{i+1}_{k|k-1} via eqtn 2
            # 6) Calculate (A_h, b_h, Ω_h) via linearisation of h about X^i_{k|k}, P^i_{k|k}
            # 7) Compute X^{i+1}_{k|k-1}, P^{i+1}_{k|k-1} via eqtn 3
            # 8) Compute X^{i+1}_{k-1|k}, P^{i+1}_{k-1|k} via eqtn 4

            # (1) Predict from prior_state
            pred_state = self.predictor.predict(prior_state,
                                                timestamp=hypothesis.prediction.timestamp,
                                                linearisation_point=smoothed_state)
            nhypothesis.prediction = pred_state

            # (2) Update using hypothesis made from new prediction and old measurement
            nhypothesis.measurement_prediction = None
            new_posterior = self.updater.update(nhypothesis, linearisation_point=old_posterior)

            # (3) Smooth again and update the smoothed state
            track_to_smooth = Track(states=[prior_state, new_posterior])
            smoothed_track = self.smoother.smooth(
                track_to_smooth, linearisation_point=smoothed_state)
            smoothed_state = smoothed_track[0]

            # (4) Update iteration count
            iterations += 1

        # Reassign original hypothesis
        new_posterior.hypothesis = hypothesis

        return new_posterior


class DynamicallyIteratedEKFUpdater(DynamicallyIteratedUpdater):
    predictor = None
    updater = None
    smoother = None
    measurement_model: MeasurementModel = Property(doc="measurement model")
    transition_model: TransitionModel = Property(doc="The transition model to be used.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updater = ExtendedKalmanUpdater(self.measurement_model)
        self.predictor = ExtendedKalmanPredictor(self.transition_model)
        self.smoother = ExtendedKalmanSmoother(self.transition_model)


ExtendedKalmanUpdater.register(DynamicallyIteratedEKFUpdater)
