import copy
import warnings

from . import Updater
from ..base import Property
from ..measures import Measure, Euclidean, KLDivergence
from ..predictor import Predictor

from .kalman import ExtendedKalmanUpdater
from ..smoother import Smoother
from ..types.track import Track
from ..types.update import Update


class DynamicallyIteratedUpdater(ExtendedKalmanUpdater):

    predictor: Predictor = Property(doc="Predictor to use for iterating over the predict step. "
                                        "Probably should be the same predictor used for the "
                                        "initial predict step")
    updater: Updater = Property(doc="Updater to use for iterating over update step")
    smoother: Smoother = Property(doc="Smoother used to smooth the prior ")
    tolerance: float = Property(
        default=1e-6,
        doc="The value of the difference in the measure used as a stopping criterion.")
    measure: Measure = Property(
        default=Euclidean(),
        doc="The measure to use to test the iteration stopping criterion. Defaults to the "
            "Euclidean distance between current and prior posterior state estimate.")
    max_iterations: int = Property(
        default=1000,
        doc="Number of iterations before while loop is exited and a non-convergence warning is "
            "returned")

    def update(self, hypothesis, **kwargs):

        # 1) Compute X^0_{k|k-1}, P^0_{k|k-1} via eqtn 2 (predict step)
        # Step 1 is completed by predict step, provided in hypothesis.prediction

        # 2) Compute X^0_{k|k}, P^0_{k|k} via eqtn 3 (update step)
        posterior_state = self.updater.update(hypothesis, **kwargs)

        # 3) Compute X^0_{k-1|k}, P^0_{k-1|k} via eqtn 4 (smooth)
        track_to_smooth = Track(states=[hypothesis.prediction.prior, posterior_state])
        # Feed posterior and prior update into the smoother
        smoothed_track = self.smoother.smooth(track_to_smooth)
        # Extract smoothed prior state
        prior_state = smoothed_track[0]

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
                                                timestamp=hypothesis.prediction.timestamp)
            nhypothesis.prediction = pred_state

            # (2) Update using hypothesis made from new prediction and old measurement
            nhypothesis.measurement_prediction = None
            new_posterior = self.updater.update(nhypothesis)

            # (3) Smooth again and update the prior state
            track_to_smooth = Track(states=[nhypothesis.prediction.prior, new_posterior])
            smoothed_track = self.smoother.smooth(track_to_smooth)
            prior_state = smoothed_track[0]

            # (4) Update iteration count
            iterations += 1

        return new_posterior

