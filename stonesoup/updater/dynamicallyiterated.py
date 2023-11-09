import copy

from . import Updater
from ..base import Property
from ..measures import Measure, Euclidean
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

        # Get the measurement model out of the measurement if it's there.
        # If not, use the one native to the updater (which might still be
        # none)
        measurement_model = hypothesis.measurement.measurement_model
        measurement_model = self._check_measurement_model(measurement_model)

        # # If there is no measurement prediction in the hypothesis then do the
        # # measurement prediction (and attach it back to the hypothesis).
        # if hypothesis.measurement_prediction is None:
        #     # Attach the measurement prediction to the hypothesis
        #     hypothesis.measurement_prediction = self.predict_measurement(
        #         hypothesis.prediction.state_vector,
        #         measurement_model=measurement_model, **kwargs)

        # 1) Compute X^0_{k|k-1}, P^0_{k|k-1} via eqtn 2 (predict)
        pred_state = hypothesis.prediction.state_vector
        pred_covar = hypothesis.prediction.covar

        # 2) Compute X^0_{k|k}, P^0_{k|k} via eqtn 3 (measurement update)
        # measurement_state = hypothesis.measurement_prediction
        # measurement_covar = measurement_model.covar()

        # The first iteration is just the application of the EKF
        posterior_state = self.updater.update(hypothesis, **kwargs)

        # 3) Compute X^0_{k-1|k}, P^0_{k-1|k} via eqtn 4 (smooth)
        # Take track, make a copy, and cut off everything before the second last state
        # of type == update
        track_to_smooth = Track(states=[hypothesis.prediction.prior, posterior_state])

        print(track_to_smooth)

        smoothed_track = self.smoother.smooth(track_to_smooth)

        prior_state = smoothed_track[0]

        #prev_state = posterior_state

        nhypothesis = copy.deepcopy(hypothesis)
        iterations = 0

        old_posterior = None
        new_posterior = posterior_state

        # While not converged, loop through predict, update, smooth steps:
        while iterations == 0 or self.measure(old_posterior, new_posterior) > self.tolerance:

            # Update time: New posterior from previous iteration becomes old posterior
            old_posterior = new_posterior

            # # Clear measurement prediction and re-predict
            # nhypothesis.measurement_prediction = None
            # nhypothesis = self._check_measurement_prediction(nhypothesis)


            # The rest of the loop is equivalent to the following steps:
            # 4) Calculate (A_f, b_f, Ω_f) via linearisation of f about X^i_{k-1|k}, P^i_{k-1|k}
            # 5) Compute X^{i+1}_{k|k-1}, P^{i+1}_{k|k-1} via eqtn 2
            # 6) Calculate (A_h, b_h, Ω_h) via linearisation of h about X^i_{k|k}, P^i_{k|k}
            # 7) Compute X^{i+1}_{k|k-1}, P^{i+1}_{k|k-1} via eqtn 3
            # 8) Compute X^{i+1}_{k-1|k}, P^{i+1}_{k-1|k} via eqtn 4

            # Predict from prior_state
            pred_state = self.predictor.predict(prior_state)
            nhypothesis.prediction = pred_state

            # Update using hypothesis made from new prediction and old measurement
            new_posterior = self.updater.update(nhypothesis)

            # Smooth again
            track_to_smooth = Track(states=[nhypothesis.prediction.prior, new_posterior])
            smoothed_track = self.smoother.smooth(track_to_smooth)

            prior_state = smoothed_track[0]


            iterations += 1

        # Kalman gain and posterior covariance
        posterior_covariance, kalman_gain = self._posterior_covariance(nhypothesis)

        posterior_mean = self._posterior_mean(nhypothesis.prediction, kalman_gain,
                                              nhypothesis.measurement,
                                              nhypothesis.measurement_prediction)

        return Update.from_state(nhypothesis.prediction,
                                 posterior_mean,
                                 posterior_covariance,
                                 timestamp=hypothesis.measurement.timestamp,
                                 hypothesis=hypothesis)

