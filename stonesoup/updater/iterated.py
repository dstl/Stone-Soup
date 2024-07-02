import copy
import warnings
from typing import Callable

from . import Updater
from ..base import Property
from ..measures import Measure, KLDivergence
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..predictor import Predictor
from .kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from ..predictor.kalman import ExtendedKalmanPredictor
from ..smoother import Smoother
from ..smoother.kalman import ExtendedKalmanSmoother
from ..types.prediction import Prediction, AugmentedGaussianMeasurementPrediction
from ..types.track import Track
from ..functions import slr_definition
from ..models.measurement.linear import GeneralLinearGaussian
from functools import partial


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


class IPLFKalmanUpdater(UnscentedKalmanUpdater):
    """
    This implements the updater of the Iterated Posterior Linearization FIlter
    (IPLF) described in [1]. It is obtained by iteratively linearising the
    measurement function using statistical linear regression (SLR) with respect
    to the posterior (rather than the prior), to take into account the information
    provided by the measurement. Nevertheless, on the first iteration linearisation
    is performed with respect to the prior, so the output is equivalent to that of
    a standard :class:`~.UnscentedKalmanUpdater`. This class inherits most of the
    functionality from :class:`~.UnscentedKalmanUpdater`.

    References
    ----------
    [1] Á. F. García-Fernández, L. Svensson, M. R. Morelande and S. Särkkä,
    "Posterior Linearization Filter: Principles and Implementation Using Sigma Points,"
    in IEEE Transactions on Signal Processing, vol. 63, no. 20, pp. 5561-5573,
    Oct.15, 2015, doi: 10.1109/TSP.2015.2454485.
    """

    tolerance: float = Property(
        default=1e-1,
        doc="The value used as a convergence criterion.")
    measure: Measure = Property(
        default=KLDivergence(),
        doc="The measure to use to test the convergence. Defaults to the "
            "GaussianKullbackLeiblerDivergence between previous and current posteriors.")
    max_iterations: int = Property(
        default=5,
        doc="The maximum number of iterations. "
            "Setting `max_iterations=1` is equivalent to using UKF.")
    slr_func: Callable = Property(default=slr_definition,
                                  doc="Function to compute "
                                      "the SLR parameters.")

    def update(self, hypothesis, keep_linearisation=True,
               force_symmetric_covariance=True, **kwargs):

        # Get the nonlinear measurement model and its parameters
        measurement_model = self._check_measurement_model(hypothesis.measurement.measurement_model)
        meas_func = partial(self.predict_measurement,
                            measurement_model=measurement_model,
                            measurement_noise=False)
        ndim_state = measurement_model.ndim_state
        r_cov_matrix = measurement_model.covar()

        # Initial approximation for the posterior
        post_state = hypothesis.prediction

        for i in range(self.max_iterations):

            # Preserve the previous approximation for convergence evaluation
            prev_post_state = post_state

            # Compute the parameters of Statistical Linear Regression (SLR)
            h_matrix, b_vector, omega_cov_matrix \
                = self.slr_func(post_state, meas_func,
                                force_symmetric_covariance=force_symmetric_covariance)
            slr_parameters = {
                'h_matrix': h_matrix,
                'b_vector': b_vector,
                'omega_cov_matrix': omega_cov_matrix
            }

            # Create a linear measurement model using the SLR parameters
            measurement_model_linear = GeneralLinearGaussian(
                ndim_state=ndim_state,
                meas_matrix=slr_parameters['h_matrix'],
                bias_value=slr_parameters['b_vector'],
                noise_covar=slr_parameters['omega_cov_matrix']+r_cov_matrix)

            # Predict the measurement using the linearised model
            measurement_prediction_linear = KalmanUpdater.predict_measurement(
                self,
                predicted_state=hypothesis.prediction,
                measurement_model=measurement_model_linear
            )

            if keep_linearisation:

                # Store the SLR parameters
                metadata = {
                    'slr_parameters': slr_parameters,
                    'r_cov_matrix': r_cov_matrix,
                    'iteration': i,
                    'max_iterations': self.max_iterations,
                    'tolerance': self.tolerance
                }

                # Wrap the prediction into a custom class
                # that preserves the linearised model and metadata
                measurement_prediction_linear = AugmentedGaussianMeasurementPrediction(
                    state_vector=measurement_prediction_linear.state_vector,
                    covar=measurement_prediction_linear.covar,
                    timestamp=measurement_prediction_linear.timestamp,
                    cross_covar=measurement_prediction_linear.cross_covar,
                    measurement_model=measurement_model_linear,
                    metadata=metadata
                )

            # Perform a linear update using the predicted measurement
            hypothesis.measurement_prediction = measurement_prediction_linear
            post_state = KalmanUpdater.update(self, hypothesis, **kwargs)

            if force_symmetric_covariance:
                post_state.covar = (post_state.covar + post_state.covar.T) / 2

            # KLD between the previous and current posteriors to measure convergence
            if self.measure(prev_post_state, post_state) < self.tolerance:
                break

        return post_state
