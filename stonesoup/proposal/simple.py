from typing import Union

import numpy as np
from enum import Enum
from scipy.stats import multivariate_normal as mvn

from stonesoup.base import Property
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.state import State, GaussianState, SqrtGaussianState
from stonesoup.types.prediction import Prediction
from stonesoup.updater.base import Updater
from stonesoup.predictor.base import Predictor
from stonesoup.predictor.kalman import SqrtKalmanPredictor
from stonesoup.types.hypothesis import SingleHypothesis


class PriorAsProposal(Proposal):
    """Proposal that uses the dynamics model as the importance density.
    This proposal uses the dynamics model to predict the next state, and then
    uses the predicted state as the prior for the measurement model.
    """
    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")

    def rvs(self, prior: State, measurement=None, time_interval=None,
            **kwargs) -> Union[StateVector, StateVectors]:
        """Generate samples from the proposal.
        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.
        Returns
        -------
        : :class:`~.ParticlePrediction` with samples drawn from the updated proposal

        """

        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - prior.timestamp
        else:
            timestamp = prior.timestamp + time_interval

        new_state_vector = self.transition_model.function(prior,
                                                          time_interval=time_interval,
                                                          **kwargs)
        return Prediction.from_state(prior,
                                     parent=prior,
                                     state_vector=new_state_vector,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model,
                                     prior=prior)



class KFasProposal(Proposal):
    """This proposal uses the kalman filter prediction and update steps to
    generate new set of particles and weights
    """
    predictor: Predictor = Property(
        doc="predictor to use the various values")
    updater: Updater = Property(
        doc="Updater used for update the values")

    def rvs(self, prior: State, measurement: Detection = None, time_interval=None,
            **kwargs):
        """Generate samples from the proposal.
            Use the kalman filter predictor and updater to create a new distribution
        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.
        measurement: :class:`~.Detection`
            the measurement that is used in the update step of the kalman prediction,
            (the default is `None`)
        time_interval: :class:`datetime.time_delta`
            time interval of the prediction is needed to propagate the states

        Returns
        -------
        : :class:`~.ParticlePrediction`
        """

        # get the number of particles
        number_particles = prior.state_vector.shape[1]

        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - prior.timestamp
        else:
            timestamp = prior.timestamp + time_interval

        if time_interval.total_seconds() == 0:
            return Prediction.from_state(prior,
                                         parent=prior,
                                         state_vector=prior.state_vector,
                                         timestamp=prior.timestamp,
                                         transition_model=self.predictor.transition_model,
                                         prior=prior)

        prior_cls = GaussianState  # Default
        if isinstance(self.predictor, SqrtKalmanPredictor):
            prior_cls = SqrtGaussianState

        # Null covariance
        null_covar = np.zeros_like(prior.covar)

        predictions = [
            self.predictor.predict(
                prior_cls(particle_sv, null_covar, prior.timestamp),
                timestamp=timestamp)
            for particle_sv in prior.state_vector]

        if measurement is not None:
            updates = [self.updater.update(SingleHypothesis(prediction, measurement))
                       for prediction in predictions]
        else:
            updates = predictions  # keep the prediction

        # Draw the samples
        samples = np.array([state.state_vector.reshape(-1) +
                            mvn.rvs(cov=state.covar).T
                            for state in updates])

        # Compute the log of q(x_k|x_{k-1}, y_k)
        post_log_weights = np.array([mvn.logpdf(sample,
                                                np.array(update.mean).reshape(-1),
                                                update.covar)
                                     for sample, update in zip(samples, updates)])

        pred_state = Prediction.from_state(prior,
                                           parent=prior,
                                           state_vector=StateVectors(samples.T),
                                           timestamp=timestamp,
                                           transition_model=self.predictor.transition_model,
                                           prior=prior)

        prior_log_weights = self.predictor.transition_model.logpdf(pred_state, prior,
                                                                   time_interval=time_interval)

        pred_state.log_weight = (pred_state.log_weight + prior_log_weights - post_log_weights)

        return pred_state
