import copy
from functools import lru_cache
from typing import Callable
import warnings

import numpy as np
from scipy.linalg import inv
from scipy.special import logsumexp

from .base import Updater
from .kalman import KalmanUpdater, ExtendedKalmanUpdater
from ..base import Property
from ..functions import cholesky_eps, sde_euler_maruyama_integration
from ..predictor.particle import MultiModelPredictor, RaoBlackwellisedMultiModelPredictor
from ..resampler import Resampler
from ..regulariser import Regulariser
from ..types.prediction import (
    Prediction, ParticleMeasurementPrediction, GaussianStatePrediction, MeasurementPrediction)
from ..types.update import ParticleStateUpdate, Update


class ParticleUpdater(Updater):
    """Particle Updater

    Perform an update by multiplying particle weights by PDF of measurement
    model (either :attr:`~.Detection.measurement_model` or
    :attr:`measurement_model`), and normalising the weights. If provided, a
    :attr:`resampler` will be used to take a new sample of particles (this is
    called every time, and it's up to the resampler to decide if resampling is
    required).
    """

    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    regulariser: Regulariser = Property(
        default=None,
        doc='Regulariser to prevent particle impoverishment. The regulariser '
            'is normally used after resampling. If a class:`~.Resampler` is defined,'
            ' then regularisation will only take place if the particles have been '
            'resampled. If the class:`~.Resampler` is not defined but a '
            'class:`~.Regulariser` is, then regularisation will be conducted under the '
            'assumption that the user intends for this to occur.')

    constraint_func: Callable = Property(
        default=None,
        doc="Callable, user defined function for applying "
            "constraints to the states. This is done by setting the weights "
            "of particles to 0 for particles that are not correctly constrained. "
            "This function provides indices of the unconstrained particles and "
            "should accept a :class:`~.ParticleState` object and return an array-like "
            "object of logical indices. "
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.resampler is None and self.regulariser is not None:
            warnings.warn('`regulariser` has been defined but a `resampler` has not. This'
                          ' is not normal procedure.')

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """

        predicted_state = Update.from_state(
            state=hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.prediction.timestamp
        )

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        new_weight = predicted_state.log_weight + measurement_model.logpdf(
            hypothesis.measurement, predicted_state, **kwargs)

        # Apply constraints if defined
        if self.constraint_func is not None:
            part_indx = self.constraint_func(predicted_state)
            new_weight[part_indx] = -1*np.inf

        # Normalise the weights
        new_weight -= logsumexp(new_weight)

        predicted_state.log_weight = new_weight

        # Resample
        resample_flag = True
        if self.resampler is not None:
            resampled_state = self.resampler.resample(predicted_state)
            if resampled_state == predicted_state:
                resample_flag = False

        if self.regulariser is not None and resample_flag:
            prior = hypothesis.prediction.parent
            if self.resampler is None:
                predicted_state = self.regulariser.regularise(prior,
                                                              predicted_state)
            else:
                predicted_state = self.regulariser.regularise(prior,
                                                              resampled_state)

        return predicted_state

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_state_vector = measurement_model.function(state_prediction, **kwargs)

        return MeasurementPrediction.from_state(
            state_prediction, state_vector=new_state_vector, timestamp=state_prediction.timestamp)


class GromovFlowParticleUpdater(Updater):
    """Gromov Flow Particle Updater

    This is implementation of Gromov method for stochastic particle flow
    filters [#]_. The Euler Maruyama method is used for integration, over 20
    steps using an exponentially increase step size.

    Parameters
    ----------

    References
    ----------
    .. [#] Daum, Fred & Huang, Jim & Noushin, Arjang. "Generalized Gromov
           method for stochastic particle flow filters." 2017
    """

    def update(self, hypothesis, **kwargs):

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        num_steps = 20
        b = 2
        s0 = (b-1) / (b**num_steps - 1)
        steps = [s0*b**n for n in range(num_steps)]

        time_steps = np.zeros((len(steps) + 1, ))
        time_steps[1:] = np.cumsum(steps)

        P = hypothesis.prediction.covar
        R = measurement_model.covar()
        inv_R = inv(R)

        # Start by making our own copy of the particle before we move them...
        particles = [copy.copy(particle) for particle in hypothesis.prediction.particles]

        def function(state, lambda_):
            try:
                H = measurement_model.matrix()
            except AttributeError:
                H = measurement_model.jacobian(state)

            # Eq. (12) Ref [1]
            a = P - lambda_*P@H.T@inv(R + lambda_*H@P@H.T)@H@P
            b = a @ H.T @ inv_R

            measurement_particle_state_vector = measurement_model.function(state, **kwargs)
            f = -b @ (measurement_particle_state_vector - hypothesis.measurement.state_vector)

            Q = b @ H @ a
            B = cholesky_eps((Q+Q.T)/2)

            return f, B

        for particle in particles:
            particle.state_vector = sde_euler_maruyama_integration(function, time_steps, particle)

        return ParticleStateUpdate(
            None,
            hypothesis,
            particle_list=particles,
            timestamp=hypothesis.measurement.timestamp)

    predict_measurement = ParticleUpdater.predict_measurement


class GromovFlowKalmanParticleUpdater(GromovFlowParticleUpdater):
    """Gromov Flow Parallel Kalman Particle Updater

    This is a wrapper around the :class:`~.GromovFlowParticleUpdater` which
    can use a :class:`~.ExtendedKalmanUpdater` or
    :class:`~.UnscentedKalmanUpdater` in parallel in order to maintain a state
    covariance, as proposed in [#]_. In this implementation, the mean of the
    :class:`~.ParticleState` is used the EKF/UKF update.

    This should be used in conjunction with the
    :class:`~.ParticleFlowKalmanPredictor`.

    Parameters
    ----------

    References
    ----------
    .. [#] Ding, Tao & Coates, Mark J., "Implementation of the Daum-Huang
       Exact-Flow Particle Filter" 2012
    """
    kalman_updater: KalmanUpdater = Property(
        default=None,
        doc="Kalman updater to use. Default `None` where a new instance of"
            ":class:`~.ExtendedKalmanUpdater` will be created utilising the"
            "same measurement model.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.kalman_updater is None:
            self.kalman_updater = ExtendedKalmanUpdater(self.measurement_model)

    def update(self, hypothesis, **kwargs):
        particle_update = super().update(hypothesis, **kwargs)

        particle_pred = hypothesis.prediction

        kalman_hypothesis = copy.copy(hypothesis)
        # Convert to GaussianState
        kalman_hypothesis.prediction = GaussianStatePrediction(
            particle_pred.mean, particle_pred.covar, particle_pred.timestamp)
        # Needed for cross covar
        kalman_hypothesis.measurement_prediction = None
        kalman_update = self.kalman_updater.update(kalman_hypothesis, **kwargs)

        return ParticleStateUpdate(
            particle_update.state_vector,
            hypothesis,
            weight=particle_update.weight,
            fixed_covar=kalman_update.covar,
            timestamp=particle_update.timestamp)

    def predict_measurement(self, state_prediction, *args, **kwargs):
        particle_prediction = super().predict_measurement(
            state_prediction, *args, **kwargs)

        kalman_prediction = self.kalman_updater.predict_measurement(
            Prediction.from_state(
                state_prediction, state_prediction.state_vector, state_prediction.covar,
                target_type=GaussianStatePrediction),
            *args, **kwargs)

        return ParticleMeasurementPrediction(
            state_vector=particle_prediction.state_vector,
            weight=state_prediction.weight,
            fixed_covar=kalman_prediction.covar,
            timestamp=particle_prediction.timestamp)


class MultiModelParticleUpdater(ParticleUpdater):
    """Particle Updater for the Multi Model system"""

    predictor: MultiModelPredictor = Property(
        doc="Predictor which hold holds transition matrix")

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.MultiModelParticleStateUpdate`
            The state posterior
        """
        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        update = Update.from_state(
            hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.measurement.timestamp,
        )

        transition_matrix = np.asanyarray(self.predictor.transition_matrix)

        update.log_weight = update.log_weight \
            + measurement_model.logpdf(hypothesis.measurement, update, **kwargs) \
            + np.log(transition_matrix[update.parent.dynamic_model, update.dynamic_model])

        # Normalise the weights
        update.log_weight -= logsumexp(update.log_weight)

        if self.resampler:
            update = self.resampler.resample(update)
        return update


class RaoBlackwellisedParticleUpdater(MultiModelParticleUpdater):
    """Particle Updater for the Raoblackwellised scheme"""

    predictor: RaoBlackwellisedMultiModelPredictor = Property(
        doc="Predictor which hold holds transition matrix, models and mappings")

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.RaoBlackwellisedParticleStateUpdate`
            The state posterior
        """

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        update = Update.from_state(
            hypothesis.prediction,
            log_weight=copy.copy(hypothesis.prediction.log_weight),
            hypothesis=hypothesis,
            timestamp=hypothesis.measurement.timestamp,
        )

        update.model_probabilities = self.calculate_model_probabilities(
            hypothesis.prediction, self.predictor)

        update.log_weight = update.log_weight + measurement_model.logpdf(hypothesis.measurement,
                                                                         update,
                                                                         **kwargs)

        # Normalise the weights
        update.log_weight -= logsumexp(update.log_weight)

        if self.resampler:
            update = self.resampler.resample(update)
        return update

    @staticmethod
    def calculate_model_probabilities(prediction, predictor):
        """Calculates the new model probabilities based
            on the ones calculated in the previous time step"""

        denominator_components = []
        # Loop over the previous models m_k-1
        for model_index, transition_model in enumerate(predictor.transition_models):
            required_space_prior = copy.copy(prediction.parent)
            required_space_prior.state_vector = \
                required_space_prior.state_vector[predictor.model_mappings[model_index], :]
            required_space_pred = copy.copy(prediction)
            required_space_pred.state_vector = \
                required_space_pred.state_vector[predictor.model_mappings[model_index], :]

            prob_position_given_model_and_old_position = transition_model.pdf(
                required_space_pred, required_space_prior,
                time_interval=prediction.timestamp - prediction.parent.timestamp
            )

            # Looks up p(m_k|m_k-1)
            log_prob_of_transition = np.log(
                np.asfarray(predictor.transition_matrix)[model_index, :])

            log_product_of_probs = \
                np.asfarray(np.log(prob_position_given_model_and_old_position)) \
                + log_prob_of_transition[:, np.newaxis] \
                + np.asfarray(np.log(prediction.model_probabilities))  # p(m_k-1|x_1:k-1)

            denominator_components.append(logsumexp(log_product_of_probs, axis=0))

        # Calculate the denominator
        new_log_probabilities = np.stack(denominator_components)
        new_log_probabilities -= logsumexp(new_log_probabilities, axis=0)

        return np.exp(new_log_probabilities)


class BernoulliParticleUpdater(ParticleUpdater):
    """Bernoulli Particle Filter Updater class

    An implementation of a particle filter updater utilising the
    Bernoulli filter formulation that estimates the spatial distribution
    of a single target and estimates its existence, as described in [1]_.

    Due to the nature of the Bernoulli particle
    filter prediction process, resampling is required at every
    time step to reduce the number of particles back down to the
    desired size.

    References
    ----------
    .. [1] Ristic, Branko & Vo, Ba-Toung & Vo, Ba-Ngu & Farina, Alfonso, A
       Tutorial on Bernoulli Filters: Theory, Implementation and Applications,
       2013, IEEE Transactions on Signal Processing, 61(13), 3406-3430.
    """

    birth_probability: float = Property(
        default=0.01,
        doc="Probability of target birth.")

    survival_probability: float = Property(
        default=0.98,
        doc="Probability of target survival")

    clutter_rate: int = Property(
        default=1,
        doc="Average number of clutter measurements per time step. Implementation assumes number "
            "of clutter measurements follows a Poisson distribution")

    clutter_distribution: float = Property(
        default=None,
        doc="Distribution used to describe clutter measurements. This is usually assumed uniform "
            "in the measurement space.")
    detection_probability: float = Property(
        default=None,
        doc="Probability of detection assigned to the generated samples of the birth distribution."
            " If None, it will inherit from the input.")
    nsurv_particles: float = Property(
        default=None,
        doc="Number of particles describing the surviving distribution, which will be output from "
            "the update algorithm.")

    def update(self, hypotheses, **kwargs):
        """Bernoulli Particle Filter update step

        Parameters
        ----------
        hypotheses : :class:`~.MultipleHypothesis`
            Hypotheses containing the sequence of single hypotheses
            that contain the prediction and unassociated measurements.
        Returns
        -------
        : :class:`~.BernoulliParticleStateUpdate`
            The state posterior.
        """
        # copy prediction
        prediction = hypotheses.single_hypotheses[0].prediction

        # updated_state = copy.copy(prediction)
        updated_state = Update.from_state(
            state=prediction,
            hypothesis=hypotheses,
            timestamp=prediction.timestamp
        )
        if any(hypotheses):
            detections = [single_hypothesis.measurement
                          for single_hypothesis in hypotheses.single_hypotheses]

            # Evaluate measurement likelihood and approximate integrals
            log_meas_likelihood = []
            delta_part2 = []
            log_detection_probability = np.full(len(prediction),
                                                np.log(self.detection_probability))

            for detection in detections:
                measurement_model = detection.measurement_model or self.measurement_model
                log_meas_likelihood.append(measurement_model.logpdf(detection, updated_state))
                delta_part2.append(self._log_space_product(
                    log_detection_probability,
                    log_meas_likelihood[-1]
                    - np.log(self.clutter_rate * self.clutter_distribution)
                    + updated_state.log_weight))

            delta = \
                np.exp(self._log_space_product(log_detection_probability,
                                               updated_state.log_weight)) \
                - np.exp(logsumexp(delta_part2))

            updated_state.existence_probability = \
                (1 - delta) \
                / (1 - updated_state.existence_probability * delta) \
                * updated_state.existence_probability

            updated_state.log_weight = \
                np.logaddexp(log_detection_probability
                             + logsumexp(log_meas_likelihood, axis=0)
                             - np.log(self.clutter_rate * self.clutter_distribution),
                             np.log(1 - self.detection_probability)) \
                + updated_state.log_weight

            # Normalise weights
            updated_state.log_weight -= logsumexp(updated_state.log_weight)

        # Resampling
        if self.resampler is not None:
            updated_state = self.resampler.resample(updated_state,
                                                    self.nsurv_particles)

        if any(hypotheses):
            # Regularisation
            if self.regulariser is not None:
                updated_state = self.regulariser.regularise(updated_state.parent,
                                                            updated_state)

        return updated_state

    @staticmethod
    def _log_space_product(A, B):
        if A.ndim < 2:
            A = np.atleast_2d(A)
        if B.ndim < 2:
            B = np.atleast_2d(B).T
        Astack = np.stack([A] * B.shape[1]).transpose(1, 0, 2)
        Bstack = np.stack([B] * A.shape[0]).transpose(0, 2, 1)
        return np.squeeze(logsumexp(Astack + Bstack, axis=2))
