import copy
import numpy as np
from collections.abc import Callable, Sequence

from scipy.stats import multivariate_normal, uniform

from .base import Regulariser
from ..base import Property
from ..functions import cholesky_eps
from ..models.transition import TransitionModel
from ..predictor.particle import MultiModelPredictor
from ..types.state import ParticleState


class MCMCRegulariser(Regulariser):
    """Markov chain Monte-Carlo (MCMC) move steps, or regularisation steps, can be implemented in
    particle filters to prevent sample impoverishment that results from resampling.
    One way of avoiding this is to only perform resampling when deemed necessary by some measure
    of effectiveness. Sometimes this is not desirable, or possible, when a particular algorithm
    requires the introduction of new samples as part of the filtering process for example.

    This is a particular implementation of a MCMC move step that uses the Metropolis-Hastings
    algorithm [1]_. After resampling, particles are moved a small amount, according do a Gaussian
    kernel, to a new state only if the Metropolis-Hastings acceptance probability is met by a
    random number assigned to each particle from a uniform random distribution, otherwise they
    remain the same. Further details on the implementation are given in [2]_.

    References
    ----------
    .. [1] Robert, Christian P. & Casella, George, Monte Carlo Statistical Methods, Springer, 1999.

    .. [2] Ristic, Branko & Arulampalam, Sanjeev & Gordon, Neil, Beyond the Kalman Filter:
        Particle Filters for Target Tracking Applications, Artech House, 2004. """

    transition_model: TransitionModel = Property(doc="Transition model used for prediction",
                                                 default=None)
    constraint_func: Callable = Property(
        default=None,
        doc="Callable, user defined function for applying "
            "constraints to particle states. This is done by reverting particles "
            "that are moved to a state outside of the defined constraints "
            "back to the state prior to the move step. Particle states that are "
            "input are assumed to be constrained. This function provides indices "
            "of the unconstrained particles and should accept a :class:`~.ParticleState` "
            "object and return an array-like object of logical indices. "
    )

    def _transition_prior(self, prior, posterior, **kwargs):
        hypothesis = posterior.hypothesis[0] if isinstance(posterior.hypothesis, Sequence) \
            else posterior.hypothesis
        transition_model = hypothesis.prediction.transition_model
        if not transition_model:
            transition_model = self.transition_model

        if transition_model:
            transitioned_prior = copy.copy(prior)
            transitioned_prior.state_vector = transition_model.function(
                prior, noise=False, time_interval=posterior.timestamp - prior.timestamp)
            return transitioned_prior
        else:
            return prior

    def regularise(self, prior, posterior, **kwargs):
        """Regularise the particles

        Parameters
        ----------
        prior : :class:`~.ParticleState` type
            prior particle distribution.
        posterior : :class:`~.ParticleState` type
            posterior particle distribution.

        Returns
        -------
        particle state: :class:`~.ParticleState`
           The particle state after regularisation
        """

        if not isinstance(posterior, ParticleState):
            raise TypeError('Only ParticleState type is supported!')

        if not isinstance(prior, ParticleState):
            raise TypeError('Only ParticleState type is supported!')

        hypotheses = posterior.hypothesis if isinstance(posterior.hypothesis, Sequence) \
            else [posterior.hypothesis]
        detections = {hypothesis.measurement for hypothesis in hypotheses if hypothesis}

        if detections:
            ndim = prior.state_vector.shape[0]
            nparticles = len(posterior)

            measurement_model = next(iter(detections)).measurement_model

            # calculate the optimal bandwidth for the Gaussian kernel
            hopt = (4/(ndim+2))**(1/(ndim+4))*nparticles**(-1/(ndim+4))
            covar_est = posterior.covar

            # move particles
            moved_particles = copy.copy(posterior)
            moved_particles.state_vector = moved_particles.state_vector + \
                hopt * cholesky_eps(covar_est) @ np.random.randn(ndim, nparticles)

            # Apply constraints if defined
            if self.constraint_func is not None:
                part_indx = self.constraint_func(moved_particles)
                moved_particles.state_vector[:, part_indx] = posterior.state_vector[:, part_indx]

            # Evaluate likelihoods
            transitioned_prior = self._transition_prior(prior, posterior, **kwargs)
            part_diff = moved_particles.state_vector - transitioned_prior.state_vector
            move_likelihood = multivariate_normal.logpdf(part_diff.T,
                                                         cov=covar_est)
            post_part_diff = posterior.state_vector - transitioned_prior.state_vector
            post_likelihood = multivariate_normal.logpdf(post_part_diff.T,
                                                         cov=covar_est)

            # Evaluate measurement likelihoods
            move_meas_likelihood = []
            post_meas_likelihood = []
            for detection in detections:
                move_meas_likelihood.append(measurement_model.logpdf(detection, moved_particles))
                post_meas_likelihood.append(measurement_model.logpdf(detection, posterior))

            # In the case that there are multiple measurements,
            # we select the highest overall likelihood.
            max_likelihood_idx = np.argmax(np.sum(move_meas_likelihood, axis=1))

            # Calculate acceptance probability (alpha)
            # with np.errstate(invalid="ignore"):
            with np.errstate(invalid='ignore', over='ignore'):
                alpha = np.exp((move_meas_likelihood[max_likelihood_idx] + move_likelihood) -
                               (post_meas_likelihood[max_likelihood_idx] + post_likelihood))

            # All 'jittered' particles that are above the alpha threshold are kept, the rest are
            # rejected and the original posterior used
            selector = uniform.rvs(size=nparticles)
            index = alpha > selector

            posterior = copy.copy(posterior)
            posterior.state_vector = copy.copy(posterior.state_vector)
            posterior.state_vector[:, index] = moved_particles.state_vector[:, index]

        return posterior


class MultiModelMCMCRegulariser(MCMCRegulariser):
    """MCMC Regulariser for :class:`~.MultiModelParticleState`

    This is a version of the :class:`~.MCMCRegulariser` that supports case where multiple
    models are used i.e. with :class:`~.MultiModelParticleUpdater`.
    """
    transition_model = None
    transition_models: Sequence[TransitionModel] = Property(
        doc="Transition models used to for particle transition, selected by model index on "
            "particle. Models dimensions can be subset of the overall state space, by "
            "using :attr:`model_mappings`."
    )
    model_mappings: Sequence[Sequence[int]] = Property(
        doc="Sequence of mappings associated with each transition model. This enables mapping "
            "between model and state space, enabling use of models that may have different "
            "dimensions (e.g. velocity or acceleration). Parts of the state that aren't mapped "
            "are set to zero.")

    def _transition_prior(self, prior, posterior, **kwargs):
        transitioned_prior = copy.copy(prior)
        transitioned_prior.state_vector = copy.copy(prior.state_vector)
        for model_index, transition_model in enumerate(self.transition_models):
            current_model_indices = prior.dynamic_model == model_index
            current_model_count = np.count_nonzero(current_model_indices)
            if current_model_count == 0:
                continue

            new_state_vector = MultiModelPredictor.apply_model(
                prior[current_model_indices], transition_model, posterior.timestamp,
                self.model_mappings[model_index], noise=False)

            transitioned_prior.state_vector[:, current_model_indices] = new_state_vector
        return transitioned_prior
