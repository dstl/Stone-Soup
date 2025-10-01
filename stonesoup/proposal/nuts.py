import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import uniform
from copy import deepcopy, copy
from datetime import timedelta

from stonesoup.base import Property
from stonesoup.types.array import StateVectors
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.detection import Detection
from stonesoup.types.prediction import Prediction


class NUTSProposal(Proposal):
    """No-U Turn Sampler proposal

        This implementation follows the papers:
        [1] Varsi, A., Devlin, L., Horridge, P., & Maskell, S. (2024).
        A general-purpose fixed-lag no-u-turn sampler for nonlinear non-gaussian
        state space models. IEEE Transactions on Aerospace and Electronic Systems.

        [2] Devlin, L., Carter, M., Horridge, P., Green, P. L., & Maskell, S. (2024).
        The no-u-turn sampler as a proposal distribution in a sequential monte carlo
        sampler without accept/reject. IEEE Signal Processing Letters, 31, 1089-1093.
    """

    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")
    step_size: float = Property(doc='Step size used in the LeapFrog calculation')
    mass_matrix: float = Property(doc='Mass matrix needed for the Hamilitonian equation')
    mapping: tuple = Property(doc="Localisation mapping")
    num_dims: int = Property(doc='State dimension')
    num_samples: int = Property(doc='Number of samples/particles')
    target_proposal_input: float = Property(
        doc='Particle distribution',
        default=None)
    grad_target: float = Property(
        doc='Gradient of the particle distribution',
        default=None)
    max_tree_depth: int = Property(
        doc="Maximum tree depth NUTS can take to stop excessive tree growth.",
        default=10)
    delta_max: int = Property(
        doc='Rejection criteria threshold',
        default=100)

    # Initialise
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MM = np.tile(self.mass_matrix, (self.num_samples, 1, 1))  # mass matrix
        self.inv_MM = np.tile(np.linalg.pinv(self.mass_matrix), (self.num_samples, 1, 1))

        # Ensure step size is an array
        if np.isscalar(self.step_size):
            self.step_size = np.repeat(self.step_size, self.num_samples).reshape(1, -1)

    def _blend_state(self, new_state, old_state, mask):
        """
        Blend two states using a mask
        """
        new_state_vector = StateVectors(
            mask * new_state.state_vector + (1 - mask) * old_state.state_vector)

        return Prediction.from_state(old_state,
                                     parent=old_state,
                                     state_vector=new_state_vector,
                                     timestamp=new_state.timestamp,
                                     transition_model=self.transition_model,
                                     prior=old_state)

    def _blend_array(self, new_array, old_array, mask):
        """
        Blend two arrays using a mask (in this case for velocity and gradients)
        """
        return mask * new_array + (1 - mask) * old_array

    def target_proposal(self, prior_state, state_prediction, detection,
                        time_interval):
        """
        Computes the target proposal distribution

        prior_state: State = Prior state (at time k-1)
        state_prediction: State = Predicted state (at time k)
        detection: Detection = Current detection (at time k)
        time_interval: timedelta = Time interval between prior and predicted state
        """

        if self.target_proposal_input is None:
            # log prior
            tx_logpdf = self.transition_model.logpdf(state_prediction,
                                                     prior_state,
                                                     time_interval=time_interval)

            # log likelihood
            mx_logpdf = self.measurement_model.logpdf(detection,
                                                      state_prediction)

            return tx_logpdf + mx_logpdf
        else:
            # use user defined distribution
            return self.target_proposal_input(prior_state, state_prediction,
                                              detection, time_interval)

    def grad_target_proposal(self, prior_state, state_prediction,
                             detection, time_interval, **kwargs):

        """
        Computes the gradient of the target proposal distribution

        prior_state: State = Prior state (at time k-1)
        state_prediction: State = Predicted state (at time k)
        detection: Detection = Current detection (at time k)
        time_interval: timedelta = Time interval between prior and predicted state
        """

        # use stone soup in case we don't have a target distribution
        if self.grad_target is None:
            # grad log prior
            dx = state_prediction.state_vector - self.transition_model.function(prior_state,
                                                                                time_interval=time_interval,
                                                                                **kwargs)

            grad_log_prior = np.linalg.pinv(self.transition_model.covar(time_interval=time_interval)) @ -dx

            # grad log likelihood
            # Get Jacobians of measurements
            H = np.array([self.measurement_model.jacobian(particle) for particle in
                          state_prediction.particles])

            # Get innov
            dy = detection.state_vector - self.measurement_model.function(state_prediction, **kwargs)

            # Compute the gradient H^T * inv(R) * innov
            if len(H.shape) < 3:
                # Single Jacobian matrix
                grad_log_pdf = H.T @ np.linalg.pinv(self.measurement_model.covar()) @ dy
            else:
                # Jacobian matrix for each point
                HTinvR = H.transpose((0, 2, 1)) @ np.linalg.pinv(self.measurement_model.covar())
                grad_log_pdf = (HTinvR @ dy)[0, :, :]

            return grad_log_prior + grad_log_pdf
        else:
            # use user defined gradient distribution
            return self.grad_target(prior_state, state_prediction,
                                    detection, time_interval, **kwargs)

    def rvs(self, state, measurement: Detection = None,
            time_interval: timedelta = None, **kwargs):

        # check if we have a measurement or not
        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - state.timestamp

            # check that we have the correct measurement model or multiple sensors
            if isinstance(self.measurement_model, measurement.measurement_model.__class__):
                self.measurement_model = measurement.measurement_model
        else:
            timestamp = state.timestamp + time_interval

        # if the time interval is zero return the same state
        if time_interval.total_seconds() == 0:
            return Prediction.from_state(state,
                                         parent=state,
                                         state_vector=state.state_vector,
                                         timestamp=state.timestamp,
                                         transition_model=self.transition_model,
                                         prior=state)

        # Copy the old state for the parent
        previous_state = copy(state)
        # Create a copy of the state vector to ensure the original is not modified
        previous_state.state_vector = deepcopy(state.state_vector)

        # state is the prior - propagate
        new_state = self.transition_model.function(state,
                                                   time_interval=time_interval,
                                                   **kwargs)

        new_state_pred = Prediction.from_state(previous_state,
                                               parent=previous_state,
                                               state_vector=new_state,
                                               timestamp=timestamp,
                                               transition_model=self.transition_model,
                                               prior=previous_state)

        # evaluate the momentum
        v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0],
                    size=self.num_samples).T

        if measurement is not None:
            # evaluate the gradient of the starting state
            grad_x = self.grad_target_proposal(previous_state, new_state_pred,
                                               measurement, time_interval)

            x_new, v_new, _ = self.generate_nuts_samples(previous_state, new_state_pred,
                                                         v, grad_x, measurement,
                                                         time_interval)

            ns = Prediction.from_state(previous_state,
                                       parent=previous_state,
                                       state_vector=x_new.state_vector,
                                       timestamp=x_new.timestamp,
                                       transition_model=self.transition_model,
                                       prior=previous_state)

            # pi(x_k)
            pi_x_k = self.target_proposal(previous_state, ns, measurement, time_interval)

            # pi(x_k-1)
#            pi_x_k1 = self.target_proposal(state, state, measurement, time_interval)

            # re-evaluate the gradient with the new state
            grad_x = self.grad_target_proposal(previous_state, ns,
                                               measurement, time_interval)

            # we need to add the deteminat of the jacobian of the LF integrator
            # following eq 22 in Alessandro's papers
            # wt = wt-1 * (pi_x_k * qv(-v)/det(J))/(pi_x_k1 * p_x_xk1 * qv(v)/det(J))
            # 1/-1 id the direction

            jk_minus, _, _ = self.integrate_lf_vec(ns, v_new, grad_x, -1,
                                                   time_interval, measurement)  # ns
            jk_plus, _, _ = self.integrate_lf_vec(ns, v, grad_x, 1,
                                                  time_interval, measurement)  # ns

            # determinant of the jacobian evaluation
            determinant_m = np.abs(np.linalg.det(self.get_grad(jk_minus, time_interval)))
            determinant_p = np.abs(np.linalg.det(self.get_grad(jk_plus, time_interval)))

            # qv (-v)/det(J)
            q_star_minus = mvn.logpdf(-v_new.T, mean=np.zeros(v.shape[0]),
                                      cov=self.mass_matrix) / determinant_m
            # qv (v)/det(J)
            q_star_plus = mvn.logpdf(v.T, mean=np.zeros(v.shape[0]),
                                     cov=self.mass_matrix) / determinant_p

        else:
            # No updates
            x_new = new_state_pred
            pi_x_k = 0
            q_star_minus = 0
            q_star_plus = 0

        final_state = Prediction.from_state(previous_state,
                                            parent=previous_state,
                                            state_vector=x_new.state_vector,
                                            timestamp=timestamp,
                                            transition_model=self.transition_model,
                                            prior=state)

        final_state.log_weight += pi_x_k + q_star_minus - q_star_plus

        return final_state

    def get_grad(self, new_state, time_interval):
        """Use the Jacobian of the model"""
        return self.transition_model.jacobian(new_state, time_interval=time_interval)

    def integrate_lf_vec(self, state, v, grad_x, direction, time_interval,
                         measurement):
        """Leapfrog integral step

        state: State = Current state
        v: np.ndarray = Current momentum
        grad_x: np.ndarray = Gradient of the target at current state
        direction: int = Direction of the leapfrog step (+1 or -1)
        time_interval: timedelta = Time interval between prior and predicted state
        measurement: Detection = Current detection
        """
        # we might need to do this otherwise we modify the state class
        temp_state = copy(state)
        temp_state.state_vector = deepcopy(state.state_vector)

        v = v + direction * (self.step_size / 2) * grad_x
        einsum = np.einsum('bij,jb->ib', self.inv_MM, v)
        temp_state.state_vector = temp_state.state_vector + direction * self.step_size * einsum

        new_grad_x = self.grad_target_proposal(state, temp_state, measurement, time_interval)
        v = v + direction * (self.step_size / 2.) * new_grad_x
        return temp_state, v, new_grad_x

    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        """
        Particles stop criterion
        """

        # Return True for particles we want to stop
        # (NB opposite way round to s in Hoffman and Gelman paper)
        dx = xplus.state_vector - xminus.state_vector
        vminus = np.einsum('bij,jb->ib', self.inv_MM, rminus)
        vplus = np.einsum('bij,jb->ib', self.inv_MM, rplus)
        left = np.sum(dx * vminus, axis=0) < 0
        right = np.sum(dx * vplus, axis=0) < 0
        return np.logical_or(left, right)

    def get_hamiltonian(self, v, logp):
        """
        Hamiltonian calculation
        """
        # Get Hamiltonian energy of system given log target weight logp
        weight_v = np.einsum('bij,jb->ib', self.inv_MM, v)
        return logp - 0.5 * np.sum(v * weight_v, axis=0)

    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus,
                         grad_xplus, direction):
        """
        Auxiliary function to merge the states
        """

        # Return xmerge = vectors of xminus where direction < 0
        # and xplus where direction > 0, and similarly for v and grad_x
        merge_state = copy(xminus)
        merge_state.state_vector = deepcopy(xminus.state_vector)
        mask = direction < 0
        xmerge = self._blend_state(xminus, xplus, mask).state_vector
        vmerge = self._blend_array(vminus, vplus, mask)
        grad_xmerge = self._blend_array(grad_xminus, grad_xplus, mask)
        merge_state.state_vector = xmerge
        return merge_state, vmerge, grad_xmerge

    def generate_nuts_samples(self, x0, x1, v0, grad_x0, detection, time_interval):
        """
        Generate NUTS samples

        x0: State = Prior state (at time k-1)
        x1: State = Predicted state (at time k)
        v0: np.ndarray = Initial momentum
        grad_x0: np.ndarray = Gradient of the target at predicted state
        detection: Detection = Current detection (at time k)
        time_interval: timedelta = Time interval between prior and predicted state
        """

        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        logp0 = self.target_proposal(x0, x1, detection,
                                     time_interval=time_interval).reshape(1, -1)
        joint = self.get_hamiltonian(v0, logp0)
        logu = joint + np.log(uniform.rvs(size=self.num_samples))

        # initialisation / consider the copy and deepcopy of it
        xminus, xplus, xprime = copy(x1), copy(x1), copy(x1)
        xminus.state_vector = deepcopy(x1.state_vector)
        xplus.state_vector = deepcopy(x1.state_vector)
        xprime.state_vector = deepcopy(x1.state_vector)
        vplus, vminus, vprime = v0, v0, v0
        grad_xplus, grad_xminus = grad_x0, grad_x0
        depth = 0

        # criteria
        stopped = np.zeros((1, self.num_samples), dtype=bool)
        numnodes = np.ones((1, self.num_samples), dtype=int)

        # Used to compute acceptance rate
        alpha = np.zeros((1, self.num_samples))
        nalpha = np.zeros((1, self.num_samples), dtype=int)

        while np.any(~stopped):

            # Generate random direction in {-1, +1}
            direction = (2 * (uniform.rvs(size=self.num_samples) < 0.5).astype(int) - 1).reshape(1, -1)

            # Get new states from minus and plus depending on direction and build tree
            x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                          vplus, grad_xplus, direction)

            xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, x0, v_pm, grad_x_pm,
                                                                       joint,
                                                                       logu, direction, stopped,
                                                                       depth,
                                                                       time_interval, detection)

            # Split the output back based on direction - keep the stopped samples the same
            idxminus = np.logical_and(~stopped, direction < 0, dtype=int)
            xminus = self._blend_state(xminus2, xminus, idxminus)
            vminus = self._blend_array(vminus2, vminus, idxminus)
            grad_xminus = self._blend_array(grad_xminus2, grad_xminus, idxminus)

            idxplus = np.logical_and(~stopped, direction > 0, dtype=int)
            xplus = self._blend_state(xplus2, xplus, idxplus)
            vplus = self._blend_array(vplus2, vplus, idxplus)
            grad_xplus = self._blend_array(grad_xplus2, grad_xplus, idxplus)

            # Update acceptance rate
            alpha = (~stopped) * alpha2 + stopped * alpha
            nalpha = (~stopped) * nalpha2 + stopped * nalpha

            # If no U-turn, choose new state
            samples = uniform.rvs(size=self.num_samples).reshape(1, -1)
            u = numnodes * samples < numnodes2

            selectnew = np.logical_and(~stopped2, u, dtype=int)

            xprime = self._blend_state(xprime2, xprime, selectnew)
            vprime = self._blend_array(vprime2, vprime, selectnew)

            # Update number of nodes and tree height
            numnodes = numnodes + numnodes2
            depth += 1
            if depth > self.max_tree_depth:
                print("Max tree size in NUTS reached")
                break

            # Do U-turn test
            stopped = np.logical_or(stopped, stopped2)
            stopped = np.logical_or(stopped,
                                    self.stop_criterion_vec(xminus, xplus, vminus, vplus))

            acceptance = alpha / nalpha

        return xprime, vprime, acceptance

    def build_tree(self, x, x1, v, grad_x, joint, logu, direction,
                   stopped, depth, time_interval, detection):
        """
        Function to build the particle trees
        """

        if depth == 0:

            # Base case
            # ---------

            not_stopped = ~stopped
            idx_notstopped = not_stopped.astype(int)

            # Do leapfrog
            xprime2, vprime2, grad_xprime2 = self.integrate_lf_vec(x, v, grad_x, direction,
                                                                   time_interval, detection)

            xprime = self._blend_state(xprime2, x, idx_notstopped)
            vprime = self._blend_array(vprime2, v, idx_notstopped)
            grad_xprime = self._blend_array(grad_xprime2, grad_x, idx_notstopped)

            # Get number of nodes
            logpprime = self.target_proposal(x1, xprime, detection,
                                             time_interval=time_interval).reshape(1, -1)
            jointprime = self.get_hamiltonian(vprime, logpprime)
            numnodes = (logu <= jointprime).astype(int)

            # Update acceptance rate
            logalphaprime = np.where(jointprime > joint, 0.0, jointprime - joint)
            alphaprime = np.zeros((1, self.num_samples))
            alphaprime[0, not_stopped[0, :]] = np.exp(logalphaprime[0, not_stopped[0, :]])
            alphaprime[np.isnan(alphaprime)] = 0.0
            nalphaprime = np.ones_like(alphaprime, dtype=int)

            # Stop bad samples
            stopped = np.logical_or(stopped, logu - self.delta_max >= jointprime)

            return xprime, vprime, grad_xprime, xprime, vprime, grad_xprime, xprime, vprime, \
                numnodes, stopped, alphaprime, nalphaprime

        else:

            # Recursive case
            # --------------

            # Build one subtree
            xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, vprime, \
                numnodes, stopped, alpha, nalpha = self.build_tree(x, x1, v, grad_x, joint, logu,
                                                                   direction, stopped, depth - 1,
                                                                   time_interval,
                                                                   detection)

            if np.any(~stopped):
                # Get new states from minus and plus depending on direction and build tree
                x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                              vplus, grad_xplus, direction)

                xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                    numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, x1, v_pm,
                                                                           grad_x_pm, joint,
                                                                           logu, direction,
                                                                           stopped, depth - 1,
                                                                           time_interval,
                                                                           detection)

                # Split the output back based on direction - keep the stopped samples the same
                idxminus = np.logical_and(~stopped, direction < 0, dtype=int)
                xminus = self._blend_state(xminus2, xminus, idxminus)
                vminus = self._blend_array(vminus2, vminus, idxminus)
                grad_xminus = self._blend_array(grad_xminus2, grad_xminus, idxminus)

                idxplus = np.logical_and(~stopped, direction > 0, dtype=int)
                xplus = self._blend_state(xplus2, xplus, idxplus)
                vplus = self._blend_array(vplus2, vplus, idxplus)
                grad_xplus = self._blend_array(grad_xplus2, grad_xplus, idxplus)

                # Do new sampling
                samples = uniform.rvs(size=self.num_samples).reshape(1, -1)
                u = numnodes * samples < numnodes2

                selectnew = np.logical_and(~stopped2, u, dtype=int)

                xprime = self._blend_state(xprime2, xprime, selectnew)
                vprime = self._blend_array(vprime2, vprime, selectnew)

                # Do U-turn test
                stopped = np.logical_or(stopped, stopped2)
                stopped = np.logical_or(stopped,
                                        self.stop_criterion_vec(xminus, xplus,
                                                                vminus, vplus))

                # Update number of nodes
                not_stopped = ~stopped
                numnodes = numnodes + numnodes2

                # Update acceptance rate
                alpha += not_stopped * alpha2
                nalpha += not_stopped * nalpha2

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, \
                vprime, numnodes, stopped, alpha, nalpha
