import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import uniform
from copy import deepcopy, copy

from stonesoup.base import Property
from stonesoup.types.array import StateVectors
from stonesoup.types.state import State
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

        [2] Devlin, L., Horridge, P., Green, P. L., & Maskell, S. (2021).
        The No-U-Turn sampler as a proposal distribution in a sequential Monte Carlo
        sampler with a near-optimal L-kernel.
        arXiv preprint arXiv:2108.02498.
    """

    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")
    step_size: float = Property(doc='Step size used in the LeapFrog calculation')
    mass_matrix: float = Property(doc='Mass matrix needed for the Hamilitonian equation')
    mapping: tuple = Property(doc="Localisation mapping")
    v_mapping: tuple = Property(doc="Velocity mapping")
    num_dims: int = Property(doc='State dimension')
    num_samples: int = Property(doc='Number of samples')
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
            self.step_size = np.repeat(self.step_size, self.num_samples)

    def target_proposal(self, prior, state, detection,
                        time_interval):
        """Target proposal"""

        tx_logpdf = self.transition_model.logpdf(state, prior, time_interval=time_interval,
                                                 allow_singular=True)
        mx_logpdf = self.measurement_model.logpdf(detection, state, allow_singular=True)
        tg_proposal = tx_logpdf + mx_logpdf

        return tg_proposal

    def grad_target_proposal(self, prior, state, detection, time_interval, **kwargs):

        # grad log prior
        dx = state.state_vector - self.transition_model.function(prior,
                                                                 time_interval=time_interval,
                                                                 **kwargs)

        grad_log_prior = (
            np.linalg.pinv(self.transition_model.covar(time_interval=time_interval)) @ (-dx)
        ).T

        # temporary fix to make the jacobian work with particle state:
        # worth understanding if it is the better choice overall
        temp_x = State(state_vector=state.mean,
                       timestamp=state.timestamp)
        # Get Jacobians of measurements
        H = self.measurement_model.jacobian(temp_x)

        # Get innov
        dy = detection.state_vector - self.measurement_model.function(state, **kwargs)

        # Compute the gradient H^T * inv(R) * innov
        if len(H.shape) < 3:
            # Single Jacobian matrix
            grad_log_pdf = (H.T @ np.linalg.pinv(self.measurement_model.covar()) @ dy).T
        else:
            # Jacobian matrix for each point
            HTinvR = H.transpose((0, 2, 1)) @ self.linalg.pinv(self.measurement_model.covar())
            grad_log_pdf = (HTinvR @ np.atleast_3d(dy))[:, :, 0]

        return grad_log_prior + grad_log_pdf

    def rvs(self, state, measurement: Detection = None, time_interval=None,
            **kwargs):

        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - state.timestamp
        else:
            timestamp = state.timestamp + time_interval

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

        new_state_pred = Prediction.from_state(state,
                                               parent=state,
                                               state_vector=new_state,
                                               timestamp=timestamp,
                                               transition_model=self.transition_model,
                                               prior=state)

        # evaluate the momentum
        v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0], size=self.num_samples)

        if measurement is not None:
            grad_x = self.grad_target_proposal(state, new_state_pred, measurement, time_interval)

            x_new, v_new, acceptance = self.generate_nuts_samples(state, new_state_pred,
                                                                  v, grad_x, measurement,
                                                                  time_interval)

            # pi(x_k)  # state is like prior
            pi_x_k = self.target_proposal(x_new, state, measurement, time_interval)

            # pi(x_k-1)
            pi_x_k1 = self.target_proposal(state, state, measurement, time_interval)

            # we need to add the deteminat of the jacobian of the LF integrator
            # following eq 22 in Alessandro's papers
            # wt = wt-1 * (pi_x_k * qv(-v)/det(J))/(pi_x_k1 * p_x_xk1 * qv(v)/det(J))
            # 1/-1 id the direction

            jk_minus = self.integrate_lf_vec(x_new, state, v, grad_x, 1, self.step_size,
                                             time_interval, measurement)
            jk_plus = self.integrate_lf_vec(x_new, state, v_new, grad_x, -1, self.step_size,
                                            time_interval, measurement)

            j_cab_m = self.get_grad(jk_minus, time_interval)
            j_cab_p = self.get_grad(jk_plus, time_interval)

            determinant_m = np.linalg.det(j_cab_m)
            determinant_p = np.linalg.det(j_cab_p)

            # qv (-v)/det(J)
            q_star_minus = mvn.logpdf(-v_new, mean=np.zeros(v.shape[1]),
                                      cov=self.mass_matrix)/determinant_m
            # qv (v)/det(J)
            q_star_plus = mvn.logpdf(v, mean=np.zeros(v.shape[1]),
                                     cov=self.mass_matrix)/determinant_p

        else:
            # No updates
            x_new = new_state_pred
            pi_x_k = 0
            pi_x_k1 = 0
            q_star_minus = 0
            q_star_plus = 0

        final_state = Prediction.from_state(previous_state,
                                            parent=previous_state,
                                            state_vector=x_new.state_vector,
                                            timestamp=timestamp,
                                            transition_model=self.transition_model,
                                            prior=state)

        final_state.log_weight += pi_x_k - pi_x_k1 + q_star_minus - q_star_plus

        return final_state

    def get_grad(self, new_state, time_interval):
        """Use the Jacobian of the model"""
        return self.transition_model.jacobian(new_state, time_interval=time_interval)

    def integrate_lf_vec(self, state, new_state_pred, v, grad_x, direction, h, time_interval,
                         measurement):
        """Leapfrog integration"""
        h = h.reshape(self.num_samples, 1)
        v = v + direction * (h / 2) * grad_x
        einsum = np.einsum('bij,bj->bi', self.inv_MM, v)
        state.state_vector = (state.state_vector.T + direction * h * einsum).T

        grad_x = self.grad_target_proposal(state, new_state_pred, measurement, time_interval)
        v = v + direction * (h / 2) * grad_x
        return state, v, grad_x

    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        """Stop Criterion"""

        # Return True for particles we want to stop (NB opposite way round to s in Hoffman
        # and Gelman paper)
        dx = xplus.state_vector.T - xminus.state_vector.T
        left = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rminus), axis=1) < 0)
        right = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rplus), axis=1) < 0)
        return np.logical_or(left, right)

    def get_hamiltonian(self, v, logp):
        """Hamiltonian calculation"""

        # Get Hamiltonian energy of system given log target weight logp
        return logp - 0.5 * np.sum(v * np.einsum('bij,bj->bi', self.inv_MM, v),
                                   axis=1).reshape(-1, 1)

    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction):
        """ Auxiliary function to merge the states"""

        # Return xmerge = vectors of xminus where direction < 0 and xplus where direction > 0, and
        # similarly for v and grad_x
        mask = direction[:, 0] < 0
        mask = mask.reshape(len(mask), 1).astype(int)
        xmerge = mask * xminus.state_vector.T + (1 - mask) * xplus.state_vector.T
        vmerge = mask * vminus + (1 - mask) * vplus
        grad_xmerge = mask * grad_xminus + (1 - mask) * grad_xplus
        xminus.state_vector = xmerge.T
        return xminus, vmerge, grad_xmerge

    def generate_nuts_samples(self, x0, x1, v0, grad_x0, detection, time_interval):
        """ Generate NUTS samples"""

        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        logp0 = self.target_proposal(x1, x0, detection,
                                     time_interval=time_interval).reshape(-1, 1)
        joint = self.get_hamiltonian(v0, logp0)
        logu = joint + np.log(uniform.rvs())

        # initialisation
        xminus = x0
        xplus = x0
        vminus = v0
        vplus = v0
        xprime = x0
        vprime = v0
        grad_xplus = grad_x0
        grad_xminus = grad_x0
        depth = 0

        # criteria
        stopped = np.zeros((self.num_samples, 1)).astype(bool)
        numnodes = np.ones((self.num_samples, 1)).astype(int)

        # Used to compute acceptance rate
        alpha = np.zeros((self.num_samples, 1))
        nalpha = np.zeros((self.num_samples, 1)).astype(int)

        while np.any(stopped == 0):

            # Generate random direction in {-1, +1}
            direction = (2 * (uniform.rvs(0, 1, size=self.num_samples)
                              < 0.5).astype(int) - 1).reshape(-1, 1)

            # Get new states from minus and plus depending on direction and build tree
            x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                          vplus, grad_xplus, direction)

            xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, x1, v_pm, grad_x_pm,
                                                                       joint,
                                                                       logu, direction, stopped,
                                                                       depth,
                                                                       time_interval, detection)

            # Split the output back based on direction - keep the stopped samples the same
            idxminus = np.logical_and(np.logical_not(stopped), direction < 0).astype(int)
            xminus = State(state_vector=StateVectors((idxminus * xminus2.state_vector.T
                                                      + (1 - idxminus) * xminus.state_vector.T).T))
            vminus = idxminus * vminus2 + (1 - idxminus) * vminus
            grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
            idxplus = np.logical_and(np.logical_not(stopped), direction >
                                     0).reshape(self.num_samples, 1).astype(int)
            xplus = State(state_vector=StateVectors((idxplus * xplus2.state_vector.T +
                                                     (1 - idxplus) * xplus.state_vector.T).T))
            vplus = idxplus * vplus2 + (1 - idxplus) * vplus
            grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

            # Update acceptance rate
            alpha = np.logical_not(stopped) * alpha2 + stopped * alpha
            nalpha = np.logical_not(stopped) * nalpha2 + stopped * nalpha

            # If no U-turn, choose new state
            samples = uniform.rvs(size=self.num_samples).reshape(-1, 1)
            u = numnodes.reshape(-1, 1) * samples < numnodes2.reshape(-1, 1)

            selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples,
                                                                            1).astype(int)

            xprime = State(state_vector=StateVectors((selectnew * xprime2.state_vector.T +
                                                      (1 - selectnew) * xprime.state_vector.T).T))
            vprime = selectnew * vprime2 + (1 - selectnew) * vprime

            # Update number of nodes and tree height
            numnodes = numnodes + numnodes2
            depth = depth + 1
            if depth > self.max_tree_depth:
                print("Max tree size in NUTS reached")
                break

            # Do U-turn test
            stopped = np.logical_or(stopped, stopped2)
            stopped = np.logical_or(stopped,
                                    self.stop_criterion_vec(xminus, xplus, vminus, vplus).reshape(
                                        -1, 1))

            acceptance = alpha / nalpha

        return xprime, vprime, acceptance

    def build_tree(self, x, x1, v, grad_x, joint, logu, direction, stopped, depth, time_interval,
                   detection):
        """Function to build the particle trees"""

        if depth == 0:

            # Base case
            # ---------

            not_stopped = np.logical_not(stopped)

            # Do leapfrog
            xprime2, vprime2, grad_xprime2 = self.integrate_lf_vec(x, x1, v, grad_x, direction,
                                                                   self.step_size,
                                                                   time_interval, detection)

            idx_notstopped = not_stopped.astype(int)

            xprime = State(state_vector=StateVectors(
                (idx_notstopped * xprime2.state_vector.T + (1 - idx_notstopped)
                 * x.state_vector.T).T))
            vprime = idx_notstopped * vprime2 + (1 - idx_notstopped) * v
            grad_xprime = idx_notstopped * grad_xprime2 + (1 - idx_notstopped) * grad_x

            # Get number of nodes
            logpprime = self.target_proposal(xprime, x, detection,
                                             time_interval=time_interval).reshape(-1, 1)
            jointprime = self.get_hamiltonian(vprime, logpprime)  # xprime
            numnodes = (logu <= jointprime).astype(int)

            # Update acceptance rate
            logalphaprime = np.where(jointprime > joint, 0.0, jointprime - joint)
            alphaprime = np.zeros((self.num_samples, 1))
            alphaprime[not_stopped] = np.exp(logalphaprime[not_stopped[:, 0], 0])
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

            if np.any(stopped == 0):
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
                idxminus = np.logical_and(np.logical_not(stopped), direction < 0).astype(int)
                xminus = State(
                    state_vector=StateVectors((idxminus * xminus2.state_vector.T + (1 - idxminus)
                                               * xminus.state_vector.T).T))
                vminus = idxminus * vminus2 + (1 - idxminus) * vminus
                grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
                idxplus = np.logical_and(np.logical_not(stopped), direction > 0).astype(int)
                xplus = State(
                    state_vector=StateVectors((idxplus * xplus2.state_vector.T + (1 - idxplus)
                                               * xplus.state_vector.T).T))
                vplus = idxplus * vplus2 + (1 - idxplus) * vplus
                grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

                # Do new sampling
                u = numnodes.reshape(-1, 1) * \
                    uniform.rvs(size=self.num_samples).reshape(-1, 1) < \
                    numnodes2.reshape(-1, 1)

                selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples,
                                                                                1).astype(int)
                xprime = State(state_vector=StateVectors((selectnew * xprime2.state_vector.T
                                                          + (1 - selectnew) *
                                                          xprime.state_vector.T).T))
                vprime = selectnew * vprime2 + (1 - selectnew) * vprime

                # Do U-turn test
                stopped = np.logical_or(stopped, stopped2)
                stopped = np.logical_or(stopped, self.stop_criterion_vec(xminus, xplus, vminus,
                                                                         vplus).reshape(-1, 1))

                # Update number of nodes
                not_stopped = np.logical_not(stopped)
                numnodes = numnodes + numnodes2

                # Update acceptance rate
                alpha += not_stopped * alpha2
                nalpha += not_stopped * nalpha2

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, \
                vprime, numnodes, stopped, alpha, nalpha
