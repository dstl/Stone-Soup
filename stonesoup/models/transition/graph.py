from typing import Optional, Union

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from stonesoup.base import Property
from stonesoup.functions.graph import normalise_re
from stonesoup.models.base import TimeVariantModel
from stonesoup.models.transition import TransitionModel
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.graph import RoadNetwork
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State


class ShortestPathToDestinationTransitionModel(TransitionModel, TimeVariantModel):
    r""" Shortest path to destination transition model.

    A transition model that models a target travelling along the shortest path to a destination,
    on a given road network. The model is a generalised implementation of the transition model
    described in [1]_.

    The state vector :math:`x_k` is assumed to take the form

    .. math::
        x_k = \left[r_k, \cdots, e_k, d_k, s_k\right]

    where :math:`e_k` denotes the edge the target is currently on, :math:`r_k` is the distance
    travelled along the edge, :math:`d_k` is the destination node, and :math:`s_k` is the source
    node. The notation :math:`\cdots` denotes additional state variables thar are propagated using
    the selected `transition_model` along with `r_k` (e.g. velocity :math:`\dot{r}_k`).

    The transition model provides the ability to resample the destination node with a given
    `destination_resample_probability`. This is useful when the destination node is unknown and
    needs to be estimated (e.g. using a particle filter). The destination node is resampled from
    the set of possible destinations, given the current edge. To avoid searching the entire graph
    for possible destinations, the `possible_destinations` argument can be used to specify a list
    of possible destinations.

    References
    ----------
    .. [1] L. Vladimirov and S. Maskell, "A SMC Sampler for Joint Tracking and Destination
           Estimation from Noisy Data," 2020 IEEE 23rd International Conference on Information
           Fusion (FUSION), Rustenburg, South Africa, 2020, pp. 1-8,
           doi: 10.23919/FUSION45008.2020.9190463.
    """
    transition_model: LinearGaussianTransitionModel = Property(
        doc=r"A base transition model that models the movement of the target along a given edge. "
            "This can be any model, with the restriction that the first state variable of the "
            "transition model must be the distance travelled along the edge (i.e. :math:`r`).")
    graph: RoadNetwork = Property(
        doc="The road network that the target is moving on.")
    destination_resample_probability: Probability = Property(
        default=0.1,
        doc="The probability of resampling the destination. This is useful when the destination "
            "node is unknown and needs to be estimated (e.g. using a particle filter). If None, "
            "or 0, then the destination is not resampled.")
    possible_destinations: list = Property(
        default=None,
        doc="The possible destinations that the target can travel to. Restricting the possible "
            "destinations can greatly speed up the destination sampling process, since the "
            "shortest path algorithm does not need to search the entire graph. If None, then all "
            "nodes in the graph are considered possible destinations.")
    seed: Optional[int] = Property(
        default=None,
        doc="Seed for random number generation.")

    @property
    def ndim_state(self):
        return self.transition_model.ndim + 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(self.seed) if self.seed is not None else None

    def function(self, state, noise=False, **kwargs):

        num_particles = state.state_vector.shape[1]

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=num_particles, **kwargs)
            else:
                noise = 0

        # 1) CV (Position-Velocity) Propagation
        t_matrix = block_diag(*[self.transition_model.matrix(**kwargs), np.eye(3)])
        new_state_vectors = t_matrix @ state.state_vector + noise

        # 2) SMC destination sampling
        # Get all valid destinations given the current edge
        if self.destination_resample_probability:
            edges = new_state_vectors[-3, :].astype(int)
            unique_edges = np.unique(edges)
            sources = list(np.unique(new_state_vectors[-1, :].astype(int)))
            s_paths = self.graph.shortest_path(sources, self.possible_destinations,
                                               path_type='edge')
            v_dest = dict()
            for edge in unique_edges:
                # Filter paths that contain the edge
                filtered_paths = filter(lambda x: np.any(x[1] == edge), s_paths.items())
                v_dest_tmp = {dest for (_, dest), _ in filtered_paths}
                if len(v_dest_tmp):
                    try:
                        v_dest[edge] |= v_dest_tmp
                    except KeyError:
                        v_dest[edge] = v_dest_tmp

            # Perform destination sampling
            resample_inds = np.flatnonzero(
                np.random.binomial(1, float(self.destination_resample_probability), num_particles)
            )
            for i in resample_inds:
                try:
                    v_dest_tmp = list(v_dest[edges[i]])
                except KeyError:
                    # If no valid destinations exist for the current edge, keep the current
                    # destination
                    continue
                new_state_vectors[-2, i] = np.random.choice(v_dest_tmp)

        # 3) Process edge change
        # The CV propagation may lead to r's which are either less that zero or more than the
        # length of the edge. This means that the range and edge identifier needs to be adjusted
        # to correctly place the particle.

        # Get shortcuts for faster accessing
        r = new_state_vectors[0, :]
        e = new_state_vectors[-3, :].astype(int)
        d = new_state_vectors[-2, :].astype(int)
        s = new_state_vectors[-1, :].astype(int)

        for i in range(num_particles):
            try:
                path = self.graph.shortest_path(s[i], d[i], path_type='edge')[(s[i], d[i])]
                r_i, e_i = normalise_re(r[i], e[i], path, self.graph)
                new_state_vectors[0, i] = r_i
                new_state_vectors[-3, i] = e_i
            except KeyError:
                continue

        return new_state_vectors

    def rvs(self, num_samples: int = 1, random_state=None, **kwargs) ->\
            Union[StateVector, StateVectors]:

        covar = self._covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate rvs from None-type covariance")

        random_state = random_state if random_state is not None else self.random_state

        noise = multivariate_normal.rvs(
            np.zeros(self.ndim), covar, num_samples, random_state=random_state)

        noise = np.atleast_2d(noise)

        noise = noise.T  # numpy.rvs method squeezes 1-dimensional matrices to integers

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        covar = self._covar(**kwargs)

        # If model has None-type covariance or contains None, it does not represent a Gaussian
        if covar is None or None in covar:
            raise ValueError("Cannot generate pdf from None-type covariance")

        # Calculate difference before to handle custom types (mean defaults to zero)
        # This is required as log pdf coverts arrays to floats
        likelihood = np.atleast_1d(
            multivariate_normal.logpdf((state1.state_vector - self.function(state2, **kwargs)).T,
                                       cov=covar, allow_singular=True))

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))

    def _covar(self, **kwargs) -> np.ndarray:
        """Model pseudo-covariance matrix calculation function (private) """
        covar_list = [self.transition_model.covar(**kwargs), np.zeros((3, 3))]
        return block_diag(*covar_list)
