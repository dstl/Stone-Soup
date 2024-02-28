import numpy as np
from scipy.stats import multivariate_normal as mvn

from stonesoup.base import Property
from stonesoup.functions.graph import get_xy_from_range_edge
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.graph import RoadNetwork


class ShortestPathToDestinationMeasurementModel(NonLinearGaussianMeasurement):
    r"""Shortest path to destination measurement model

    This is a measurement model that projects the target's position on the road network to a 2D
    position, via a non-linear transformation function :math:`h(x)`, which parameterizes the
    likelihood of the measurement given the target's state. This is a generalised implementation
    of the measurement model described in [1]_.

    The positional measurement noise is modelled as a zero-mean Gaussian distribution with
    covariance :math:`R`, such that:

    .. math::
        y_k = h(x_k) + v_k, v_k \sim \mathcal{N}(0, R)

    where the state vector :math:`x_k` is assumed to take the form:

    .. math::
        x_k = \left[r_k, \cdots, e_k, d_k, s_k\right]

    and :math:`e_k` denotes the edge the target is currently on, :math:`r_k` is the distance
    travelled along the edge, :m    ath:`d_k` is the destination node, and :math:`s_k` is the source
    node. The notation :math:`\cdots` denotes additional state variables that are not used in this
    model (e.g. velocity).

    The likelihood function is defined in either of two ways, depending on the value of
    :attr:`use_indicator`:

    - If :attr:`use_indicator` is `False`, then the likelihood function is defined as:

    .. math::
        p(y_k|x_k) = \mathcal{N}(y_k; h(x_k), R)

    - If :attr:`use_indicator` is `True`, then the likelihood function is defined as:

    .. math::

        p(y_k|x_k) = \begin{cases}\mathcal{N}(y_k; h(x_k), R), & \text{if } e_k \in \text{shortest_path}(s_k, d_k) \\
                    0 & \text{otherwise}\end{cases}

    where :math:`\text{shortest_path}(s_k, d_k)` is the shortest path between the source node
    :math:`s_k` and destination node :math:`d_k` on the road network. The indicator function
    implements Eq. (26) in [1]_.

    References
    ----------
    .. [1] L. Vladimirov and S. Maskell, "A SMC Sampler for Joint Tracking and Destination
           Estimation from Noisy Data," 2020 IEEE 23rd International Conference on Information
           Fusion (FUSION), Rustenburg, South Africa, 2020, pp. 1-8,
           doi: 10.23919/FUSION45008.2020.9190463.
    """

    graph: RoadNetwork = Property(
        doc="The road network that the target is moving on")
    use_indicator: bool = Property(
        default=True,
        doc="Whether to use the indicator function in the evaluation of the likelihood")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state, noise=False, **kwargs):

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Transform range and edge to xy
        r = state.state_vector[0, :]
        e = state.state_vector[-3, :]
        xy = get_xy_from_range_edge(r, e, self.graph)
        return xy + noise

    def logpdf(self, state1, state2, **kwargs):
        sv = self.function(state2, **kwargs)
        num_particles = sv.shape[1]
        likelihood = mvn.logpdf(
            sv.T,
            mean=state1.state_vector.ravel(),
            cov=self.covar(**kwargs)
        )
        if self.use_indicator:
            # If edge is not in the path, set likelihood to 0 (log(0)=-inf)
            e = state2.state_vector[-3, :]
            d = state2.state_vector[-2, :]
            s = state2.state_vector[-1, :]
            for i in range(num_particles):
                try:
                    path = self.graph.shortest_path(s[i], d[i], path_type='edge')[(s[i], d[i])]
                except KeyError:
                    # If no path exists, set likelihood to -inf
                    likelihood[i] = -np.inf
                    continue
                idx = np.where(path == e[i])[0]
                if len(idx) == 0:
                    likelihood[i] = -np.inf

        return likelihood
