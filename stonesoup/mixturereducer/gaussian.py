from scipy.linalg import pinv

from . import MixtureReducer
from ..types.state import GaussianState


class BasicConvexCombination(MixtureReducer):
    @staticmethod
    def merge_components(*components):
        """
        Merge two similar components

        Parameters
        ----------
        *components : :class:`~.GaussianState`
            Components to be merged

        Returns
        -------
        merged_component : :class:`~.GaussianState`
            Merged Gaussian component
        """
        inv_covs = [pinv(component.covar) for component in components]
        P = pinv(sum(inv_covs))
        x = P @ sum(inv_cov @ component.state_vector
                    for inv_cov, component in zip(inv_covs, components))

        new_component = GaussianState.from_state(next(iter(components)), x, P)
        return new_component
