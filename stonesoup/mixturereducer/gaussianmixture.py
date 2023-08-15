import uuid

import numpy as np
from ordered_set import OrderedSet
from scipy.spatial import KDTree

from ..base import Property
from .base import MixtureReducer
from ..types.state import TaggedWeightedGaussianState, WeightedGaussianState
from ..measures import SquaredMahalanobis
from operator import attrgetter
from scipy.linalg import pinv

from ..types.state import GaussianState


class GaussianMixtureReducer(MixtureReducer):
    """
    Gaussian Mixture Reducer class:

    Reduces the number of components in a Gaussian mixture to increase
    computational efficiency. See [1] for details.
    Achieved in three ways: pruning, merging, and truncating.
    Pruning is the act of removing low weight components from the mixture
    that fall below a pruning threshold.
    Merging is the act of combining similar components in the mixture
    that fall with a distance threshold into a single component.
    Truncating is the act of removing low weight components from the
    mixture so that the number of components in the mixture stays below a
    given threshold. Truncating is performed after the pruning and merging.

    References
    ----------
    [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """

    prune_threshold: float = Property(default=1e-9, doc='Mixture component weight '
                                      'threshold for pruning')
    merge_threshold: float = Property(default=16, doc='Squared Mahalanobis distance '
                                      'threshold for merging')
    max_number_components: int = Property(default=np.iinfo(np.int64).max,
                                          doc='Maximum number of components to keep '
                                              'in the Gaussian mixture')
    merging: bool = Property(default=True, doc='Flag for merging')
    pruning: bool = Property(default=True,
                             doc='Flag for pruning components whose weight is below '
                                 ':attr:`prune_threshold`')
    truncating: bool = Property(default=True,
                                doc='Flag for truncating components, keeping a maximum '
                                    'of :attr:`max_number_components` components')
    kdtree_max_distance: float = Property(
        default=None,
        doc="This defines the max Euclidean search distance for a kd-tree, "
            "used as part of the merge process as a coarse gate. Default "
            "`None` where tree isn't used and all components are checked "
            "against the merge threshold.")

    def reduce(self, components_list):
        """
        Reduce the components of Gaussian Mixture :class:`list`
        through pruning, merging, and truncating

        Parameters
        ----------
        components_list : :class:`~.list`
            The components of Gaussian Mixture

        Returns
        -------
        :class:`~.list`
            Reduced components

            """
        if len(components_list) > 0:
            if self.pruning:
                components_list = self.prune(components_list)
            if len(components_list) > 1 and self.merging:
                components_list = self.merge(components_list)
            if len(components_list) > self.max_number_components and self.truncating:
                components_list = self.truncate(components_list)
        return components_list

    def prune(self, components_list):
        """
        Pruning is the act of removing low weight components from the mixture
        that fall below a pruning threshold :attr:`prune_threshold`.

        Parameters
        ----------
        components_list : :class:`~.list`
            The components of Gaussian Mixture to be pruned

        Returns
        -------
        remaining_components : :class:`~.GaussianMixtureState`
             Components that remain after pruning

        """
        # Prune low weight components
        pruned_weight_sum = 0
        for component in components_list:
            if component.weight < self.prune_threshold:
                pruned_weight_sum += component.weight

        remaining_components = [component for component in components_list
                                if component.weight >= self.prune_threshold]
        # Distribute pruned weights across remaining components
        for component in remaining_components:
            component.weight += \
                pruned_weight_sum / len(remaining_components)
        return remaining_components

    def merge_components(self, component_1, component_2):
        """
        Merge two similar components

        Parameters
        ----------
        component_1 : :class:`~.WeightedGaussianState`
            First component to be merged
        component_2 : :class:`~.WeightedGaussianState`
            Second component to be merged

        Returns
        -------
        merged_component : :class:`~.WeightedGaussianState`
            Merged Gaussian component

        """
        weight_sum = component_1.weight + component_2.weight
        w1 = component_1.weight / weight_sum
        w2 = component_2.weight / weight_sum
        merged_mean = component_1.mean*w1 + component_2.mean*w2
        merged_covar = component_1.covar*w1 + component_2.covar*w2
        mu1_minus_m2 = component_1.mean - component_2.mean
        merged_covar = merged_covar + \
            mu1_minus_m2*mu1_minus_m2.T*w1*w2
        if weight_sum > 1:
            weight_sum = 1
        if isinstance(component_1, TaggedWeightedGaussianState):
            merged_component = TaggedWeightedGaussianState(
                state_vector=merged_mean,
                covar=merged_covar,
                weight=weight_sum,
                tag=component_1.tag,
                timestamp=component_1.timestamp
            )
        elif isinstance(component_1, WeightedGaussianState):
            merged_component = WeightedGaussianState(
                state_vector=merged_mean,
                covar=merged_covar,
                weight=weight_sum,
                timestamp=component_1.timestamp
            )

        return merged_component

    def merge(self, components_list):
        """
        Merging is the act of combining similar components in the mixture
        that fall with a distance threshold :attr:`merge_threshold` into
        a single component.

        Parameters
        ----------
        components_list : :class:`~.list`
            Components of the Gaussian Mixture to be merged

        Returns
        -------
        :class:`~.list`
            Merged components

        """
        if self.kdtree_max_distance is not None:
            tree = KDTree(
                np.vstack([component.state_vector[:, 0]
                           for component in components_list]))
        else:
            tree = None

        # Sort components by weight
        remaining_components = OrderedSet(sorted(
            components_list, key=attrgetter('weight')))

        merged_components = []
        final_merged_components = []
        measure = SquaredMahalanobis(state_covar_inv_cache_size=None)
        while remaining_components:
            # Get highest weighted component
            best_component = remaining_components.pop()

            # If kdtree_max_distance set, use this as gate
            if tree:
                indexes = tree.query_ball_point(
                    best_component.state_vector.ravel(),
                    r=self.kdtree_max_distance)
                matched_components = {components_list[i]
                                      for i in indexes
                                      if components_list[i] in remaining_components}
            else:
                # Modifying list in loop, so copy used
                matched_components = remaining_components.copy()

            # Check for similar components against threshold
            for component in matched_components:
                # Calculate distance between component and best component
                distance = measure(state1=component, state2=best_component)
                # Merge if similar
                if distance < self.merge_threshold:
                    remaining_components.remove(component)
                    best_component = self.merge_components(
                        best_component, component
                    )
            # Add potentially merged component to new mixture
            merged_components.append(best_component)
        if all(isinstance(component, TaggedWeightedGaussianState)
               for component in merged_components):
            # Check for duplicate tags
            components_tags = set(component.tag for component in merged_components)
            if len(components_tags) != len(merged_components):
                # There are duplicatze tags so assign
                # new tags to the lower weighted shared ones
                for shared_tag in components_tags:
                    shared_components = sorted(
                        (component for component in merged_components
                            if component.tag == shared_tag),
                        key=attrgetter('weight'),
                        reverse=True)
                    final_merged_components.append(shared_components[0])
                    for component in shared_components[1:]:
                        # Assign a new uuid
                        component.tag = str(uuid.uuid4())
                        final_merged_components.append(component)
            else:
                # No duplicates
                final_merged_components.extend(merged_components)
        else:
            # Just weighted components (no tags)
            final_merged_components.extend(merged_components)
        # Assign merged components to the mixture
        return final_merged_components

    def truncate(self, components_list):
        """
        Truncating is the act of removing low-weight components from the mixture
        so that the size of the mixture (number of components) stays within the given
        threshold :attr:`max_number_components`.

        Parameters
        ----------
        components_list : :class:`~.list`
            Components of the Gaussian Mixture to be truncated

        Returns
        -------
        :class:`~.list`
            The :attr:`max_number_components` components with the highest weights

        """

        # Sort components by weight from highest to lowest
        all_components = sorted(
            components_list, key=attrgetter('weight'), reverse=True)

        # Make list of truncated components. This function is called only when
        # len(components_list) > self.max_number_components, so the next line
        # will never give an index error
        truncated_components = all_components[self.max_number_components:]
        truncated_weight_sum = sum([component.weight for component in truncated_components])

        # Distribute truncated weights across remaining components
        remaining_components = all_components[:self.max_number_components]
        for component in remaining_components:
            component.weight += \
                truncated_weight_sum / self.max_number_components

        return remaining_components


class BasicConvexCombination(MixtureReducer):
    """
    Combine 'n' :class:`~.GaussianState`s using Convex Combination.

    .. math ::

        P = (\sum_{n} P_n^{-1})^{-1}
        \\\\
        x = P(\sum P_n^{-1}x^n)

    where :math:`\mathbf{x}_{n}` is the state of component `n`,
    :math:`\mathbf{P}_{n}` is the covariance matrix of component `n`,
    :math:`\mathbf{x}` is the combined state and
    :math:`\mathbf{P}` is the covariance matrix of the combined state

    """
    @staticmethod
    def merge_components(*components):
        """
        Merge two similar components

        Parameters
        ----------
        components : :class:`~.GaussianState`
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
